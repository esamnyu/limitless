#!/usr/bin/env python3
"""
POSITION STORE — Atomic, locked position file operations.

Shared module used by execute_trade.py and position_monitor.py to safely
read/write positions.json without race conditions or corruption.

Features:
  - fcntl file locking (prevents concurrent writes from cron + manual runs)
  - Atomic writes (write to temp file, then os.rename)
  - Schema validation (catches corrupted or malformed entries)
"""

import errno
import fcntl
import json
import os
import signal
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from zoneinfo import ZoneInfo

from log_setup import get_logger

logger = get_logger(__name__)

__all__ = [
    "PositionDict", "LockTimeoutError", "LOCK_TIMEOUT_SEC",
    "load_positions", "save_positions",
    "position_transaction", "register_position",
]


class ExitRulesDict(TypedDict, total=False):
    """Exit rule thresholds attached to each position."""
    freeroll_at: int
    efficiency_exit: int
    trailing_offset: int


class PositionDict(TypedDict, total=False):
    """Canonical schema for a position in positions.json.

    Required fields are marked with total=True semantics in REQUIRED_KEYS.
    Optional fields (total=False) are set by position_monitor during lifecycle.
    """
    # ── Core (required) ──
    ticker: str
    side: str           # "yes" or "no"
    avg_price: float    # Entry price in cents
    contracts: int
    status: str         # "open", "resting", "pending_sell", "closed", "settled"

    # ── Lifecycle ──
    original_contracts: int
    order_id: str
    entry_time: str     # ISO 8601
    freerolled: bool
    peak_price: int
    trailing_floor: int
    pnl_realized: float
    exit_rules: ExitRulesDict
    notes: list[str]

    # ── Set by auto_trader confidence updates ──
    last_confidence: float
    bracket_low: float
    bracket_high: float
    current_obs_temp: float
    trend: str          # "running_hot", "running_cold", "on_track"

    # ── Sell tracking ──
    sell_placed_at: str  # ISO 8601
    sell_price: int

    # ── Averaging ──
    averaged_in: bool

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent
POSITIONS_FILE = PROJECT_ROOT / "positions.json"
LOCK_FILE = PROJECT_ROOT / ".positions.lock"

# Required keys for a valid position entry
REQUIRED_KEYS = {"ticker", "side", "avg_price", "contracts", "status"}


def _validate_position(pos: dict) -> bool:
    """Check that a position dict has all required keys."""
    if not isinstance(pos, dict):
        return False
    return REQUIRED_KEYS.issubset(pos.keys())


LOCK_TIMEOUT_SEC = 10  # Max seconds to wait for file lock


class LockTimeoutError(Exception):
    """Raised when file lock acquisition exceeds LOCK_TIMEOUT_SEC."""


def _alarm_handler(signum, frame):
    raise LockTimeoutError(f"File lock acquisition timed out after {LOCK_TIMEOUT_SEC}s")


@contextmanager
def _file_lock():
    """Acquire an exclusive file lock with timeout.

    If a stale lock from a crashed process blocks for longer than
    LOCK_TIMEOUT_SEC, raises LockTimeoutError instead of blocking forever.

    Recovery sequence on stale lock:
      1. Close the timed-out fd, remove the lock file
      2. Re-open and attempt non-blocking acquire (3 retries, 0.5s apart)
      3. If NB fails, fall back to blocking acquire with a fresh SIGALRM timeout
    """
    lock_fd = open(LOCK_FILE, "w")
    try:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(LOCK_TIMEOUT_SEC)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except LockTimeoutError:
            # Stale lock detected — force-break and retry
            logger.error(
                "Lock acquisition timed out after %ds — possible stale lock. "
                "Force-removing %s and retrying.",
                LOCK_TIMEOUT_SEC, LOCK_FILE,
            )
            lock_fd.close()
            try:
                LOCK_FILE.unlink(missing_ok=True)
            except OSError:
                pass

            # Retry with non-blocking attempts first
            acquired = False
            for attempt in range(3):
                lock_fd = open(LOCK_FILE, "w")
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    lock_fd.close()
                    if attempt < 2:
                        time.sleep(0.5)
                        logger.warning("Lock NB retry %d/3 failed, retrying...", attempt + 1)

            if not acquired:
                # Final fallback: blocking acquire with fresh timeout
                logger.warning("NB retries exhausted — blocking acquire with fresh timeout")
                lock_fd = open(LOCK_FILE, "w")
                signal.alarm(LOCK_TIMEOUT_SEC)
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except LockTimeoutError:
                    lock_fd.close()
                    raise LockTimeoutError(
                        f"Lock unrecoverable after stale-lock removal and {LOCK_TIMEOUT_SEC}s retry"
                    )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        yield
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except (ValueError, OSError):
            pass  # fd already closed in error path
        try:
            lock_fd.close()
        except (ValueError, OSError):
            pass  # already closed


def _read_positions_unlocked() -> list[dict]:
    """Read and validate positions file. Caller MUST hold _file_lock."""
    if not POSITIONS_FILE.exists():
        return []
    try:
        raw = POSITIONS_FILE.read_text().strip()
        if not raw:
            return []
        positions = json.loads(raw)
        if not isinstance(positions, list):
            logger.warning(f"positions.json is not a list, got {type(positions).__name__}")
            return []
        valid = []
        for i, p in enumerate(positions):
            if _validate_position(p):
                valid.append(p)
            else:
                logger.warning(f"Skipping invalid position entry at index {i}: missing keys")
        if len(valid) < len(positions):
            logger.warning(f"Filtered {len(positions) - len(valid)} invalid position entries")
        return valid
    except json.JSONDecodeError as e:
        logger.error(f"positions.json is corrupted: {e}")
        backup = POSITIONS_FILE.with_suffix(f".corrupted.{int(datetime.now().timestamp())}")
        POSITIONS_FILE.rename(backup)
        logger.error(f"Corrupted file saved as {backup}")
        return []
    except Exception as e:
        logger.error(f"Failed to read positions.json: {e}")
        return []


def _write_positions_unlocked(positions: list[dict]):
    """Atomically write positions file. Caller MUST hold _file_lock."""
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=POSITIONS_FILE.parent,
            prefix=".positions_",
            suffix=".tmp",
        )
        with os.fdopen(fd, "w") as f:
            json.dump(positions, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, POSITIONS_FILE)
        logger.debug(f"Saved {len(positions)} positions atomically")
    except Exception as e:
        logger.error(f"Failed to save positions: {e}")
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_positions() -> list[dict]:
    """
    Load positions from JSON file with locking and validation.

    Returns empty list if file doesn't exist, is empty, or contains invalid JSON.
    Invalid entries are filtered out with a warning.
    """
    with _file_lock():
        return _read_positions_unlocked()


def save_positions(positions: list[dict]):
    """
    Atomically save positions to JSON file with locking.

    Writes to a temp file first, then does an atomic rename.
    This prevents corruption if the process is killed mid-write.
    """
    with _file_lock():
        _write_positions_unlocked(positions)


@contextmanager
def position_transaction():
    """
    Transactional read-modify-write with a SINGLE lock held throughout.

    Usage:
        with position_transaction() as positions:
            # positions is a mutable list — modify in place
            for p in positions:
                if p["ticker"] == ticker:
                    p["status"] = "closed"
            # Automatically saved on context exit (if no exception)

    This prevents the race condition where two processes both read stale data
    between separate load_positions() and save_positions() calls.
    """
    with _file_lock():
        positions = _read_positions_unlocked()
        yield positions
        _write_positions_unlocked(positions)


def register_position(
    ticker: str,
    side: str,
    price: int,
    quantity: int,
    order_id: str,
    status: str,
):
    """
    Register a new position (or average into existing) in positions.json.

    Called by execute_trade.py after a successful order placement.
    Uses a single lock transaction to prevent race conditions.
    """
    with position_transaction() as positions:
        # Check if position already exists for this ticker (average in)
        existing = None
        for p in positions:
            if p["ticker"] == ticker and p["side"] == side and p["status"] == "open":
                existing = p
                break

        now = datetime.now(ET)
        freeroll_at = int(price * 2)

        if existing:
            old_qty = existing["contracts"]
            old_price = existing["avg_price"]
            new_total = old_qty + quantity

            # ── GUARD: prevent division by zero if quantities cancel out ──
            if new_total <= 0:
                raise ValueError(
                    f"AVERAGING REJECTED on {ticker}: new_total={new_total} "
                    f"(old={old_qty} + new={quantity}) would be non-positive"
                )

            # ── GUARD: validate existing position hasn't been corrupted ──
            if old_qty <= 0 or old_price <= 0:
                raise ValueError(
                    f"AVERAGING REJECTED on {ticker}: existing position has "
                    f"invalid data (qty={old_qty}, price={old_price})"
                )

            new_avg = round((old_price * old_qty + price * quantity) / new_total, 1)

            # ── AVERAGING-IN WARNING ──
            # This doubles down on an existing position. Log prominently.
            direction = "DOWN" if price < old_price else "UP"
            logger.warning(
                "AVERAGING %s on %s: %dx@%dc → %dx@%.1fc (was %dx@%.1fc)",
                direction, ticker, quantity, price, new_total, new_avg, old_qty, old_price
            )

            existing["avg_price"] = new_avg
            existing["contracts"] = new_total
            existing["original_contracts"] = new_total
            existing["averaged_in"] = True  # Flag for position monitor to track
            existing.setdefault("exit_rules", {})["freeroll_at"] = int(new_avg * 2)
            existing.setdefault("notes", []).append(
                f"{now.isoformat()}: ⚠ AVERAGED {direction} — added {quantity}x @ {price}c (avg now {new_avg}c)"
            )
            logger.info(f"Updated existing position: {new_total}x @ {new_avg}c avg")
        else:
            # Map Kalshi order status to position status
            if status.upper() in ("RESTING", "PENDING"):
                pos_status = "resting"
            else:
                pos_status = "open"

            position = {
                "ticker": ticker,
                "side": side,
                "avg_price": price,
                "contracts": quantity,
                "original_contracts": quantity,
                "order_id": order_id,
                "status": pos_status,
                "entry_time": now.isoformat(),
                "freerolled": False,
                "peak_price": price,
                "trailing_floor": 0,
                "pnl_realized": 0.0,
                "exit_rules": {
                    "freeroll_at": freeroll_at,
                    "efficiency_exit": 90,
                    "trailing_offset": 8,
                },
                "notes": [
                    f"{now.isoformat()}: Opened {quantity}x {side.upper()} @ {price}c "
                    f"(order: {order_id}, status: {status})"
                ],
            }
            positions.append(position)
            logger.info(f"Position registered: {quantity}x {side.upper()} {ticker} @ {price}c (freeroll at {freeroll_at}c)")
