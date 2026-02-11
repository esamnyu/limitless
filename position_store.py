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

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from log_setup import get_logger

logger = get_logger(__name__)

__all__ = ["load_positions", "save_positions", "position_transaction", "register_position"]

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


@contextmanager
def _file_lock():
    """Acquire an exclusive file lock. Blocks until lock is available."""
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


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
            existing["avg_price"] = round((old_price * old_qty + price * quantity) / new_total, 1)
            existing["contracts"] = new_total
            existing["original_contracts"] = new_total
            existing["exit_rules"]["freeroll_at"] = int(existing["avg_price"] * 2)
            existing.setdefault("notes", []).append(
                f"{now.isoformat()}: Added {quantity}x @ {price}c (avg now {existing['avg_price']}c)"
            )
            logger.info(f"Updated existing position: {new_total}x @ {existing['avg_price']}c avg")
        else:
            position = {
                "ticker": ticker,
                "side": side,
                "avg_price": price,
                "contracts": quantity,
                "original_contracts": quantity,
                "order_id": order_id,
                "status": "open",
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
