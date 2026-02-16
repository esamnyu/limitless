"""
STALE PRICE DETECTOR — Ensemble shift tracking between scans.

Compares the current scan's ensemble mean per city against the previous
scan.  If the ensemble has shifted by ≥ STALE_PRICE_MIN_SHIFT_F but
the market bid on the affected bracket hasn't repriced by at least
STALE_PRICE_MIN_GAP_CENTS, that's a stale-price opportunity.

State persistence:
  Saves {city: {mean, bracket_bids}} after each scan to a JSON file.
  Next scan loads the previous state and computes deltas.

Usage:
  Called from auto_scan.py after each city scan completes.
  Returns a list of StaleAlert objects for Discord notification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from config import (
    STALE_PRICE_ENABLED,
    STALE_PRICE_MIN_SHIFT_F,
    STALE_PRICE_MIN_GAP_CENTS,
    STALE_PRICE_STATE_FILE,
)
from log_setup import get_logger

logger = get_logger(__name__)

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent
STATE_PATH = PROJECT_ROOT / STALE_PRICE_STATE_FILE


@dataclass
class StaleAlert:
    """A single stale-price detection."""
    city: str
    direction: str              # "warmer" or "cooler"
    mean_shift_f: float         # How far the ensemble moved (°F)
    prev_mean: float
    curr_mean: float
    bracket_title: str          # The bracket that should have repriced
    ticker: str
    expected_bid_change: int    # How much the bid should have moved (approx)
    actual_bid: int             # Current market bid
    prev_bid: int               # Previous scan's bid


@dataclass
class ScanSnapshot:
    """Per-city snapshot from one scan."""
    mean: float
    std: float
    timestamp: str              # ISO timestamp
    bracket_bids: dict          # {ticker: {"bid": int, "title": str}}

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "timestamp": self.timestamp,
            "bracket_bids": self.bracket_bids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScanSnapshot:
        return cls(
            mean=d.get("mean", 0),
            std=d.get("std", 0),
            timestamp=d.get("timestamp", ""),
            bracket_bids=d.get("bracket_bids", {}),
        )


def load_previous_state() -> dict[str, ScanSnapshot]:
    """Load previous scan snapshots from disk."""
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH) as f:
            data = json.load(f)
        return {k: ScanSnapshot.from_dict(v) for k, v in data.items()}
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Stale price state corrupt, resetting: %s", e)
        return {}


def save_current_state(states: dict[str, ScanSnapshot]) -> None:
    """Save current scan state for next comparison."""
    data = {k: v.to_dict() for k, v in states.items()}
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(STATE_PATH)


def build_snapshot(
    city_key: str,
    ensemble_mean: float,
    ensemble_std: float,
    brackets: list[dict],
) -> ScanSnapshot:
    """Build a snapshot from current scan data.

    Parameters
    ----------
    brackets : list[dict]
        Raw Kalshi market dicts with 'ticker', 'title', 'yes_bid'.
    """
    bracket_bids = {}
    for mkt in brackets:
        ticker = mkt.get("ticker", "")
        if not ticker:
            continue
        bracket_bids[ticker] = {
            "bid": mkt.get("yes_bid", 0),
            "title": mkt.get("title", "") or mkt.get("subtitle", ""),
        }
    return ScanSnapshot(
        mean=ensemble_mean,
        std=ensemble_std,
        timestamp=datetime.now(ET).isoformat(),
        bracket_bids=bracket_bids,
    )


def detect_stale_prices(
    city_key: str,
    current: ScanSnapshot,
    previous: ScanSnapshot | None,
) -> list[StaleAlert]:
    """Compare current vs previous scan to find stale-priced brackets.

    Returns list of StaleAlert for brackets where:
    1. Ensemble mean shifted by >= STALE_PRICE_MIN_SHIFT_F
    2. Market bid barely moved (< expected repricing)
    """
    if not STALE_PRICE_ENABLED:
        return []
    if previous is None:
        return []
    if current.mean == 0 or previous.mean == 0:
        return []

    mean_shift = current.mean - previous.mean
    abs_shift = abs(mean_shift)

    if abs_shift < STALE_PRICE_MIN_SHIFT_F:
        return []

    direction = "warmer" if mean_shift > 0 else "cooler"
    alerts = []

    # For each bracket in current scan, check if its bid moved appropriately
    for ticker, curr_data in current.bracket_bids.items():
        prev_data = previous.bracket_bids.get(ticker)
        if not prev_data:
            continue

        curr_bid = curr_data.get("bid", 0)
        prev_bid = prev_data.get("bid", 0)

        if curr_bid == 0 or prev_bid == 0:
            continue

        bid_change = curr_bid - prev_bid

        # If ensemble shifted warmer (higher mean), brackets above the old mean
        # should have gotten more expensive (higher bid). If they didn't move,
        # that's stale pricing. Similarly for cooler shifts.
        #
        # Expected repricing: rough estimate is ~1¢ per 0.5°F shift for
        # brackets near the mean. We flag if bid barely moved vs expected.
        expected_change = int(abs_shift * 2)  # ~2¢ per °F shift (conservative)

        # Check for stale brackets:
        # If warmer → brackets that should have gotten more expensive but didn't
        # If cooler → brackets that should have gotten cheaper but didn't
        if direction == "warmer" and bid_change < expected_change - STALE_PRICE_MIN_GAP_CENTS:
            # Bracket didn't get more expensive enough — stale if it should have
            # Only flag brackets currently priced mid-range (15-85¢) where
            # there's actual trading opportunity
            if 15 <= curr_bid <= 85:
                alerts.append(StaleAlert(
                    city=city_key,
                    direction=direction,
                    mean_shift_f=round(mean_shift, 1),
                    prev_mean=round(previous.mean, 1),
                    curr_mean=round(current.mean, 1),
                    bracket_title=curr_data.get("title", ticker),
                    ticker=ticker,
                    expected_bid_change=expected_change,
                    actual_bid=curr_bid,
                    prev_bid=prev_bid,
                ))
        elif direction == "cooler" and bid_change > -(expected_change - STALE_PRICE_MIN_GAP_CENTS):
            # Bracket didn't get cheaper enough — stale
            if 15 <= curr_bid <= 85:
                alerts.append(StaleAlert(
                    city=city_key,
                    direction=direction,
                    mean_shift_f=round(mean_shift, 1),
                    prev_mean=round(previous.mean, 1),
                    curr_mean=round(current.mean, 1),
                    bracket_title=curr_data.get("title", ticker),
                    ticker=ticker,
                    expected_bid_change=expected_change,
                    actual_bid=curr_bid,
                    prev_bid=prev_bid,
                ))

    return alerts


def format_stale_alerts(alerts: list[StaleAlert]) -> str:
    """Format stale alerts into a human-readable Discord message."""
    if not alerts:
        return ""
    lines = [f"**📊 STALE PRICE ALERT — {len(alerts)} bracket(s)**\n"]
    for a in alerts[:5]:  # Cap at 5 per message
        shift_icon = "🔴" if a.direction == "warmer" else "🔵"
        from edge_scanner_v2 import shorten_bracket_title
        short = shorten_bracket_title(a.bracket_title)
        lines.append(
            f"{shift_icon} **{a.city}** {short}: ensemble shifted {a.mean_shift_f:+.1f}°F "
            f"({a.prev_mean:.1f}→{a.curr_mean:.1f}) but bid only moved "
            f"{a.actual_bid - a.prev_bid:+d}¢ ({a.prev_bid}→{a.actual_bid}¢)\n"
            f"   Ticker: `{a.ticker}`"
        )
    return "\n".join(lines)
