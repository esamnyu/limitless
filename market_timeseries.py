#!/usr/bin/env python3
"""
MARKET TIMESERIES — Intraday orderbook snapshot collector and price series analyzer.

Captures Kalshi weather market orderbook snapshots throughout the day to build
price time series data. Enables:
  - Mean-reversion detection after DSM/6-hour bot repricing events
  - Optimal entry timing window discovery
  - Price autocorrelation analysis (momentum vs mean-reversion)
  - Volatility profiling by time of day

Storage: backtest/market_snapshots/YYYY-MM-DD_CITY.jsonl (append-only)

Usage:
    python3 market_timeseries.py --capture               # one snapshot
    python3 market_timeseries.py --watch --interval 15    # continuous
    python3 market_timeseries.py --analyze --city NYC --date 2026-02-14
    python3 market_timeseries.py --report --days 7
    python3 market_timeseries.py --bot-events --city NYC --date 2026-02-14
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from log_setup import get_logger
from config import STATIONS, SETTLEMENT_HOUR_ET

logger = get_logger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
SNAPSHOT_DIR = PROJECT_ROOT / "backtest" / "market_snapshots"

# ── Timezone constants ─────────────────────────────────────────────────────────

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ── All city codes ─────────────────────────────────────────────────────────────

ALL_CITIES = list(STATIONS.keys())

# ── Time-of-day bucket definitions ─────────────────────────────────────────────

TIME_BUCKETS = [
    ("06-08", 6, 8),
    ("08-10", 8, 10),
    ("10-12", 10, 12),
    ("12-14", 12, 14),
    ("14-16", 14, 16),
    ("16-18", 16, 18),
    ("18-20", 18, 20),
    ("20-22", 20, 22),
    ("22-00", 22, 24),
]


# =============================================================================
# BRACKET PARSING (replicates edge_scanner_v2 logic for standalone use)
# =============================================================================

def parse_bracket_from_title(title: str) -> str:
    """Extract human-readable bracket string from Kalshi market title.

    Returns strings like '36-37', '<30', '>50', or 'unknown'.
    """
    clean = title.replace("\u00b0F", "").replace("\u00b0", "").replace("*", "").strip()
    if re.search(r"below|under|or less|<", clean, re.I):
        nums = re.findall(r"([\d.]+)", clean)
        if nums:
            return f"<{nums[0]}"
    if re.search(r"above|or more|or higher|>", clean, re.I):
        nums = re.findall(r"([\d.]+)", clean)
        if nums:
            return f">{nums[0]}"
    match = re.search(r"([\d.]+)\s*(?:to|-)\s*([\d.]+)", clean)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return "unknown"


def extract_target_date_from_ticker(ticker: str) -> Optional[str]:
    """Extract the target date from a Kalshi ticker like KXHIGHNY-26FEB15-B38.5.

    Returns ISO date string (YYYY-MM-DD) or None.
    """
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
    if not match:
        return None
    yy = int(match.group(1))
    mon_str = match.group(2)
    dd = int(match.group(3))
    mon = months.get(mon_str)
    if mon is None:
        return None
    year = 2000 + yy
    try:
        return date(year, mon, dd).isoformat()
    except ValueError:
        return None


# =============================================================================
# SNAPSHOT CAPTURE
# =============================================================================

def _snapshot_path(date_str: str, city: str) -> Path:
    """Return the JSONL file path for a given date and city."""
    return SNAPSHOT_DIR / f"{date_str}_{city}.jsonl"


def _determine_target_date() -> date:
    """Determine which date's markets to capture.

    Before settlement (7 AM ET), the active markets are for today.
    After settlement, they are for tomorrow.
    """
    now_et = datetime.now(ET)
    if now_et.hour < SETTLEMENT_HOUR_ET:
        return now_et.date()
    return (now_et + timedelta(days=1)).date()


def _hours_to_settlement(target: date) -> float:
    """Compute hours from now until settlement (7 AM ET on target date)."""
    settlement = datetime.combine(target, time(hour=SETTLEMENT_HOUR_ET), tzinfo=ET)
    delta = settlement - datetime.now(ET)
    return max(0.0, delta.total_seconds() / 3600)


async def capture_snapshot(cities: Optional[List[str]] = None) -> List[dict]:
    """Capture orderbook snapshots for all open markets across specified cities.

    Parameters
    ----------
    cities : list of str, optional
        City codes to capture (default: all 5 cities).

    Returns
    -------
    list of dict
        One snapshot dict per city successfully captured.
    """
    from kalshi_client import KalshiClient

    city_list = cities or ALL_CITIES
    target = _determine_target_date()
    target_str = target.isoformat()
    hrs = _hours_to_settlement(target)

    now_et = datetime.now(ET)
    now_utc = datetime.now(UTC)
    ts_et = now_et.isoformat()
    ts_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ensure output directory exists
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize Kalshi client
    api_key = os.getenv("KALSHI_API_KEY_ID", "")
    pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    client = KalshiClient(api_key_id=api_key, private_key_path=pk_path, demo_mode=False)
    snapshots = []

    try:
        await client.start()

        for city_code in city_list:
            city_code = city_code.upper()
            if city_code not in STATIONS:
                logger.warning("Unknown city code: %s, skipping", city_code)
                continue

            station = STATIONS[city_code]
            try:
                markets = await client.get_markets(
                    series_ticker=station.series_ticker, status="open", limit=100
                )
                if not markets:
                    logger.debug("No open markets for %s", city_code)
                    continue

                brackets: Dict[str, dict] = {}
                for mkt in markets:
                    ticker = mkt.get("ticker", "")
                    if not ticker:
                        continue

                    # Only capture markets for the target date
                    mkt_date = extract_target_date_from_ticker(ticker)
                    if mkt_date and mkt_date != target_str:
                        continue

                    # Fetch orderbook for top-of-book bid/ask
                    orderbook = await client.get_orderbook(ticker, depth=1)
                    yes_bids = orderbook.get("yes", []) if isinstance(orderbook.get("yes"), list) else []
                    no_bids = orderbook.get("no", []) if isinstance(orderbook.get("no"), list) else []

                    # Extract best bid/ask from orderbook
                    # yes side: bids are buy-yes, asks are derived from no-bids
                    yes_bid = 0
                    yes_ask = 0

                    if yes_bids:
                        # yes_bids is list of [price, quantity] — highest is best bid
                        yes_bid = max(entry[0] for entry in yes_bids) if yes_bids else 0
                    if no_bids:
                        # no_bids: best no bid at price P means yes ask at (100 - P)
                        best_no_bid = max(entry[0] for entry in no_bids) if no_bids else 0
                        yes_ask = 100 - best_no_bid if best_no_bid > 0 else 0

                    # Fallback to market-level fields if orderbook was empty
                    if yes_bid == 0:
                        yes_bid = mkt.get("yes_bid", 0)
                    if yes_ask == 0:
                        yes_ask = mkt.get("yes_ask", 0)

                    title = mkt.get("title", "") or mkt.get("subtitle", "")
                    bracket_label = parse_bracket_from_title(title)
                    volume = mkt.get("volume", 0)

                    brackets[ticker] = {
                        "yes_bid": yes_bid,
                        "yes_ask": yes_ask,
                        "volume": volume,
                        "bracket": bracket_label,
                    }

                if not brackets:
                    logger.debug("No target-date brackets found for %s", city_code)
                    continue

                snapshot = {
                    "ts": ts_et,
                    "ts_utc": ts_utc,
                    "city": city_code,
                    "target_date": target_str,
                    "hours_to_settlement": round(hrs, 1),
                    "brackets": brackets,
                }

                # Append to JSONL file
                out_path = _snapshot_path(target_str, city_code)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(snapshot, separators=(",", ":")) + "\n")

                snapshots.append(snapshot)
                logger.info(
                    "Snapshot captured: %s — %d brackets, %.1fh to settlement",
                    city_code, len(brackets), hrs,
                )

            except Exception as e:
                logger.warning("Snapshot capture failed for %s: %s", city_code, e)
                continue

    finally:
        await client.stop()

    return snapshots


async def capture_loop(interval_minutes: int = 15, cities: Optional[List[str]] = None) -> None:
    """Run capture_snapshot on a recurring interval.

    Parameters
    ----------
    interval_minutes : int
        Minutes between captures (default: 15).
    cities : list of str, optional
        City codes to capture.
    """
    logger.info(
        "Starting capture loop: interval=%dm, cities=%s",
        interval_minutes, cities or "ALL",
    )
    cycle = 0
    try:
        while True:
            cycle += 1
            logger.info("Capture cycle %d starting", cycle)
            try:
                results = await capture_snapshot(cities)
                logger.info(
                    "Cycle %d complete: %d cities captured", cycle, len(results),
                )
            except Exception as e:
                logger.error("Capture cycle %d failed: %s", cycle, e)

            await asyncio.sleep(interval_minutes * 60)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Capture loop stopped after %d cycles", cycle)


# =============================================================================
# SNAPSHOT LOADING
# =============================================================================

def load_snapshots(city: str, date_str: str) -> List[dict]:
    """Load all snapshots for a city/date from the JSONL file.

    Parameters
    ----------
    city : str
        City code (e.g., 'NYC').
    date_str : str
        ISO date string (e.g., '2026-02-14').

    Returns
    -------
    list of dict
        Snapshots sorted by timestamp.
    """
    path = _snapshot_path(date_str, city.upper())
    if not path.exists():
        logger.debug("No snapshot file: %s", path)
        return []

    snapshots = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                snapshots.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Malformed JSON on line %d of %s: %s", line_num, path.name, e)

    # Sort by UTC timestamp for consistent ordering
    snapshots.sort(key=lambda s: s.get("ts_utc", ""))
    return snapshots


def _available_dates(city: str) -> List[str]:
    """List available snapshot dates for a city, sorted ascending."""
    if not SNAPSHOT_DIR.exists():
        return []
    dates = []
    for p in SNAPSHOT_DIR.glob(f"*_{city.upper()}.jsonl"):
        # filename: YYYY-MM-DD_CITY.jsonl
        date_part = p.stem.rsplit("_", 1)[0]
        dates.append(date_part)
    dates.sort()
    return dates


# =============================================================================
# PRICE SERIES ANALYSIS
# =============================================================================

def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation between two equal-length lists.

    Returns 0.0 if insufficient data or zero variance.
    """
    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0

    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if denom_x == 0 or denom_y == 0:
        return 0.0
    return num / (denom_x * denom_y)


def _midpoint(bid: int, ask: int) -> float:
    """Compute midpoint price. Falls back to whichever is non-zero."""
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if bid > 0:
        return float(bid)
    if ask > 0:
        return float(ask)
    return 0.0


def analyze_price_series(
    city: str, date_str: str, ticker: Optional[str] = None,
) -> Dict[str, dict]:
    """Analyze price time series from snapshots.

    Parameters
    ----------
    city : str
        City code.
    date_str : str
        ISO date string.
    ticker : str, optional
        Specific ticker to analyze. If None, analyzes all brackets.

    Returns
    -------
    dict
        Keyed by ticker, each value contains:
        - price_changes: list of (timestamp, delta)
        - volatility: std of price changes
        - autocorrelation_lag1: Pearson corr of consecutive changes
        - max_drawdown: largest peak-to-trough drop
        - total_range: max price - min price
        - final_price: last observed midpoint
        - n_snapshots: number of data points
    """
    snapshots = load_snapshots(city, date_str)
    if not snapshots:
        return {}

    # Collect price series per ticker
    # {ticker: [(ts, midpoint), ...]}
    series: Dict[str, List[Tuple[str, float]]] = {}

    for snap in snapshots:
        ts = snap.get("ts_utc", snap.get("ts", ""))
        for t, bkt in snap.get("brackets", {}).items():
            if ticker and t != ticker:
                continue
            mid = _midpoint(bkt.get("yes_bid", 0), bkt.get("yes_ask", 0))
            if mid <= 0:
                continue
            series.setdefault(t, []).append((ts, mid))

    results: Dict[str, dict] = {}

    for t, points in series.items():
        if len(points) < 2:
            results[t] = {
                "price_changes": [],
                "volatility": 0.0,
                "autocorrelation_lag1": 0.0,
                "max_drawdown": 0.0,
                "total_range": 0.0,
                "final_price": points[-1][1] if points else 0.0,
                "n_snapshots": len(points),
            }
            continue

        # Compute price changes
        changes: List[Tuple[str, float]] = []
        deltas: List[float] = []
        for i in range(1, len(points)):
            delta = points[i][1] - points[i - 1][1]
            changes.append((points[i][0], delta))
            deltas.append(delta)

        # Volatility (stdev of changes)
        vol = statistics.stdev(deltas) if len(deltas) >= 2 else 0.0

        # Autocorrelation lag-1
        autocorr = 0.0
        if len(deltas) >= 4:
            autocorr = _pearson_corr(deltas[:-1], deltas[1:])

        # Max drawdown
        prices = [p[1] for p in points]
        peak = prices[0]
        max_dd = 0.0
        for price in prices:
            if price > peak:
                peak = price
            dd = peak - price
            if dd > max_dd:
                max_dd = dd

        total_range = max(prices) - min(prices)

        results[t] = {
            "price_changes": changes,
            "volatility": round(vol, 2),
            "autocorrelation_lag1": round(autocorr, 3),
            "max_drawdown": round(max_dd, 1),
            "total_range": round(total_range, 1),
            "final_price": round(points[-1][1], 1),
            "n_snapshots": len(points),
        }

    return results


# =============================================================================
# BOT EVENT DETECTION
# =============================================================================

def detect_bot_events(city: str, date_str: str) -> List[dict]:
    """Detect significant price movements around DSM/6-hour release times.

    Parameters
    ----------
    city : str
        City code.
    date_str : str
        ISO date string.

    Returns
    -------
    list of dict
        Each dict: {event_time_z, event_type, price_changes, max_move}
    """
    city = city.upper()
    snapshots = load_snapshots(city, date_str)
    if len(snapshots) < 2:
        return []

    station = STATIONS.get(city)
    if station is None:
        logger.warning("Unknown city for bot event detection: %s", city)
        return []

    # Collect all release times
    release_times: List[Tuple[str, str]] = []  # (HH:MM, type)
    for t_str in station.dsm_times_z:
        release_times.append((t_str, "dsm"))
    for t_str in station.six_hour_z:
        release_times.append((t_str, "6hour"))

    # Parse snapshot UTC times into minutes-of-day for comparison
    def utc_minutes(ts_utc: str) -> int:
        """Parse a UTC timestamp string into minutes since midnight."""
        # Handle both "2026-02-14T20:30:00Z" and ISO formats
        try:
            dt = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        except ValueError:
            return -1
        return dt.hour * 60 + dt.minute

    snap_minutes = [(utc_minutes(s.get("ts_utc", "")), s) for s in snapshots]
    snap_minutes = [(m, s) for m, s in snap_minutes if m >= 0]

    if not snap_minutes:
        return []

    events: List[dict] = []

    for release_str, event_type in release_times:
        parts = release_str.split(":")
        if len(parts) != 2:
            continue
        release_min = int(parts[0]) * 60 + int(parts[1])

        # Find closest snapshot before and after the release
        before_snap = None
        after_snap = None
        best_before_diff = float("inf")
        best_after_diff = float("inf")

        for snap_min, snap in snap_minutes:
            diff = release_min - snap_min
            # Handle midnight wrap: if release is at 05:17 (317 min)
            # and snap is at 23:50 (1430 min), diff = 317 - 1430 = -1113
            # Adjust: diff += 1440 => 327 (snap is 327 min before release)
            if diff < -720:
                diff += 1440
            elif diff > 720:
                diff -= 1440

            if diff > 0 and diff < best_before_diff:
                best_before_diff = diff
                before_snap = snap
            elif diff <= 0 and abs(diff) < best_after_diff:
                best_after_diff = abs(diff)
                after_snap = snap

        if before_snap is None or after_snap is None:
            continue

        # Compare prices across the event
        price_changes: Dict[str, float] = {}
        before_brackets = before_snap.get("brackets", {})
        after_brackets = after_snap.get("brackets", {})

        all_tickers = set(before_brackets.keys()) | set(after_brackets.keys())
        for t in all_tickers:
            b_data = before_brackets.get(t, {})
            a_data = after_brackets.get(t, {})
            b_mid = _midpoint(b_data.get("yes_bid", 0), b_data.get("yes_ask", 0))
            a_mid = _midpoint(a_data.get("yes_bid", 0), a_data.get("yes_ask", 0))
            if b_mid > 0 and a_mid > 0:
                price_changes[t] = round(a_mid - b_mid, 1)

        if not price_changes:
            continue

        max_move = max(abs(v) for v in price_changes.values())

        event = {
            "event_time_z": release_str,
            "event_type": event_type,
            "price_changes": price_changes,
            "max_move": round(max_move, 1),
            "significant": max_move > 5,
        }
        events.append(event)

    return events


# =============================================================================
# OPTIMAL ENTRY WINDOWS
# =============================================================================

def _hour_bucket(hour: int) -> Optional[str]:
    """Map an hour (0-23) to a time bucket label."""
    for label, start, end in TIME_BUCKETS:
        if end == 24:
            if start <= hour < 24:
                return label
        elif start <= hour < end:
            return label
    return None


def optimal_entry_windows(
    city: str, dates: Optional[List[str]] = None,
) -> List[dict]:
    """Analyze multiple days of snapshots to find optimal entry windows.

    Parameters
    ----------
    city : str
        City code.
    dates : list of str, optional
        List of date strings to analyze. If None, uses all available dates.

    Returns
    -------
    list of dict
        Sorted by entry quality (best first). Each dict:
        - window: time bucket label
        - avg_volatility: average price volatility in this window
        - avg_abs_movement: average absolute price movement
        - price_bias: 'low' (below mean) or 'high' (above mean) on average
        - bot_event_frequency: fraction of days with bot events in this window
        - entry_quality: composite score (higher = better entry)
        - n_observations: number of data points
    """
    city = city.upper()
    if dates is None:
        dates = _available_dates(city)

    if not dates:
        return []

    # Collect per-bucket stats across all days
    # {bucket: {"volatilities": [...], "abs_moves": [...], "price_deviations": [...], "bot_events": int, "days_seen": int}}
    bucket_stats: Dict[str, dict] = {}
    for label, _, _ in TIME_BUCKETS:
        bucket_stats[label] = {
            "volatilities": [],
            "abs_moves": [],
            "price_deviations": [],  # deviation from day's mean price
            "bot_events": 0,
            "days_seen": 0,
        }

    for date_str in dates:
        snapshots = load_snapshots(city, date_str)
        if len(snapshots) < 2:
            continue

        # Compute daily mean price per ticker
        daily_prices: Dict[str, List[float]] = {}
        for snap in snapshots:
            for t, bkt in snap.get("brackets", {}).items():
                mid = _midpoint(bkt.get("yes_bid", 0), bkt.get("yes_ask", 0))
                if mid > 0:
                    daily_prices.setdefault(t, []).append(mid)

        daily_means: Dict[str, float] = {}
        for t, prices in daily_prices.items():
            if prices:
                daily_means[t] = statistics.mean(prices)

        # Get bot events for this day
        bot_events = detect_bot_events(city, date_str)
        bot_event_minutes: List[int] = []
        for ev in bot_events:
            if ev.get("significant"):
                parts = ev["event_time_z"].split(":")
                if len(parts) == 2:
                    bot_event_minutes.append(int(parts[0]) * 60 + int(parts[1]))

        # Bucket snapshots by time-of-day (using ET time)
        seen_buckets: set = set()
        prev_mids: Dict[str, float] = {}

        for snap in snapshots:
            ts_str = snap.get("ts", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            hour = ts.hour
            bucket = _hour_bucket(hour)
            if bucket is None:
                continue

            seen_buckets.add(bucket)

            for t, bkt in snap.get("brackets", {}).items():
                mid = _midpoint(bkt.get("yes_bid", 0), bkt.get("yes_ask", 0))
                if mid <= 0:
                    continue

                # Absolute movement from previous snapshot of same ticker
                if t in prev_mids:
                    move = abs(mid - prev_mids[t])
                    bucket_stats[bucket]["abs_moves"].append(move)
                    bucket_stats[bucket]["volatilities"].append(move)
                prev_mids[t] = mid

                # Price deviation from daily mean
                if t in daily_means:
                    dev = mid - daily_means[t]
                    bucket_stats[bucket]["price_deviations"].append(dev)

        # Count days per bucket
        for b in seen_buckets:
            bucket_stats[b]["days_seen"] += 1

        # Check which buckets had bot events
        for bot_min in bot_event_minutes:
            # Convert UTC minutes to approximate ET (subtract 5h = 300 min, rough)
            # Use station timezone for accuracy
            tz = ZoneInfo(STATIONS[city].timezone)
            try:
                utc_dt = datetime(2000, 1, 1, bot_min // 60, bot_min % 60, tzinfo=UTC)
                local_dt = utc_dt.astimezone(tz)
                local_hour = local_dt.hour
            except (ValueError, OverflowError):
                continue
            b = _hour_bucket(local_hour)
            if b is not None:
                bucket_stats[b]["bot_events"] += 1

    # Compute summary stats and entry quality
    results: List[dict] = []

    for label, _, _ in TIME_BUCKETS:
        stats = bucket_stats[label]
        n_obs = len(stats["abs_moves"])
        days = stats["days_seen"]
        if n_obs < 2 or days < 1:
            continue

        avg_vol = statistics.mean(stats["volatilities"]) if stats["volatilities"] else 0.0
        avg_abs = statistics.mean(stats["abs_moves"]) if stats["abs_moves"] else 0.0
        avg_dev = statistics.mean(stats["price_deviations"]) if stats["price_deviations"] else 0.0
        bot_freq = stats["bot_events"] / days if days > 0 else 0.0

        # Price bias: negative deviation = prices tend to be below fair value
        price_bias = "low" if avg_dev < -0.5 else ("high" if avg_dev > 0.5 else "neutral")

        # Entry quality: low volatility + prices below fair value = good entry
        # Higher score = better entry opportunity
        # Penalize: high volatility, prices above mean, frequent bot events
        quality = 10.0
        quality -= avg_vol * 0.5       # penalize volatility
        quality -= max(0, avg_dev) * 0.3  # penalize above-mean prices
        quality -= bot_freq * 3.0      # heavy penalty for bot event frequency
        quality += max(0, -avg_dev) * 0.2  # reward below-mean prices

        results.append({
            "window": label,
            "avg_volatility": round(avg_vol, 2),
            "avg_abs_movement": round(avg_abs, 2),
            "avg_price_deviation": round(avg_dev, 2),
            "price_bias": price_bias,
            "bot_event_frequency": round(bot_freq, 2),
            "entry_quality": round(quality, 2),
            "n_observations": n_obs,
            "days_analyzed": days,
        })

    # Sort by entry quality descending
    results.sort(key=lambda r: r["entry_quality"], reverse=True)
    return results


# =============================================================================
# MULTI-DAY REPORT
# =============================================================================

def generate_report(city: Optional[str] = None, days: int = 7) -> str:
    """Generate a multi-day market analysis report.

    Parameters
    ----------
    city : str, optional
        City code. If None, reports on all cities.
    days : int
        Number of recent days to analyze.

    Returns
    -------
    str
        Formatted text report.
    """
    cities_to_report = [city.upper()] if city else ALL_CITIES
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("MARKET TIMESERIES REPORT")
    lines.append(f"Generated: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    lines.append(f"Period: last {days} days")
    lines.append("=" * 70)

    for c in cities_to_report:
        available = _available_dates(c)
        recent = available[-days:] if len(available) > days else available

        if not recent:
            lines.append(f"\n{'='*40}")
            lines.append(f"  {c}: No snapshot data available")
            continue

        lines.append(f"\n{'='*40}")
        lines.append(f"  {c} ({STATIONS[c].city_name})")
        lines.append(f"  Dates with data: {len(recent)}")
        lines.append(f"{'='*40}")

        # ── Section 1: Price Volatility by Time of Day ──
        lines.append("\n--- PRICE VOLATILITY BY TIME OF DAY ---")
        windows = optimal_entry_windows(c, recent)
        if windows:
            lines.append(f"  {'Window':<8} {'AvgVol':>7} {'AvgMove':>8} {'Bias':>8} {'BotFreq':>8} {'Quality':>8}")
            lines.append(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            for w in windows:
                lines.append(
                    f"  {w['window']:<8} {w['avg_volatility']:>7.2f} "
                    f"{w['avg_abs_movement']:>8.2f} {w['price_bias']:>8} "
                    f"{w['bot_event_frequency']:>8.2f} {w['entry_quality']:>8.2f}"
                )
        else:
            lines.append("  Insufficient data")

        # ── Section 2: Bot Event Impact Summary ──
        lines.append("\n--- BOT EVENT IMPACT ---")
        total_events = 0
        significant_events = 0
        max_moves: List[float] = []

        for d in recent:
            events = detect_bot_events(c, d)
            for ev in events:
                total_events += 1
                if ev.get("significant"):
                    significant_events += 1
                max_moves.append(ev["max_move"])

        if total_events > 0:
            lines.append(f"  Total release events observed: {total_events}")
            lines.append(f"  Significant (>5c move): {significant_events} ({100*significant_events/total_events:.0f}%)")
            lines.append(f"  Average max move: {statistics.mean(max_moves):.1f}c")
            if max_moves:
                lines.append(f"  Largest move: {max(max_moves):.1f}c")
        else:
            lines.append("  No bot events detected (need more snapshots around release times)")

        # ── Section 3: Autocorrelation Summary ──
        lines.append("\n--- MEAN-REVERSION vs MOMENTUM ---")
        all_autocorrs: List[float] = []
        for d in recent:
            analysis = analyze_price_series(c, d)
            for t, stats in analysis.items():
                if stats["n_snapshots"] >= 5 and stats["autocorrelation_lag1"] != 0:
                    all_autocorrs.append(stats["autocorrelation_lag1"])

        if all_autocorrs:
            avg_ac = statistics.mean(all_autocorrs)
            regime = "MEAN-REVERSION" if avg_ac < -0.1 else ("MOMENTUM" if avg_ac > 0.1 else "RANDOM WALK")
            lines.append(f"  Average lag-1 autocorrelation: {avg_ac:+.3f}")
            lines.append(f"  Price regime: {regime}")
            if avg_ac < -0.1:
                interp = "Prices overshoot then correct. Bot repricing creates mean-reversion."
            elif avg_ac > 0.1:
                interp = "Prices trend. Information is incorporated gradually."
            else:
                interp = "No consistent pattern. Price changes are approximately random."
            lines.append(f"  Interpretation: {interp}")
        else:
            lines.append("  Insufficient data for autocorrelation (need 5+ snapshots per day)")

        # ── Section 4: Recommended Entry Windows ──
        lines.append("\n--- RECOMMENDED ENTRY WINDOWS ---")
        if windows:
            best = windows[0]
            lines.append(f"  BEST:  {best['window']} ET (quality={best['entry_quality']:.1f})")
            if len(windows) > 1:
                second = windows[1]
                lines.append(f"  2ND:   {second['window']} ET (quality={second['entry_quality']:.1f})")
            worst = windows[-1]
            lines.append(f"  AVOID: {worst['window']} ET (quality={worst['entry_quality']:.1f})")
        else:
            lines.append("  Insufficient data")

    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Market Timeseries — Kalshi orderbook snapshot collector and analyzer",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true", help="Capture one snapshot")
    mode.add_argument("--watch", action="store_true", help="Continuous capture loop")
    mode.add_argument("--analyze", action="store_true", help="Analyze price series for one day")
    mode.add_argument("--report", action="store_true", help="Multi-day analysis report")
    mode.add_argument("--bot-events", action="store_true", help="Detect bot repricing events")

    parser.add_argument("--city", type=str, default=None, help="City code (NYC, CHI, DEN, MIA, LAX)")
    parser.add_argument("--date", type=str, default=None, help="Date to analyze (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="Number of days for report (default: 7)")
    parser.add_argument("--interval", type=int, default=15, help="Capture interval in minutes (default: 15)")
    parser.add_argument("--ticker", type=str, default=None, help="Specific ticker to analyze")

    args = parser.parse_args()

    if args.capture:
        cities = [args.city] if args.city else None
        results = asyncio.run(capture_snapshot(cities))
        print(f"Captured {len(results)} snapshots")
        for r in results:
            n_brackets = len(r.get("brackets", {}))
            print(f"  {r['city']}: {n_brackets} brackets, {r['hours_to_settlement']:.1f}h to settlement")

    elif args.watch:
        cities = [args.city] if args.city else None
        try:
            asyncio.run(capture_loop(args.interval, cities))
        except KeyboardInterrupt:
            print("\nStopped.")

    elif args.analyze:
        if not args.date:
            print("ERROR: --date is required for --analyze")
            sys.exit(1)
        city = (args.city or "NYC").upper()
        analysis = analyze_price_series(city, args.date, args.ticker)
        if not analysis:
            print(f"No data for {city} on {args.date}")
            sys.exit(0)

        print(f"\nPrice Series Analysis: {city} — {args.date}")
        print(f"{'Ticker':<35} {'Snaps':>5} {'Vol':>6} {'AC1':>7} {'MaxDD':>6} {'Range':>6} {'Final':>6}")
        print("-" * 80)
        for t, stats in sorted(analysis.items()):
            # Shorten ticker for display
            short = t.split("-")[-1] if "-" in t else t
            print(
                f"  {short:<33} {stats['n_snapshots']:>5} "
                f"{stats['volatility']:>6.2f} {stats['autocorrelation_lag1']:>+7.3f} "
                f"{stats['max_drawdown']:>6.1f} {stats['total_range']:>6.1f} "
                f"{stats['final_price']:>6.1f}"
            )

    elif args.report:
        report = generate_report(city=args.city, days=args.days)
        print(report)

    elif args.bot_events:
        if not args.date:
            print("ERROR: --date is required for --bot-events")
            sys.exit(1)
        city = (args.city or "NYC").upper()
        events = detect_bot_events(city, args.date)
        if not events:
            print(f"No bot events detected for {city} on {args.date}")
            print("(Need snapshots captured around DSM/6-hour release times)")
            sys.exit(0)

        print(f"\nBot Event Detection: {city} — {args.date}")
        print(f"DSM times (Z):    {STATIONS[city].dsm_times_z}")
        print(f"6-hour times (Z): {STATIONS[city].six_hour_z}")
        print()

        for ev in events:
            sig = " *** SIGNIFICANT" if ev["significant"] else ""
            print(f"  {ev['event_type'].upper():>6} @ {ev['event_time_z']}Z — max move: {ev['max_move']:.1f}c{sig}")
            # Show top movers
            sorted_changes = sorted(
                ev["price_changes"].items(), key=lambda x: abs(x[1]), reverse=True,
            )
            for t, delta in sorted_changes[:3]:
                short = t.split("-")[-1] if "-" in t else t
                direction = "+" if delta > 0 else ""
                print(f"         {short}: {direction}{delta:.1f}c")


if __name__ == "__main__":
    main()
