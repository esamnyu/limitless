#!/usr/bin/env python3
"""
BACKTEST DATA COLLECTOR — Daily settlement data for model calibration.

Runs daily via cron AFTER settlement (~8 AM ET) to collect:
  1. Actual settlement temperature (from NWS daily observations)
  2. Yesterday's ensemble forecast (from saved snapshots)
  3. Market settlement results (from Kalshi API)

Appends to backtest/daily_data.jsonl for later analysis.

After 30+ days of data:
  - Optimize model weights (AIFS 1.30x, IFS 1.15x currently assumed)
  - Calibrate confidence thresholds (is 90 the right gate?)
  - Validate KDE bandwidth selection
  - Compare KDE vs histogram probability accuracy

Cron setup:
  0 8 * * * cd /Users/miqadmin/Documents/limitless && python3 backtest_collector.py >> /tmp/backtest_collector.log 2>&1

Manual:
  python3 backtest_collector.py
  python3 backtest_collector.py --date 2025-02-10   # Specific date
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv

load_dotenv()

from log_setup import get_logger
from edge_scanner_v2 import CITIES

logger = get_logger(__name__)

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent
BACKTEST_DIR = PROJECT_ROOT / "backtest"
DAILY_DATA_FILE = BACKTEST_DIR / "daily_data.jsonl"
SNAPSHOT_DIR = BACKTEST_DIR / "ensemble_snapshots"

# NWS observation endpoint for daily summary
NWS_OBS_BASE = "https://api.weather.gov/stations"


async def fetch_actual_high(session: aiohttp.ClientSession, city_key: str, date: datetime) -> Optional[float]:
    """
    Fetch the actual recorded high temperature for a city on a given date.

    Uses NWS observations API to get the daily max temperature.
    """
    city = CITIES.get(city_key)
    if not city:
        return None

    obs_url = city["nws_obs"].replace("/latest", "")
    headers = {"User-Agent": "BacktestCollector/1.0", "Accept": "application/geo+json"}

    # Query observations for the target date
    start = date.replace(hour=0, minute=0, second=0).isoformat()
    end = date.replace(hour=23, minute=59, second=59).isoformat()

    try:
        params = {"start": start, "end": end, "limit": 50}
        async with session.get(obs_url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                logger.warning(f"NWS obs for {city_key}: HTTP {resp.status}")
                return None
            data = await resp.json()

        features = data.get("features", [])
        if not features:
            logger.warning(f"No observations for {city_key} on {date.date()}")
            return None

        # Extract max temperature from all observations that day
        temps_f = []
        for obs in features:
            props = obs.get("properties", {})
            temp_c = props.get("temperature", {}).get("value")
            if temp_c is not None:
                temps_f.append(temp_c * 1.8 + 32)

        if not temps_f:
            return None

        actual_high = max(temps_f)
        logger.info(f"{city_key} actual high on {date.date()}: {actual_high:.1f}°F (from {len(temps_f)} obs)")
        return round(actual_high, 1)

    except Exception as e:
        logger.error(f"Failed to fetch actual high for {city_key}: {e}")
        return None


async def fetch_settlement_results(session: aiohttp.ClientSession, city_key: str, date: datetime) -> List[Dict]:
    """
    Fetch market settlement results for a city on a given date.

    Returns list of {ticker, settled_yes, settled_price} for each bracket.
    """
    city = CITIES.get(city_key)
    if not city:
        return []

    series = city["series"]
    kalshi_base = "https://api.elections.kalshi.com/trade-api/v2"

    try:
        url = f"{kalshi_base}/markets?series_ticker={series}&status=settled&limit=50"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()

        # Filter for the target date's markets
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        date_str = f"{date.year % 100:02d}{months[date.month - 1]}{date.day:02d}"

        results = []
        for m in data.get("markets", []):
            ticker = m.get("ticker", "")
            if date_str in ticker:
                results.append({
                    "ticker": ticker,
                    "title": m.get("title", ""),
                    "result": m.get("result", ""),
                    "yes_bid_close": m.get("yes_bid", 0),
                    "volume": m.get("volume", 0),
                })

        return results

    except Exception as e:
        logger.error(f"Failed to fetch settlement for {city_key}: {e}")
        return []


def load_ensemble_snapshot(city_key: str, date: datetime) -> Optional[Dict]:
    """Load saved ensemble snapshot for a city/date if available."""
    filename = SNAPSHOT_DIR / f"{date.date().isoformat()}_{city_key}.json"
    if filename.exists():
        try:
            return json.loads(filename.read_text())
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load snapshot {filename}: {e}")
    return None


def save_ensemble_snapshot(city_key: str, date: datetime, data: dict):
    """Save ensemble snapshot for later backtest analysis."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    filename = SNAPSHOT_DIR / f"{date.date().isoformat()}_{city_key}.json"
    try:
        filename.write_text(json.dumps(data, indent=2, default=str))
        logger.debug(f"Saved ensemble snapshot: {filename.name}")
    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")


async def collect_daily_data(target_date: datetime = None):
    """
    Collect settlement data for a given date (default: yesterday).

    Appends records to backtest/daily_data.jsonl.
    """
    now = datetime.now(ET)
    if target_date is None:
        target_date = now - timedelta(days=1)

    target_date = target_date.replace(tzinfo=ET) if target_date.tzinfo is None else target_date

    print(f"\n  BACKTEST COLLECTOR — {now.strftime('%I:%M %p ET')}")
    print(f"  Collecting data for: {target_date.date()}")
    print(f"  {'─'*40}")

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    async with aiohttp.ClientSession() as session:
        for city_key in CITIES:
            print(f"  {city_key}...", end=" ")

            try:
                # Fetch actual high and settlement results in parallel
                actual_task = fetch_actual_high(session, city_key, target_date)
                settle_task = fetch_settlement_results(session, city_key, target_date)
                actual_high, settlements = await asyncio.gather(actual_task, settle_task)

                # Load ensemble snapshot if saved during yesterday's scan
                snapshot = load_ensemble_snapshot(city_key, target_date)

                record = {
                    "date": target_date.date().isoformat(),
                    "city": city_key,
                    "actual_high": actual_high,
                    "settlements": settlements,
                    "collected_at": now.isoformat(),
                }

                if snapshot:
                    record["ensemble_mean"] = snapshot.get("mean")
                    record["ensemble_std"] = snapshot.get("std")
                    record["ensemble_count"] = snapshot.get("total_count")
                    record["per_model_means"] = snapshot.get("per_model_means", {})
                    record["nws_forecast"] = snapshot.get("nws_forecast_high")

                records.append(record)

                settled_count = len(settlements)
                high_str = f"{actual_high:.1f}°F" if actual_high else "N/A"
                snap_str = "✓" if snapshot else "✗"
                print(f"High: {high_str} | Settled: {settled_count} brackets | Snapshot: {snap_str}")

            except Exception as e:
                logger.error(f"Failed to collect data for {city_key}: {e}")
                print(f"FAILED — {e}")

    # Append to JSONL
    if records:
        with open(DAILY_DATA_FILE, "a") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"\n  Saved {len(records)} records to {DAILY_DATA_FILE.name}")

    # Summary stats
    if DAILY_DATA_FILE.exists():
        line_count = sum(1 for _ in open(DAILY_DATA_FILE))
        print(f"  Total records: {line_count} (collecting since {_first_date_in_file()})")
    else:
        print(f"  First collection — no historical data yet")

    print(f"  {'─'*40}")


def _first_date_in_file() -> str:
    """Get the earliest date in the JSONL file."""
    try:
        with open(DAILY_DATA_FILE) as f:
            first = json.loads(f.readline())
            return first.get("date", "unknown")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Collector — Daily settlement data for model calibration")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD). Default: yesterday")
    args = parser.parse_args()

    target = None
    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=ET)

    asyncio.run(collect_daily_data(target))
