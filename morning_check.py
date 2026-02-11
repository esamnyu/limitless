#!/usr/bin/env python3
"""
MORNING CHECK — Dynamic Pre-Settlement Position Monitor

Loads ALL open positions from positions.json and evaluates each:
  1. Fresh ensemble forecast (has uncertainty collapsed?)
  2. Current NWS observations (what's the actual temp trend?)
  3. Current Kalshi bracket prices (has our position repriced?)
  4. Decision per position: HOLD, SELL, or LET SETTLE

Runs via cron at 6 AM ET (before ~7 AM settlement):
  0 6 * * * cd /Users/miqadmin/Documents/limitless && python3 morning_check.py >> /tmp/morning_check.log 2>&1

Or manually:
  python3 morning_check.py
  python3 morning_check.py --city NYC   # Check only NYC positions
"""

import asyncio
import math
import re
from datetime import datetime
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv

load_dotenv()

from position_store import load_positions
from notifications import send_discord_alert
from edge_scanner_v2 import (
    CITIES,
    fetch_ensemble_v2,
    fetch_nws,
    kde_probability,
    compute_confidence_score,
    parse_bracket_range,
)
from config import SETTLEMENT_HOUR_ET

ET = ZoneInfo("America/New_York")
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _city_key_from_ticker(ticker: str) -> Optional[str]:
    """Extract city key from Kalshi ticker (e.g. KXHIGHNY-... → NYC)."""
    series_to_city = {cfg["series"]: key for key, cfg in CITIES.items()}
    # Extract series prefix from ticker (e.g. KXHIGHNY from KXHIGHNY-26FEB11-B36.5)
    match = re.match(r'^([A-Z]+)', ticker)
    if match:
        series = match.group(1)
        return series_to_city.get(series)
    return None


async def fetch_nws_obs(session: aiohttp.ClientSession, city_key: str) -> Optional[Dict]:
    """Current temperature observation for a city."""
    city = CITIES.get(city_key)
    if not city:
        return None
    headers = {"User-Agent": "MorningCheck/2.0", "Accept": "application/geo+json"}
    try:
        async with session.get(city["nws_obs"], headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
        props = data.get("properties", {})
        temp_c = props.get("temperature", {}).get("value")
        wind_ms = props.get("windSpeed", {}).get("value")
        obs_time = props.get("timestamp", "")
        if temp_c is not None:
            temp_f = temp_c * 1.8 + 32
            wind_mph = wind_ms * 2.237 if wind_ms else 0
            return {"temp_f": round(temp_f, 1), "wind_mph": round(wind_mph, 1), "time": obs_time}
    except Exception as e:
        print(f"    [ERR] NWS obs for {city_key}: {e}")
    return None


async def fetch_bracket_price(session: aiohttp.ClientSession, ticker: str, series: str) -> Optional[Dict]:
    """Current bid/ask and title for a specific bracket ticker."""
    try:
        url = f"{KALSHI_BASE}/markets?series_ticker={series}&status=open&limit=50"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
        for m in data.get("markets", []):
            if m.get("ticker") == ticker:
                return {
                    "bid": m.get("yes_bid", 0),
                    "ask": m.get("yes_ask", 0),
                    "volume": m.get("volume", 0),
                    "title": m.get("title", "") or m.get("subtitle", ""),
                }
    except Exception as e:
        print(f"    [ERR] Kalshi price for {ticker}: {e}")
    return None


async def fetch_balance() -> float:
    """Current account balance."""
    from kalshi_client import fetch_balance_quick
    return await fetch_balance_quick()


async def check_position(
    session: aiohttp.ClientSession,
    pos: dict,
    balance: float,
) -> dict:
    """
    Evaluate a single position against fresh data.

    Returns a decision dict: {ticker, action, reason, details}
    """
    ticker = pos["ticker"]
    side = pos["side"]
    entry_price = pos["avg_price"]
    contracts = pos["contracts"]
    cost = contracts * entry_price / 100

    city_key = _city_key_from_ticker(ticker)
    if not city_key:
        return {"ticker": ticker, "action": "MANUAL_CHECK", "reason": f"Unknown city for ticker {ticker}"}

    city = CITIES[city_key]
    series = city["series"]

    print(f"\n  {'─'*50}")
    print(f"  {side.upper()} {contracts}x {ticker} @ {entry_price}c")

    # Fetch fresh data in parallel
    try:
        today = datetime.now(ET).date()
        target_date_str = today.isoformat()

        results = await asyncio.gather(
            fetch_ensemble_v2(session, city_key, target_date_str),
            fetch_nws(session, city_key, today),
            fetch_nws_obs(session, city_key),
            fetch_bracket_price(session, ticker, series),
            return_exceptions=True,
        )

        errors = [(i, r) for i, r in enumerate(results) if isinstance(r, Exception)]
        for i, err in errors:
            names = ["ensemble", "nws_forecast", "nws_obs", "bracket_price"]
            print(f"    [WARN] {names[i]} fetch failed: {err}")

        ensemble = results[0] if not isinstance(results[0], Exception) else None
        nws_data = results[1] if not isinstance(results[1], Exception) else None
        obs = results[2] if not isinstance(results[2], Exception) else None
        price = results[3] if not isinstance(results[3], Exception) else None

    except Exception as e:
        return {"ticker": ticker, "action": "MANUAL_CHECK", "reason": f"Data fetch failed: {e}"}

    # Parse bracket range from market TITLE (not ticker — ticker has no bracket info)
    market_title = price.get("title", "") if price else ""
    if not market_title:
        return {"ticker": ticker, "action": "MANUAL_CHECK", "reason": "Could not fetch market title for bracket parsing"}

    low, high, bracket_type = parse_bracket_range(market_title)
    if bracket_type == "unknown":
        return {"ticker": ticker, "action": "MANUAL_CHECK", "reason": f"Could not parse bracket from title: {market_title}"}

    print(f"  City: {city['name']} | Bracket: {low}-{high}°F | Cost: ${cost:.2f}")

    # ── Current Observations ──
    if obs:
        print(f"  Current: {obs['temp_f']}°F  Wind: {obs['wind_mph']} mph")

    # ── Ensemble Analysis ──
    ensemble_prob = 0.0
    ensemble_mean = 0.0
    ensemble_std = 0.0
    if ensemble and ensemble.weighted_members:
        ensemble_mean = ensemble.mean
        ensemble_std = ensemble.std
        if bracket_type == "range":
            ensemble_prob = kde_probability(ensemble.weighted_members, low, high, ensemble.kde_bandwidth)
        elif bracket_type == "high_tail":
            ensemble_prob = kde_probability(ensemble.weighted_members, low, 200, ensemble.kde_bandwidth)
        elif bracket_type == "low_tail":
            ensemble_prob = kde_probability(ensemble.weighted_members, -100, high, ensemble.kde_bandwidth)

        print(f"  Ensemble: {ensemble.total_count} members | Mean: {ensemble_mean:.1f}°F ±{ensemble_std:.1f}")
        print(f"  KDE prob in bracket: {ensemble_prob*100:.1f}%")

        # Confidence score
        if nws_data:
            conf_label, conf_score, _ = compute_confidence_score(ensemble, nws_data)
            print(f"  Confidence: {conf_label} ({conf_score:.0f}/100)")

    # ── NWS Forecast ──
    if nws_data:
        print(f"  NWS forecast high: {nws_data.forecast_high:.0f}°F | Physics: {nws_data.physics_high:.1f}°F")
        nws_in_bracket = low <= nws_data.forecast_high < high if bracket_type == "range" else (
            nws_data.forecast_high >= low if bracket_type == "high_tail" else nws_data.forecast_high < high
        )
        if nws_in_bracket:
            print(f"  NWS forecast IS in our bracket ✓")
        else:
            print(f"  NWS forecast is OUTSIDE our bracket ✗")

    # ── Market Price ──
    bid = 0
    if price:
        bid = price["bid"]
        ask = price["ask"]
        roi = ((bid - entry_price) / entry_price * 100) if entry_price > 0 and bid > 0 else -100
        pnl = contracts * (bid - entry_price) / 100
        print(f"  Market: Bid={bid}c Ask={ask}c | ROI: {roi:+.0f}% | P&L: ${pnl:+.2f}")

    # ══════════════════════════════════════════
    #  DECISION ENGINE
    # ══════════════════════════════════════════
    now = datetime.now(ET)
    hours_to_settlement = (SETTLEMENT_HOUR_ET - now.hour) % 24
    if hours_to_settlement > 12:
        hours_to_settlement = 0  # We're past settlement time

    # Rule 1: Price exploded — take profit
    if bid >= entry_price * 2 and contracts > 1:
        sell_qty = max(1, contracts // 2)
        action = f"SELL {sell_qty} of {contracts}"
        reason = f"Price doubled ({entry_price}c → {bid}c). Freeroll: sell half, let rest ride to settlement."
        print(f"  >>> {action}: {reason}")
        return {"ticker": ticker, "action": action, "reason": reason, "sell_qty": sell_qty, "price": bid}

    # Rule 2: Efficiency exit — price near max
    if bid >= 90:
        action = f"SELL ALL {contracts}"
        reason = f"Price at {bid}c — lock in 90%+ of max payout rather than risk settlement."
        print(f"  >>> {action}: {reason}")
        return {"ticker": ticker, "action": action, "reason": reason, "sell_qty": contracts, "price": bid}

    # Rule 3: Thesis broken — model shifted away
    if ensemble_prob < 0.10 and ensemble and ensemble.weighted_members:
        action = f"SELL ALL {contracts}"
        reason = f"Ensemble probability dropped to {ensemble_prob*100:.0f}%. Thesis broken."
        print(f"  >>> {action}: {reason}")
        return {"ticker": ticker, "action": action, "reason": reason, "sell_qty": contracts, "price": bid}

    # Rule 4: NWS forecast far outside bracket
    if nws_data:
        nws_h = nws_data.forecast_high
        if bracket_type == "range":
            bracket_mid = (low + high) / 2
            distance = abs(nws_h - bracket_mid)
            if distance > 4:
                action = f"SELL ALL {contracts}"
                reason = f"NWS forecast {nws_h:.0f}°F is {distance:.0f}°F from bracket center ({bracket_mid:.0f}°F)."
                print(f"  >>> {action}: {reason}")
                return {"ticker": ticker, "action": action, "reason": reason, "sell_qty": contracts, "price": bid}

    # Rule 5: Strong alignment — hold for settlement
    if ensemble_prob > 0.30 and nws_data:
        nws_in = low <= nws_data.forecast_high < high if bracket_type == "range" else True
        if nws_in:
            action = "HOLD — LET SETTLE"
            reason = f"Ensemble {ensemble_prob*100:.0f}% + NWS agrees. {hours_to_settlement:.0f}h to settlement."
            payout = contracts  # $1 per contract
            print(f"  >>> {action}: {reason}")
            print(f"      Payout if correct: ${payout:.2f}")
            return {"ticker": ticker, "action": action, "reason": reason}

    # Default: cautious hold
    action = "HOLD (monitor)"
    reason = f"No strong signal. Ensemble: {ensemble_prob*100:.0f}%, Bid: {bid}c. Watch for changes."
    print(f"  >>> {action}: {reason}")
    return {"ticker": ticker, "action": action, "reason": reason}


async def main(city_filter: str = None):
    now = datetime.now(ET)
    positions = load_positions()
    open_positions = [p for p in positions if p["status"] == "open"]

    if not open_positions:
        print(f"\n  MORNING CHECK — {now.strftime('%I:%M %p ET, %A %B %d')}")
        print(f"  No open positions. Nothing to check.")
        return

    # Filter by city if specified
    if city_filter:
        city_filter = city_filter.upper()
        open_positions = [
            p for p in open_positions
            if _city_key_from_ticker(p["ticker"]) == city_filter
        ]
        if not open_positions:
            print(f"  No open positions for {city_filter}.")
            return

    print(f"\n{'='*60}")
    print(f"  MORNING CHECK — {now.strftime('%I:%M %p ET, %A %B %d')}")
    print(f"  Open positions: {len(open_positions)}")
    print(f"{'='*60}")

    balance = await fetch_balance()
    print(f"  Balance: ${balance:.2f}")

    decisions = []
    async with aiohttp.ClientSession() as session:
        for pos in open_positions:
            try:
                decision = await check_position(session, pos, balance)
                decisions.append(decision)
            except Exception as e:
                print(f"  [ERR] Failed to check {pos['ticker']}: {e}")
                decisions.append({
                    "ticker": pos["ticker"],
                    "action": "MANUAL_CHECK",
                    "reason": f"Check failed: {e}",
                })

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    sell_actions = [d for d in decisions if "SELL" in d.get("action", "")]
    hold_actions = [d for d in decisions if "HOLD" in d.get("action", "")]
    manual_actions = [d for d in decisions if "MANUAL" in d.get("action", "")]

    for d in decisions:
        icon = "🔴" if "SELL" in d["action"] else "🟢" if "HOLD" in d["action"] else "⚪"
        print(f"  {icon} {d['ticker']}: {d['action']}")
        print(f"     {d['reason']}")

    # Send Discord alert if any action items
    if sell_actions:
        alert_lines = [f"**{d['ticker']}**: {d['action']}\n{d['reason']}" for d in sell_actions]
        await send_discord_alert(
            title=f"🌅 MORNING CHECK — {len(sell_actions)} SELL SIGNAL(S)",
            description="\n\n".join(alert_lines),
            color=0xFF6600,
            context="morning_check",
        )
    elif hold_actions and not manual_actions:
        tickers = [d["ticker"] for d in hold_actions]
        await send_discord_alert(
            title="🌅 MORNING CHECK — ALL CLEAR",
            description=f"All {len(hold_actions)} positions holding through settlement.\n" +
                        "\n".join(f"• {t}" for t in tickers),
            color=0x00FF00,
            context="morning_check",
        )

    print(f"\n  Settlement: ~{SETTLEMENT_HOUR_ET}:00 AM ET")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Morning Check — Dynamic Pre-Settlement Position Monitor")
    parser.add_argument("--city", type=str, default=None, help="Filter by city code (NYC, CHI, etc.)")
    args = parser.parse_args()
    asyncio.run(main(args.city))
