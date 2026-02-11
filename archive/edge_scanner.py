#!/usr/bin/env python3
"""
EDGE SCANNER v1.0 — Multi-City Ensemble vs Market Analyzer

Pulls:
  1. Open-Meteo ensemble forecasts (GFS + ECMWF + ICON = ~119 members)
  2. NWS hourly forecasts (official point forecast + wind/precip)
  3. Kalshi market prices (live bid/ask for every bracket)

Then computes:
  - Per-bracket model probability from ensemble members
  - Edge = model_prob - market_implied_prob (after fees)
  - Kelly fraction & suggested sizing
  - Strategy flags (Midnight High, Wind Penalty, Wet Bulb)
  - Confidence rating based on ensemble spread

Usage:
  python3 edge_scanner.py              # Scan all cities
  python3 edge_scanner.py --city NYC   # Single city
"""

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

CITIES = {
    "NYC": {
        "name": "New York (Central Park)",
        "series": "KXHIGHNY",
        "lat": 40.78, "lon": -73.97,
        "nws_hourly": "https://api.weather.gov/gridpoints/OKX/33,37/forecast/hourly",
        "nws_obs": "https://api.weather.gov/stations/KNYC/observations/latest",
        "tz": "America/New_York",
    },
    "CHI": {
        "name": "Chicago (Midway)",
        "series": "KXHIGHCHI",
        "lat": 41.79, "lon": -87.75,
        "nws_hourly": "https://api.weather.gov/gridpoints/LOT/75,72/forecast/hourly",
        "nws_obs": "https://api.weather.gov/stations/KMDW/observations/latest",
        "tz": "America/Chicago",
    },
}


@dataclass
class EnsembleStats:
    members: list[float] = field(default_factory=list)
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    count: int = 0


@dataclass
class NWSData:
    forecast_high: float = 0.0
    midnight_temp: float = 0.0
    afternoon_temp: float = 0.0
    peak_wind_sustained: float = 0.0
    peak_wind_gust: float = 0.0
    peak_precip_prob: int = 0
    peak_dewpoint: float = 0.0
    is_midnight_high: bool = False
    wind_penalty: float = 0.0
    wet_bulb_penalty: float = 0.0
    physics_high: float = 0.0
    hourly_temps: list[tuple] = field(default_factory=list)  # (hour, temp, wind, precip)


@dataclass
class BracketOpportunity:
    city: str
    bracket_title: str
    ticker: str
    low: float
    high: float
    # Market
    yes_bid: int = 0
    yes_ask: int = 0
    volume: int = 0
    # Model
    model_prob: float = 0.0
    model_prob_pct: float = 0.0
    # Edge
    edge_raw: float = 0.0
    edge_after_fees: float = 0.0
    taker_fee: float = 0.0
    # Sizing
    kelly: float = 0.0
    suggested_contracts: int = 0
    # Flags
    side: str = "yes"
    confidence: str = "LOW"
    strategies: list[str] = field(default_factory=list)
    rationale: str = ""


# ─── Fetchers ──────────────────────────────────────────

async def fetch_ensemble(session: aiohttp.ClientSession, city_key: str, target_date: str) -> EnsembleStats:
    """Fetch ensemble forecast from Open-Meteo (GFS + ECMWF + ICON)."""
    city = CITIES[city_key]
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "models": "gfs_seamless,ecmwf_ifs025,icon_seamless",
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "start_date": target_date,
        "end_date": target_date,
    }
    try:
        async with session.get(ENSEMBLE_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                print(f"  [WARN] Open-Meteo returned {resp.status} for {city_key}")
                return EnsembleStats()
            data = await resp.json()

        members = []
        daily = data.get("daily", {})
        for key, values in daily.items():
            if key.startswith("temperature_2m_max") and isinstance(values, list):
                for v in values:
                    if v is not None:
                        members.append(float(v))

        if not members:
            return EnsembleStats()

        sorted_m = sorted(members)
        n = len(sorted_m)
        mean = sum(sorted_m) / n
        variance = sum((v - mean) ** 2 for v in sorted_m) / n
        std = math.sqrt(variance)
        pct = lambda p: sorted_m[min(int(p / 100 * (n - 1)), n - 1)]

        return EnsembleStats(
            members=sorted_m,
            mean=mean,
            median=pct(50),
            std=std,
            min_val=sorted_m[0],
            max_val=sorted_m[-1],
            p10=pct(10), p25=pct(25), p75=pct(75), p90=pct(90),
            count=n,
        )
    except Exception as e:
        print(f"  [ERR] Ensemble fetch failed for {city_key}: {e}")
        return EnsembleStats()


async def fetch_nws(session: aiohttp.ClientSession, city_key: str, target_date) -> NWSData:
    """Fetch NWS hourly forecast and current obs."""
    city = CITIES[city_key]
    tz = ZoneInfo(city["tz"])
    result = NWSData()

    headers = {"User-Agent": "EdgeScanner/1.0", "Accept": "application/geo+json"}

    # Hourly forecast
    try:
        async with session.get(city["nws_hourly"], headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                print(f"  [WARN] NWS hourly returned {resp.status} for {city_key}")
                return result
            data = await resp.json()

        periods = data.get("properties", {}).get("periods", [])
        tomorrow_temps = []
        midnight_temps = []
        afternoon_temps = []
        peak_wind = 0.0
        peak_precip = 0
        peak_dewpoint = 0.0

        for p in periods:
            try:
                t = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00")).astimezone(tz)
                if t.date() != target_date:
                    continue

                temp_f = float(p.get("temperature", 0))
                wind_str = p.get("windSpeed", "0 mph")
                wind_match = re.search(r"(\d+)\s*(?:to\s*(\d+))?\s*mph", wind_str, re.I)
                wind_speed = float(wind_match.group(2) or wind_match.group(1)) if wind_match else 0.0
                precip_val = p.get("probabilityOfPrecipitation", {}).get("value")
                precip = int(precip_val) if precip_val is not None else 0
                dew_val = p.get("dewpoint", {}).get("value")
                dew_f = (float(dew_val) * 1.8 + 32) if dew_val is not None else 0.0

                tomorrow_temps.append(temp_f)
                result.hourly_temps.append((t.hour, temp_f, wind_speed, precip))

                if 0 <= t.hour <= 1:
                    midnight_temps.append(temp_f)
                if 14 <= t.hour <= 16:
                    afternoon_temps.append(temp_f)

                if wind_speed > peak_wind:
                    peak_wind = wind_speed
                if precip > peak_precip:
                    peak_precip = precip
                if temp_f == max(tomorrow_temps):
                    peak_dewpoint = dew_f

            except (KeyError, ValueError):
                continue

        if tomorrow_temps:
            result.forecast_high = max(tomorrow_temps)

        if midnight_temps:
            result.midnight_temp = max(midnight_temps)
        if afternoon_temps:
            result.afternoon_temp = max(afternoon_temps)

        result.is_midnight_high = (
            result.midnight_temp > result.afternoon_temp
            if result.midnight_temp and result.afternoon_temp else False
        )

        result.peak_wind_sustained = peak_wind
        result.peak_wind_gust = peak_wind * 1.5 if peak_wind > 10 else peak_wind
        result.peak_precip_prob = peak_precip
        result.peak_dewpoint = peak_dewpoint

        # Strategy B: Wind penalty
        if result.peak_wind_gust > 25:
            result.wind_penalty = 2.0
        elif result.peak_wind_gust > 15:
            result.wind_penalty = 1.0

        # Strategy D: Wet bulb
        if peak_precip >= 40:
            depression = result.forecast_high - peak_dewpoint
            if depression >= 5:
                factor = 0.40 if peak_precip >= 70 else 0.25
                result.wet_bulb_penalty = round(depression * factor, 1)

        result.physics_high = result.forecast_high - result.wind_penalty - result.wet_bulb_penalty
        if result.is_midnight_high:
            result.physics_high = result.midnight_temp

    except Exception as e:
        print(f"  [ERR] NWS fetch failed for {city_key}: {e}")

    return result


async def fetch_kalshi_brackets(session: aiohttp.ClientSession, city_key: str) -> list[dict]:
    """Fetch open Kalshi brackets for a city."""
    city = CITIES[city_key]
    try:
        url = f"{KALSHI_BASE}/markets?series_ticker={city['series']}&status=open&limit=100"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                print(f"  [WARN] Kalshi returned {resp.status} for {city_key}")
                return []
            data = await resp.json()
            return data.get("markets", [])
    except Exception as e:
        print(f"  [ERR] Kalshi fetch failed for {city_key}: {e}")
        return []


# ─── Analysis ──────────────────────────────────────────

def parse_bracket_range(title: str) -> tuple[float, float, str]:
    """Parse bracket title into (low, high, edge_type)."""
    clean = title.replace("°F", "").replace("°", "").strip()

    if re.search(r"below|under|or less", clean, re.I):
        num = float(re.search(r"([\d.]+)", clean).group(1))
        return (-999, num, "low_tail")

    if re.search(r"above|or more|or higher", clean, re.I):
        num = float(re.search(r"([\d.]+)", clean).group(1))
        return (num, 999, "high_tail")

    match = re.search(r"([\d.]+)\s*(?:to|-)\s*([\d.]+)", clean)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        return (low, high + 1, "range")  # +1 because "34 to 35" means 34.0-35.999

    return (0, 0, "unknown")


def compute_bracket_prob(members: list[float], low: float, high: float) -> float:
    """Fraction of ensemble members falling in [low, high)."""
    if not members:
        return 0.0
    count = sum(1 for t in members if low <= t < high)
    return count / len(members)


def taker_fee_cents(price_cents: int) -> float:
    """Kalshi taker fee: 7% * p * (1-p)."""
    p = price_cents / 100
    return round(0.07 * p * (1 - p) * 100, 2)


def kelly_fraction(model_prob: float, market_price: float) -> float:
    """Half-Kelly for sizing."""
    if model_prob <= 0 or market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1 / market_price) - 1
    f = (b * model_prob - (1 - model_prob)) / b
    return max(0, f * 0.5)


def is_tomorrow_ticker(ticker: str, tomorrow_date) -> bool:
    """Check if a ticker is for tomorrow's date."""
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    date_str = f"{tomorrow_date.year % 100:02d}{months[tomorrow_date.month - 1]}{tomorrow_date.day:02d}"
    return date_str in ticker


def analyze_opportunities(
    city_key: str,
    ensemble: EnsembleStats,
    nws: NWSData,
    brackets: list[dict],
    balance: float,
) -> list[BracketOpportunity]:
    """Analyze all brackets and find edge opportunities."""
    tz = ZoneInfo(CITIES[city_key]["tz"])
    tomorrow = (datetime.now(tz) + timedelta(days=1)).date()

    opps = []
    for mkt in brackets:
        ticker = mkt.get("ticker", "")
        if not is_tomorrow_ticker(ticker, tomorrow):
            continue

        title = mkt.get("title", "") or mkt.get("subtitle", "")
        low, high, edge_type = parse_bracket_range(title)
        if edge_type == "unknown":
            continue

        yes_bid = mkt.get("yes_bid", 0) or mkt.get("yes_price", 0)
        yes_ask = mkt.get("yes_ask", 0)
        volume = mkt.get("volume", 0)

        # Model probability from ensemble
        model_prob = compute_bracket_prob(ensemble.members, low, high)

        # Market implied probability (use bid for buying YES, ask for selling)
        market_prob_bid = yes_bid / 100 if yes_bid > 0 else 0.01
        market_prob_ask = yes_ask / 100 if yes_ask > 0 else 0.99

        # YES edge: model says higher prob than market bid
        yes_edge = model_prob - market_prob_bid
        # NO edge: model says lower prob than market ask
        no_edge = market_prob_ask - model_prob

        fee = taker_fee_cents(yes_bid if yes_edge > no_edge else yes_ask)

        if yes_edge > 0.04:  # >4% raw edge on YES side
            side = "yes"
            edge_raw = yes_edge
            edge_after = edge_raw - (fee / 100)
            entry_price = market_prob_bid
        elif no_edge > 0.04:  # >4% raw edge on NO side
            side = "no"
            edge_raw = no_edge
            edge_after = edge_raw - (fee / 100)
            entry_price = 1 - market_prob_ask
        else:
            continue

        if edge_after <= 0.01:  # Need at least 1% edge after fees
            continue

        # Kelly & sizing
        if side == "yes":
            k = kelly_fraction(model_prob, market_prob_bid)
        else:
            k = kelly_fraction(1 - model_prob, 1 - market_prob_ask)

        max_cost = balance * 0.15  # 15% of NLV
        price_cents = yes_bid if side == "yes" else (100 - yes_ask)
        suggested = min(100, int(max_cost / (max(price_cents, 1) / 100))) if price_cents > 0 else 0

        # Confidence from ensemble spread
        if ensemble.std < 1.5:
            confidence = "HIGH"
        elif ensemble.std < 2.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Strategy flags
        strategies = []
        if nws.is_midnight_high:
            strategies.append("A:MIDNIGHT_HIGH")
        if nws.wind_penalty > 0:
            strategies.append(f"B:WIND_PENALTY(-{nws.wind_penalty:.0f}F)")
        if nws.wet_bulb_penalty > 0:
            strategies.append(f"D:WET_BULB(-{nws.wet_bulb_penalty:.1f}F)")
        if abs(nws.forecast_high - ensemble.mean) > 2:
            strategies.append(f"E:MODEL_DIVERGE(NWS={nws.forecast_high:.0f} vs ENS={ensemble.mean:.1f})")

        # Build rationale
        rationale_parts = []
        if model_prob > 0.3 and market_prob_bid < 0.10:
            rationale_parts.append("MASSIVE MISPRICING — market hasn't updated")
        if ensemble.std < 1.5:
            rationale_parts.append("Tight ensemble = high confidence")
        if ensemble.std > 3:
            rationale_parts.append("Wide ensemble spread — uncertain")
        if nws.wind_penalty > 0:
            rationale_parts.append(f"Wind mixing caps heating (-{nws.wind_penalty:.0f}F)")
        if nws.is_midnight_high:
            rationale_parts.append(f"Midnight temp ({nws.midnight_temp:.0f}F) locks the high")
        if abs(nws.forecast_high - ensemble.mean) > 2:
            direction = "warmer" if nws.forecast_high > ensemble.mean else "cooler"
            rationale_parts.append(f"NWS {direction} than ensemble by {abs(nws.forecast_high - ensemble.mean):.1f}F")

        opp = BracketOpportunity(
            city=city_key,
            bracket_title=title,
            ticker=ticker,
            low=low,
            high=high,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            volume=volume,
            model_prob=model_prob,
            model_prob_pct=model_prob * 100,
            edge_raw=edge_raw,
            edge_after_fees=edge_after,
            taker_fee=fee,
            kelly=k,
            suggested_contracts=suggested,
            side=side,
            confidence=confidence,
            strategies=strategies,
            rationale=" · ".join(rationale_parts) if rationale_parts else "Ensemble edge",
        )
        opps.append(opp)

    opps.sort(key=lambda x: x.edge_after_fees, reverse=True)
    return opps


# ─── Display ───────────────────────────────────────────

def print_city_report(
    city_key: str,
    ensemble: EnsembleStats,
    nws: NWSData,
    brackets: list[dict],
    opps: list[BracketOpportunity],
):
    city = CITIES[city_key]
    tz = ZoneInfo(city["tz"])
    tomorrow = (datetime.now(tz) + timedelta(days=1)).date()

    print(f"\n{'='*72}")
    print(f"  {city['name'].upper()} — {tomorrow.strftime('%A %B %d, %Y')}")
    print(f"{'='*72}")

    # Ensemble summary
    if ensemble.count > 0:
        print(f"\n  ENSEMBLE FORECAST ({ensemble.count} members)")
        print(f"  ├─ Mean: {ensemble.mean:.1f}°F  ±{ensemble.std:.1f}°  (Median: {ensemble.median:.1f}°F)")
        print(f"  ├─ Range: {ensemble.min_val:.0f}°F → {ensemble.max_val:.0f}°F")
        print(f"  └─ P10={ensemble.p10:.0f}  P25={ensemble.p25:.0f}  P50={ensemble.median:.0f}  P75={ensemble.p75:.0f}  P90={ensemble.p90:.0f}")
    else:
        print(f"\n  [WARN] No ensemble data available")

    # NWS summary
    if nws.forecast_high > 0:
        print(f"\n  NWS POINT FORECAST")
        print(f"  ├─ Forecast High: {nws.forecast_high:.0f}°F")
        print(f"  ├─ Physics High:  {nws.physics_high:.1f}°F", end="")
        adjustments = []
        if nws.wind_penalty > 0:
            adjustments.append(f"wind -{nws.wind_penalty:.0f}°F")
        if nws.wet_bulb_penalty > 0:
            adjustments.append(f"wetbulb -{nws.wet_bulb_penalty:.1f}°F")
        if adjustments:
            print(f"  ({', '.join(adjustments)})")
        else:
            print()
        print(f"  ├─ Wind: {nws.peak_wind_sustained:.0f} mph sustained (est. gusts {nws.peak_wind_gust:.0f} mph)")
        print(f"  ├─ Precip: {nws.peak_precip_prob}%  Dewpoint: {nws.peak_dewpoint:.0f}°F")
        print(f"  ├─ Midnight High: {'YES ⚠' if nws.is_midnight_high else 'No'}", end="")
        if nws.midnight_temp:
            print(f"  (12AM={nws.midnight_temp:.0f}°F vs 3PM={nws.afternoon_temp:.0f}°F)")
        else:
            print()

        # NWS vs Ensemble divergence
        if ensemble.count > 0:
            div = nws.forecast_high - ensemble.mean
            if abs(div) > 1:
                direction = "WARMER" if div > 0 else "COOLER"
                print(f"  └─ ⚠ NWS is {abs(div):.1f}°F {direction} than ensemble mean")
            else:
                print(f"  └─ NWS and ensemble aligned (Δ {div:+.1f}°F)")

    # Market brackets
    tomorrow_brackets = [
        m for m in brackets if is_tomorrow_ticker(m.get("ticker", ""), tomorrow)
    ]

    if tomorrow_brackets:
        bracket_sum = sum(m.get("yes_bid", 0) or m.get("yes_price", 0) for m in tomorrow_brackets)
        total_vol = sum(m.get("volume", 0) for m in tomorrow_brackets)

        print(f"\n  KALSHI BRACKETS ({len(tomorrow_brackets)} markets, Σbid={bracket_sum}¢, vol={total_vol:,})")
        print(f"  {'Bracket':<16} {'Bid':>5} {'Ask':>5} {'Model':>7} {'Edge':>8} {'Vol':>8}")
        print(f"  {'─'*16} {'─'*5} {'─'*5} {'─'*7} {'─'*8} {'─'*8}")

        for mkt in sorted(tomorrow_brackets, key=lambda x: x.get("title", "")):
            title = mkt.get("title", "") or mkt.get("subtitle", "")
            bid = mkt.get("yes_bid", 0) or mkt.get("yes_price", 0)
            ask = mkt.get("yes_ask", 0)
            vol = mkt.get("volume", 0)

            low, high, _ = parse_bracket_range(title)
            prob = compute_bracket_prob(ensemble.members, low, high) * 100 if ensemble.members else 0

            edge = prob - bid
            edge_str = f"{edge:+.1f}¢"
            if abs(edge) > 10:
                edge_str += " ⚡"
            elif abs(edge) > 5:
                edge_str += " ●"

            print(f"  {title:<16} {bid:>4}¢ {ask:>4}¢ {prob:>5.0f}% {edge_str:>8} {vol:>8,}")
    else:
        print(f"\n  [INFO] No tomorrow brackets found for {city['series']}")

    # Opportunities
    if opps:
        print(f"\n  {'─'*68}")
        print(f"  OPPORTUNITIES ({len(opps)} found)")
        print(f"  {'─'*68}")

        for i, opp in enumerate(opps, 1):
            side_color = "YES" if opp.side == "yes" else "NO"
            print(f"\n  [{i}] {side_color} {opp.bracket_title} @ {opp.yes_bid if opp.side == 'yes' else 100 - opp.yes_ask}¢")
            print(f"      Ticker:     {opp.ticker}")
            print(f"      Model Prob: {opp.model_prob_pct:.1f}%  →  Market: {opp.yes_bid}¢ bid / {opp.yes_ask}¢ ask")
            print(f"      Edge:       {opp.edge_raw*100:+.1f}¢ raw  →  {opp.edge_after_fees*100:+.1f}¢ after fees (fee: {opp.taker_fee:.1f}¢)")
            print(f"      Kelly:      {opp.kelly*100:.1f}%  →  {opp.suggested_contracts} contracts suggested")
            print(f"      Confidence: {opp.confidence}  (σ={ensemble.std:.1f}°F)")
            if opp.strategies:
                print(f"      Strategies: {', '.join(opp.strategies)}")
            print(f"      Rationale:  {opp.rationale}")
    else:
        print(f"\n  No opportunities above threshold.")


def print_summary(all_opps: list[BracketOpportunity], balance: float):
    """Print final summary across all cities."""
    print(f"\n{'='*72}")
    print(f"  SCAN SUMMARY")
    print(f"{'='*72}")
    print(f"  Balance:       ${balance:.2f}")
    print(f"  Opportunities: {len(all_opps)}")

    if not all_opps:
        print(f"\n  No actionable opportunities found. Markets may be efficient or")
        print(f"  ensemble uncertainty is too high. Check again at next model run.")
        return

    # Rank by edge
    ranked = sorted(all_opps, key=lambda x: x.edge_after_fees, reverse=True)

    print(f"\n  TOP OPPORTUNITIES (ranked by edge after fees):")
    print(f"  {'#':<3} {'City':<5} {'Side':<4} {'Bracket':<16} {'Price':>5} {'Model':>6} {'Edge':>8} {'Conf':<6}")
    print(f"  {'─'*3} {'─'*5} {'─'*4} {'─'*16} {'─'*5} {'─'*6} {'─'*8} {'─'*6}")

    for i, opp in enumerate(ranked[:10], 1):
        price = opp.yes_bid if opp.side == "yes" else (100 - opp.yes_ask)
        print(f"  {i:<3} {opp.city:<5} {opp.side.upper():<4} {opp.bracket_title:<16} {price:>4}¢ {opp.model_prob_pct:>5.0f}% {opp.edge_after_fees*100:>+7.1f}¢ {opp.confidence:<6}")

    # Best trade recommendation
    best = ranked[0]
    print(f"\n  {'='*68}")
    print(f"  RECOMMENDED TRADE")
    print(f"  {'='*68}")
    price = best.yes_bid if best.side == "yes" else (100 - best.yes_ask)
    print(f"  {best.side.upper()} {best.bracket_title} ({best.city})")
    print(f"  Ticker:  {best.ticker}")
    print(f"  Price:   {price}¢ (limit order, maker = $0 fee)")
    print(f"  Model:   {best.model_prob_pct:.0f}% probability")
    print(f"  Edge:    {best.edge_after_fees*100:+.1f}¢ after fees")
    print(f"  Kelly:   {best.kelly*100:.1f}% of bankroll")

    if balance > 0:
        max_contracts = int(balance / (max(price, 1) / 100))
        kelly_contracts = max(1, int(balance * best.kelly / (max(price, 1) / 100)))
        print(f"  Size:    {kelly_contracts} contracts (Kelly) / {max_contracts} max")
        print(f"  Cost:    ${kelly_contracts * price / 100:.2f}")
        print(f"  Payout:  ${kelly_contracts:.2f} if correct")

    print(f"\n  Rationale: {best.rationale}")
    if best.strategies:
        print(f"  Active:    {', '.join(best.strategies)}")
    print(f"  {'='*68}")


# ─── Main ──────────────────────────────────────────────

async def scan(city_filter: str = None):
    """Run full multi-city scan."""
    print(f"\n{'#'*72}")
    print(f"  EDGE SCANNER v1.0 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"{'#'*72}")

    cities_to_scan = {city_filter.upper(): CITIES[city_filter.upper()]} if city_filter else CITIES

    # Check balance
    balance = 0.0
    try:
        from kalshi_client import KalshiClient
        api_key = os.getenv("KALSHI_API_KEY_ID")
        pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if api_key and pk_path:
            client = KalshiClient(api_key_id=api_key, private_key_path=pk_path, demo_mode=False)
            await client.start()
            balance = await client.get_balance()
            await client.stop()
            print(f"\n  Account Balance: ${balance:.2f}")
        else:
            print(f"\n  [INFO] No Kalshi credentials — using public data only")
    except Exception as e:
        print(f"\n  [WARN] Could not fetch balance: {e}")

    # Determine target date
    tz = ZoneInfo("America/New_York")
    tomorrow = (datetime.now(tz) + timedelta(days=1)).date()
    target_date_str = tomorrow.isoformat()
    print(f"  Target Date: {tomorrow.strftime('%A %B %d, %Y')}")
    print(f"  Cities: {', '.join(cities_to_scan.keys())}")

    all_opps = []

    async with aiohttp.ClientSession() as session:
        for city_key in cities_to_scan:
            print(f"\n  Scanning {city_key}...")

            # Fetch all data in parallel
            ens_task = fetch_ensemble(session, city_key, target_date_str)
            nws_task = fetch_nws(session, city_key, tomorrow)
            mkt_task = fetch_kalshi_brackets(session, city_key)

            ensemble, nws_data, brackets = await asyncio.gather(ens_task, nws_task, mkt_task)

            print(f"    Ensemble: {ensemble.count} members, mean={ensemble.mean:.1f}°F ±{ensemble.std:.1f}")
            print(f"    NWS High: {nws_data.forecast_high:.0f}°F, Physics: {nws_data.physics_high:.1f}°F")
            print(f"    Brackets: {len(brackets)} total")

            opps = analyze_opportunities(city_key, ensemble, nws_data, brackets, balance)
            all_opps.extend(opps)

            print_city_report(city_key, ensemble, nws_data, brackets, opps)

    print_summary(all_opps, balance)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edge Scanner — Multi-City Ensemble vs Market")
    parser.add_argument("--city", type=str, default=None, help="City code (NYC, CHI)")
    args = parser.parse_args()
    asyncio.run(scan(args.city))
