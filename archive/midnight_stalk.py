#!/usr/bin/env python3
"""
MIDNIGHT STALK - Rounding Arbitrage Execution Script

Strategy: At 11:55 PM ET, read the current KNYC temperature.
The NWS rounds to nearest degree (x.50+ rounds UP, x.49- rounds DOWN).
Execute based on the thermometer, not the forecast.

Usage:
  python3 midnight_stalk.py          # Analysis mode
  python3 midnight_stalk.py --live   # Live trading mode
"""

import argparse
import asyncio
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv

from kalshi_client import KalshiClient

load_dotenv(Path(".env"))

# Constants
TZ = ZoneInfo("America/New_York")
NWS_OBS_URL = "https://api.weather.gov/stations/KNYC/observations/latest"
SERIES_TICKER = "KXHIGHNY"


async def get_current_temp() -> tuple[float, datetime]:
    """Fetch current KNYC temperature from NWS."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            NWS_OBS_URL,
            headers={"User-Agent": "MidnightStalk/1.0", "Accept": "application/geo+json"}
        ) as resp:
            if resp.status != 200:
                raise Exception(f"NWS observation failed: HTTP {resp.status}")
            data = await resp.json()
            props = data.get("properties", {})

            temp_c = props.get("temperature", {}).get("value")
            if temp_c is None:
                raise Exception("No temperature data in observation")

            temp_f = (temp_c * 1.8) + 32

            timestamp_str = props.get("timestamp", "")
            obs_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            return temp_f, obs_time


def round_temp(temp_f: float) -> int:
    """NWS rounding rule: x.50+ rounds UP, x.49- rounds DOWN."""
    return int(Decimal(str(temp_f)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def get_target_bracket(rounded_temp: int) -> tuple[str, int, int]:
    """Determine target bracket based on rounded temperature."""
    # Kalshi brackets are typically X to X+1 or X-1 to X
    # For a rounded temp of 34, the bracket is 33-34
    # For a rounded temp of 35, the bracket is 35-36

    if rounded_temp <= 32:
        return "32_or_below", 0, 32
    elif rounded_temp >= 41:
        return "41_or_above", 41, 999
    else:
        # Standard 2-degree brackets
        if rounded_temp % 2 == 1:  # Odd (33, 35, 37, 39)
            low = rounded_temp
            high = rounded_temp + 1
        else:  # Even (34, 36, 38, 40)
            low = rounded_temp - 1
            high = rounded_temp
        return f"{low}_{high}", low, high


async def find_market(client: KalshiClient, bracket_desc: str, target_date: str):
    """Find the Kalshi market for the target bracket."""
    markets = await client.get_markets(series_ticker=SERIES_TICKER, status="open", limit=50)

    for m in markets:
        ticker = m.get("ticker", "")
        subtitle = m.get("subtitle", "").lower()

        if target_date not in ticker:
            continue

        # Match bracket - be specific to avoid false matches
        if bracket_desc == "32_or_below":
            # Match "32° or below" or similar
            if "below" in subtitle and ("32" in subtitle or "31" in subtitle):
                return m
        elif bracket_desc == "41_or_above":
            # Match "41° or above" or similar
            if "above" in subtitle and ("41" in subtitle or "42" in subtitle):
                return m
        elif bracket_desc.count("_") == 1:
            # Simple "low_high" format like "33_34"
            low, high = bracket_desc.split("_")
            # Must contain both numbers and be a range (contains "to")
            if low in subtitle and high in subtitle and "to" in subtitle:
                return m

    return None


async def execute_midnight_stalk(live_mode: bool = False):
    """Execute the Midnight Stalk strategy."""
    now = datetime.now(TZ)

    print("="*70)
    print("  MIDNIGHT STALK - Rounding Arbitrage Execution")
    print("="*70)
    print(f"  Execution Time: {now.strftime('%Y-%m-%d %I:%M:%S %p')} ET")
    print(f"  Mode: {'LIVE' if live_mode else 'ANALYSIS'}")
    print("="*70)

    # Step 1: Get current temperature
    print("\n[1/4] Fetching KNYC current temperature...")
    try:
        temp_f, obs_time = await get_current_temp()
        obs_local = obs_time.astimezone(TZ)
        print(f"  Raw Temperature:  {temp_f:.1f}F")
        print(f"  Observation Time: {obs_local.strftime('%I:%M %p')} ET")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # Step 2: Apply rounding rule
    print("\n[2/4] Applying NWS rounding rule...")
    rounded = round_temp(temp_f)

    # Determine direction
    decimal_part = temp_f - int(temp_f)
    if decimal_part >= 0.5:
        direction = "UP"
        rounding_note = f"{temp_f:.1f}F >= x.50 -> rounds UP to {rounded}F"
    else:
        direction = "DOWN"
        rounding_note = f"{temp_f:.1f}F < x.50 -> rounds DOWN to {rounded}F"

    print(f"  Decimal Portion:  0.{int(decimal_part*10)}")
    print(f"  Rounding:         {rounding_note}")
    print(f"  Official High:    {rounded}F (if this is the max)")

    # Step 3: Identify target bracket
    print("\n[3/4] Identifying target bracket...")
    bracket_desc, bracket_low, bracket_high = get_target_bracket(rounded)

    if bracket_desc == "32_or_below":
        bracket_display = "32F or below"
    elif bracket_desc == "41_or_above":
        bracket_display = "41F or above"
    else:
        bracket_display = f"{bracket_low}F to {bracket_high}F"

    print(f"  Target Bracket:   {bracket_display}")

    # Step 4: Get market data
    print("\n[4/4] Fetching Kalshi market...")

    # Calculate tomorrow's date string (Midnight Stalk always targets next calendar day)
    from datetime import timedelta
    tomorrow = now.date() + timedelta(days=1)

    months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    target_date = f"{tomorrow.year % 100:02d}{months[tomorrow.month-1]}{tomorrow.day:02d}"
    print(f"  Target Date:      {tomorrow.strftime('%b %d, %Y')} ({target_date})")

    api_key = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

    if not api_key or not private_key_path:
        print("  ERROR: Missing Kalshi credentials")
        return

    client = KalshiClient(
        api_key_id=api_key,
        private_key_path=private_key_path,
        demo_mode=False
    )
    await client.start()

    try:
        market = await find_market(client, bracket_desc, target_date)

        if not market:
            print(f"  ERROR: No market found for {bracket_display} on {target_date}")
            return

        ticker = market.get("ticker", "")
        bid = market.get("yes_bid", 0)
        ask = market.get("yes_ask", 0)
        spread = ask - bid if ask and bid else 0

        # Smart entry
        if spread <= 5:
            entry = ask
            entry_note = f"tight spread ({spread}c) - taking ask"
        else:
            entry = bid + 1
            entry_note = f"wide spread ({spread}c) - pegging bid+1"

        print(f"  Ticker:           {ticker}")
        print(f"  Market:           Bid {bid}c / Ask {ask}c")
        print(f"  Entry Price:      {entry}c ({entry_note})")

        # Calculate position
        balance = await client.get_balance()
        max_position = balance * 0.15
        contracts = int(max_position / (entry / 100)) if entry > 0 else 0
        cost = contracts * entry / 100
        max_profit = contracts * (100 - entry) / 100

        print("\n" + "="*70)
        print("  TRADE TICKET")
        print("="*70)
        print(f"""
  OBSERVATION:     {temp_f:.1f}F @ {obs_local.strftime('%I:%M %p')} ET
  ROUNDING:        {direction} -> {rounded}F
  TARGET BRACKET:  {bracket_display}
  TICKER:          {ticker}

  ENTRY:           {entry}c
  CONTRACTS:       {contracts}
  COST:            ${cost:.2f}
  MAX PROFIT:      ${max_profit:.2f}

  >>> RECOMMENDATION: BUY {bracket_display} <<<
""")
        print("="*70)

        if not live_mode:
            print("\n[ANALYSIS MODE] No trade executed. Use --live for real trades.")
            return

        # Live execution
        response = input(f"\nExecute BUY {contracts} @ {entry}c? (y/n): ").strip().lower()

        if response != "y":
            print("[CANCELLED] Trade not executed.")
            return

        result = await client.place_order(
            ticker=ticker,
            side="yes",
            action="buy",
            count=contracts,
            price=entry,
            order_type="limit"
        )

        order_id = result.get("order", {}).get("order_id", "N/A")
        print(f"\n[EXECUTED] Order ID: {order_id}")
        print(f"  {contracts} contracts @ {entry}c = ${cost:.2f}")

    finally:
        await client.stop()


async def main():
    parser = argparse.ArgumentParser(description="Midnight Stalk - Rounding Arbitrage")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    args = parser.parse_args()

    await execute_midnight_stalk(live_mode=args.live)


if __name__ == "__main__":
    asyncio.run(main())
