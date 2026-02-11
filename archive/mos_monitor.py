#!/usr/bin/env python3
"""MOS Release Monitor - Watch for 10:51 PM NAM data drop."""

import asyncio
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import aiohttp

# Load env
env_path = Path("/Users/miqadmin/Documents/limitless/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

from kalshi_client import KalshiClient

TICKER = "KXHIGHNY-26JAN22-B42.5"
MOS_MET_URL = "https://tgftp.nws.noaa.gov/data/forecasts/mos/nam/short/met/knyc.txt"
TZ = ZoneInfo("America/New_York")


async def fetch_mos():
    """Fetch NAM MOS (MET) bulletin."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(MOS_MET_URL, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.text()
    except Exception as e:
        return None
    return None


def parse_mos_high(text):
    """Parse max temp from MOS bulletin."""
    if not text:
        return None
    try:
        for line in text.split('\n'):
            if line.strip().startswith('X/N') or line.strip().startswith('N/X'):
                parts = line.split()
                for p in parts[1:]:
                    try:
                        return int(p)
                    except:
                        continue
    except:
        pass
    return None


async def main():
    api_key = os.getenv('KALSHI_API_KEY_ID')
    private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')

    client = KalshiClient(api_key_id=api_key, private_key_path=private_key_path, demo_mode=False)
    await client.start()

    print("=" * 60)
    print("  MOS MONITOR - WATCHING FOR 10:51 PM DATA DROP")
    print("=" * 60)

    last_mos_high = None
    last_bid = 0
    last_ask = 0
    check_count = 0

    while True:
        check_count += 1
        now = datetime.now(TZ)
        time_str = now.strftime("%I:%M:%S %p")

        # Fetch MOS
        mos_text = await fetch_mos()
        mos_high = parse_mos_high(mos_text)

        # Fetch orderbook
        ob = await client.get_orderbook(TICKER)
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])
        best_bid = yes_bids[0][0] if yes_bids else 0
        best_ask = 100 - no_bids[0][0] if no_bids else 100

        # Check positions and fills
        positions = await client.get_positions()
        current_pos = 0
        for pos in positions:
            if "B42.5" in pos.get("ticker", "") and "26JAN22" in pos.get("ticker", ""):
                current_pos = pos.get("position", 0)

        # Check resting sell orders
        orders = await client.get_orders(status="resting")
        sell_88 = 0
        sell_99 = 0
        for o in orders:
            if "B42.5" in o.get("ticker", "") and o.get("action") == "sell":
                price = o.get("yes_price", 0)
                remaining = o.get("remaining_count", 0)
                if price == 88:
                    sell_88 = remaining
                elif price == 99:
                    sell_99 = remaining

        # Detect changes
        mos_changed = mos_high != last_mos_high and mos_high is not None
        price_moved = abs(best_bid - last_bid) >= 2 or abs(best_ask - last_ask) >= 2

        # Print status
        print(f"\n[{time_str}] Check #{check_count}")
        print(f"  MOS High:    {mos_high}°F" + (" *** UPDATED ***" if mos_changed else ""))
        print(f"  Orderbook:   Bid {best_bid}¢ / Ask {best_ask}¢" + (" *** MOVED ***" if price_moved else ""))
        print(f"  Position:    {current_pos} contracts")
        print(f"  Sell Orders: {sell_88}x@88¢ | {sell_99}x@99¢")

        # Alert on significant events
        if mos_changed and last_mos_high is not None:
            print("\n" + "!" * 60)
            print(f"  !!! MOS UPDATE: {last_mos_high}°F → {mos_high}°F !!!")
            if mos_high <= 43:
                print(f"  !!! THESIS CONFIRMED - WIND WALL IN EFFECT !!!")
            print("!" * 60)

        if best_bid >= 50 and last_bid < 50:
            print("\n" + "*" * 60)
            print(f"  *** PRICE SPIKE: BID NOW {best_bid}¢ ***")
            print("*" * 60)

        if sell_88 < 50 and sell_88 > 0:
            filled_88 = 50 - sell_88
            print(f"\n  $$$ SAFETY TIER FILLING: {filled_88}/50 sold @ 88¢ $$$")

        # Update last values
        last_mos_high = mos_high if mos_high else last_mos_high
        last_bid = best_bid
        last_ask = best_ask

        # Check if we should stop (after 11:15 PM or if significant event)
        if now.hour == 23 and now.minute >= 15:
            print("\n" + "=" * 60)
            print("  MONITORING COMPLETE - Past 11:15 PM")
            print("=" * 60)
            break

        # Wait 30 seconds between checks
        await asyncio.sleep(30)

    await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
