#!/usr/bin/env python3
"""
Project Atlas - Lag Analyzer
Continuously logs Kraken and Limitless prices to analyze if exploitable lag exists.

Run for 30-60 minutes during volatile period, then analyze the CSV.

Usage:
    python3 -u lag_analyzer.py
"""

import asyncio
import csv
import time
from datetime import datetime
from typing import Optional

import aiohttp
import ccxt.pro as ccxtpro

from config import config


class LagAnalyzer:
    """Continuously logs CEX and prediction market prices for lag analysis."""

    def __init__(self):
        self.running = False

        # Latest prices (shared between tasks)
        self.kraken_price: float = 0.0
        self.kraken_timestamp: float = 0.0
        self.limitless_yes: float = 0.0
        self.limitless_no: float = 0.0
        self.limitless_timestamp: float = 0.0

        # Market info
        self.strike_price: float = 0.0
        self.market_slug: str = config.limitless_market_slug

        # CSV logging
        self.csv_file = None
        self.csv_writer = None

        # Stats
        self.kraken_updates = 0
        self.limitless_updates = 0

    async def initialize(self) -> bool:
        """Initialize and fetch market info."""
        print("\n=== Project Atlas: Lag Analyzer ===")
        print("Continuous price logging for lag analysis\n")

        # Get market info
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            url = f"https://api.limitless.exchange/markets/{self.market_slug}"
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"[ERROR] Could not fetch market: {response.status}")
                    return False

                data = await response.json()
                title = data.get("title", "Unknown")
                print(f"[INIT] Market: {title}")

                # Extract strike price from title
                import re
                match = re.search(r"\$?([\d,]+\.?\d*)", title)
                if match:
                    self.strike_price = float(match.group(1).replace(",", ""))
                    print(f"[INIT] Strike: ${self.strike_price:,.2f}")

        # Initialize CSV
        filename = f"lag_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_file = open(filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Timestamp",
            "Kraken_Price",
            "Kraken_Age_Ms",
            "Limitless_YES",
            "Limitless_NO",
            "Limitless_Age_Ms",
            "Price_vs_Strike",
            "Implied_Direction",
        ])
        self.csv_file.flush()
        print(f"[INIT] Logging to: {filename}\n")

        return True

    async def stream_kraken(self):
        """Stream real-time prices from Kraken."""
        exchange = ccxtpro.kraken({"enableRateLimit": True})

        print("[KRAKEN] Connecting to WebSocket...")

        try:
            while self.running:
                ticker = await exchange.watch_ticker("BTC/USD")
                price = ticker.get("last")

                if price:
                    self.kraken_price = price
                    self.kraken_timestamp = time.time()
                    self.kraken_updates += 1

                    # Log every 10th update to console
                    if self.kraken_updates % 10 == 0:
                        print(f"[KRAKEN] ${price:,.2f} (update #{self.kraken_updates})")

        except Exception as e:
            print(f"[KRAKEN ERROR] {e}")
        finally:
            await exchange.close()

    async def poll_limitless(self):
        """Poll Limitless prices."""
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }

        print("[LIMITLESS] Starting polling (every 3s)...")

        async with aiohttp.ClientSession(headers=headers) as session:
            while self.running:
                try:
                    url = f"https://api.limitless.exchange/markets/{self.market_slug}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            prices = data.get("prices", [0, 0])

                            self.limitless_no = float(prices[0])
                            self.limitless_yes = float(prices[1])
                            self.limitless_timestamp = time.time()
                            self.limitless_updates += 1

                            if self.limitless_updates % 5 == 0:
                                print(f"[LIMITLESS] YES: {self.limitless_yes:.4f}, NO: {self.limitless_no:.4f}")

                        elif response.status in (429, 1015):
                            print("[LIMITLESS] Rate limited, waiting 10s...")
                            await asyncio.sleep(10)
                            continue

                except Exception as e:
                    print(f"[LIMITLESS ERROR] {e}")

                await asyncio.sleep(3)  # Poll every 3 seconds

    async def log_continuously(self):
        """Log combined data every second."""
        print("[LOGGER] Starting continuous logging...\n")

        last_log = 0

        while self.running:
            now = time.time()

            # Log every second if we have data
            if now - last_log >= 1.0 and self.kraken_price > 0 and self.limitless_yes > 0:
                last_log = now

                # Calculate ages
                kraken_age_ms = (now - self.kraken_timestamp) * 1000
                limitless_age_ms = (now - self.limitless_timestamp) * 1000

                # Price vs strike
                price_vs_strike = self.kraken_price - self.strike_price

                # What direction should probability be?
                if price_vs_strike > 500:
                    implied = "STRONG_YES"
                elif price_vs_strike > 100:
                    implied = "YES"
                elif price_vs_strike < -500:
                    implied = "STRONG_NO"
                elif price_vs_strike < -100:
                    implied = "NO"
                else:
                    implied = "NEUTRAL"

                # Write to CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.csv_writer.writerow([
                    timestamp,
                    f"{self.kraken_price:.2f}",
                    f"{kraken_age_ms:.0f}",
                    f"{self.limitless_yes:.4f}",
                    f"{self.limitless_no:.4f}",
                    f"{limitless_age_ms:.0f}",
                    f"{price_vs_strike:+.2f}",
                    implied,
                ])
                self.csv_file.flush()

            await asyncio.sleep(0.5)

    async def run(self, duration_minutes: int = 30):
        """Run the analyzer for specified duration."""
        if not await self.initialize():
            return

        self.running = True

        print(f"[START] Running for {duration_minutes} minutes...")
        print("[START] Press Ctrl+C to stop early.\n")
        print("-" * 60)

        # Create tasks
        tasks = [
            asyncio.create_task(self.stream_kraken()),
            asyncio.create_task(self.poll_limitless()),
            asyncio.create_task(self.log_continuously()),
        ]

        try:
            # Run for specified duration
            await asyncio.sleep(duration_minutes * 60)
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user")
        finally:
            self.running = False

            # Wait for tasks to finish
            for task in tasks:
                task.cancel()

            await asyncio.sleep(1)

            if self.csv_file:
                self.csv_file.close()

            print("\n" + "=" * 60)
            print("[DONE] Analysis complete!")
            print(f"  Kraken updates: {self.kraken_updates}")
            print(f"  Limitless updates: {self.limitless_updates}")
            print(f"  CSV saved with ~{self.kraken_updates} rows")
            print("=" * 60)
            print("\nNext step: Run 'python3 analyze_lag.py' to analyze the data")


async def main():
    analyzer = LagAnalyzer()
    await analyzer.run(duration_minutes=30)


if __name__ == "__main__":
    asyncio.run(main())
