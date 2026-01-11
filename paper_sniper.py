#!/usr/bin/env python3
"""
Project Atlas - Phase 1: Paper Sniper
Monitors Binance Futures and Limitless to detect theoretical arbitrage opportunities.

Supports both CLOB markets (via SDK) and Simple markets (via direct API).

Usage:
    python paper_sniper.py

Output:
    Console logs + CSV file with theoretical profit calculations.
"""

import asyncio
import csv
import time
from datetime import datetime
from typing import Optional

import aiohttp
import ccxt.pro as ccxtpro

from config import config
from strategy import VelocityDetector, ArbitrageAnalyzer, ArbitrageSignal


class LimitlessClient:
    """
    Flexible client for Limitless API.
    Handles both Simple markets and CLOB markets.
    """

    def __init__(self, base_url: str = "https://api.limitless.exchange"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self._market_cache: dict = {}
        self._cache_time: float = 0

    async def _ensure_session(self):
        if self.session is None:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json",
            }
            self.session = aiohttp.ClientSession(headers=headers)

    async def close(self):
        if self.session:
            await self.session.close()

    async def get_simple_market(self, market_id: str, use_cache: bool = True) -> dict:
        """Fetch a simple market by ID with caching and retry."""
        await self._ensure_session()

        # Return cached data if fresh (within 120 seconds to reduce rate limits)
        now = time.time()
        if use_cache and market_id in self._market_cache and (now - self._cache_time) < 120:
            return self._market_cache[market_id]

        endpoint = f"{self.base_url}/markets/{market_id}"

        # Retry with backoff
        for attempt in range(3):
            try:
                async with self.session.get(endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._market_cache[market_id] = data
                        self._cache_time = time.time()
                        return data
                    elif response.status == 429 or response.status == 1015:
                        wait = (attempt + 1) * 5
                        print(f"[RATE LIMIT] Waiting {wait}s before retry...")
                        await asyncio.sleep(wait)
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2)
                continue

        raise Exception(f"Could not fetch market {market_id} from any endpoint")

    async def find_btc_market(self) -> Optional[str]:
        """Find an active BTC market expiring in the future."""
        await self._ensure_session()

        for attempt in range(3):
            try:
                async with self.session.get(f"{self.base_url}/markets/active?limit=25") as response:
                    if response.status == 200:
                        data = await response.json()
                        now = time.time() * 1000  # ms

                        for m in data.get("data", []):
                            exp_ts = m.get("expirationTimestamp", 0)
                            title = m.get("title", "")

                            # Find BTC market expiring 30+ minutes from now
                            if "BTC" in title and exp_ts > now + 1800000:
                                return m.get("slug")
                    elif response.status in (429, 1015):
                        wait = (attempt + 1) * 5
                        print(f"[RATE LIMIT] Waiting {wait}s...")
                        await asyncio.sleep(wait)
            except Exception as e:
                print(f"[WARN] Attempt {attempt+1} failed: {e}")
                await asyncio.sleep(3)

        return None

    async def get_orderbook(self, market_id: str) -> dict:
        """Fetch orderbook for a market."""
        await self._ensure_session()

        endpoints = [
            f"{self.base_url}/api-v1/simple/markets/{market_id}/orderbook",
            f"{self.base_url}/api-v1/markets/{market_id}/orderbook",
            f"{self.base_url}/simple/markets/{market_id}/orderbook",
        ]

        for endpoint in endpoints:
            try:
                async with self.session.get(endpoint) as response:
                    if response.status == 200:
                        return await response.json()
            except Exception:
                continue

        # If no orderbook endpoint, try to get prices from market data
        return {"bids": [], "asks": []}

    async def get_market_prices(self, market_id: str) -> tuple[float, float]:
        """
        Get current YES/NO prices for a simple market.
        Returns (yes_price, no_price) as probabilities (0-1).
        """
        await self._ensure_session()

        # For simple markets, prices might be in the market data itself
        market = await self.get_simple_market(market_id)

        # Try to extract prices from market data
        yes_price = 0.0
        no_price = 0.0

        # Limitless API returns prices as array: [NO_price, YES_price]
        if "prices" in market and isinstance(market["prices"], list):
            prices = market["prices"]
            if len(prices) >= 2:
                no_price = float(prices[0])
                yes_price = float(prices[1])
        elif "yesPrice" in market:
            yes_price = float(market["yesPrice"])
            no_price = float(market.get("noPrice", 1 - yes_price))

        return yes_price, no_price


class PaperSniper:
    """
    Phase 1: Paper trading bot that logs theoretical arbitrage opportunities.
    No real trades are executed.
    """

    def __init__(self):
        self.config = config
        self.velocity_detector = VelocityDetector(
            threshold_usd=config.velocity_threshold_usd,
            window_size=config.price_history_window,
        )
        self.analyzer: Optional[ArbitrageAnalyzer] = None

        # State
        self.running = False
        self.current_binance_price: float = 0.0
        self.current_yes_price: float = 0.0  # Probability of YES outcome
        self.current_no_price: float = 0.0   # Probability of NO outcome
        self.market_expiry: Optional[datetime] = None
        self.market_slug: str = ""  # Actual market slug being used

        # Limitless client
        self.client: Optional[LimitlessClient] = None
        self.market_data: Optional[dict] = None

        # CSV logging
        self.csv_file = None
        self.csv_writer = None

    async def initialize(self) -> bool:
        """Initialize connections and validate configuration."""
        print("\n=== Project Atlas: Paper Sniper ===")
        print("Phase 1 - Data Collection Mode\n")

        # Validate config
        if not self.config.validate():
            return False

        print(self.config)

        # Initialize Limitless client
        print("[INIT] Connecting to Limitless API...")
        self.client = LimitlessClient(self.config.limitless_api_url)

        # Fetch market data - try configured slug first, then auto-discover
        market_slug = self.config.limitless_market_slug

        try:
            self.market_data = await self.client.get_simple_market(market_slug, use_cache=False)
        except Exception as e:
            print(f"[WARN] Configured market failed: {e}")
            print("[INFO] Auto-discovering active BTC market...")

            market_slug = await self.client.find_btc_market()
            if not market_slug:
                print("[ERROR] No active BTC markets found")
                return False

            print(f"[INFO] Found market: {market_slug}")
            self.market_data = await self.client.get_simple_market(market_slug, use_cache=False)

        # Store the actual slug being used
        self.market_slug = market_slug

        title = self.market_data.get("title", self.market_data.get("name", "Unknown"))
        print(f"[INIT] Market loaded: {title}")

        # Extract expiry time (prefer timestamp over date string)
        if "expirationTimestamp" in self.market_data:
            ts = self.market_data["expirationTimestamp"]
            if ts > 1e12:
                ts = ts / 1000
            self.market_expiry = datetime.fromtimestamp(ts)
            print(f"[INIT] Expiry: {self.market_expiry}")

        # Set up analyzer with baseline price as strike
        baseline_price = self._extract_baseline_price()
        self.analyzer = ArbitrageAnalyzer(strike_price=baseline_price)
        print(f"[INIT] Strike/Baseline price: ${baseline_price:,.2f}")

        # Initialize CSV logging
        if self.config.log_to_csv:
            self._init_csv()

        return True

    def _extract_baseline_price(self) -> float:
        """Extract baseline/strike price from market data."""
        import re

        # Try direct field
        for field in ["baselinePrice", "strikePrice", "targetPrice"]:
            if field in self.market_data:
                return float(self.market_data[field])

        # Try to parse from title
        title = self.market_data.get("title", self.market_data.get("name", ""))
        match = re.search(r"\$?([\d,]+\.?\d*)", title)
        if match:
            price_str = match.group(1).replace(",", "")
            try:
                return float(price_str)
            except ValueError:
                pass

        print("[WARN] Could not extract baseline price, using $90,000")
        return 90000.0

    def _init_csv(self) -> None:
        """Initialize CSV file for logging."""
        self.csv_file = open(self.config.csv_filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Timestamp",
            "Binance_Price",
            "Binance_Velocity",
            "Limitless_YES_Price",
            "Limitless_NO_Price",
            "Signal_Type",
            "True_Prob",
            "Market_Prob",
            "Theoretical_Profit",
            "Time_To_Expiry_Min",
        ])
        self.csv_file.flush()  # Ensure header is written immediately
        print(f"[INIT] CSV logging to: {self.config.csv_filename}")

    def _check_expiry_safety(self) -> tuple[bool, float]:
        """Check if we're too close to expiry."""
        if self.market_expiry is None:
            return True, float("inf")

        now = datetime.now(self.market_expiry.tzinfo)
        time_to_expiry = (self.market_expiry - now).total_seconds() / 60

        is_safe = time_to_expiry >= self.config.min_time_to_expiry_minutes
        return is_safe, time_to_expiry

    def _log_signal(self, signal: ArbitrageSignal, time_to_expiry: float) -> None:
        """Log an arbitrage signal to console and CSV."""
        timestamp = datetime.fromtimestamp(signal.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        profit_color = "\033[92m" if signal.theoretical_profit > 0 else "\033[91m"
        reset_color = "\033[0m"

        print(f"\n{'='*60}")
        print(f"[SIGNAL] {signal.signal_type} @ {timestamp}")
        print(f"  Binance Price:   ${signal.binance_price:,.2f}")
        print(f"  Velocity:        ${signal.binance_velocity:+,.2f}/sec")
        print(f"  YES Price:       {self.current_yes_price:.4f}")
        print(f"  NO Price:        {self.current_no_price:.4f}")
        print(f"  True Prob:       {signal.true_prob:.4f}")
        print(f"  Market Prob:     {signal.market_prob:.4f}")
        print(f"  Time to Expiry:  {time_to_expiry:.1f} min")
        print(f"  {profit_color}THEORETICAL PROFIT: ${signal.theoretical_profit:+,.2f}{reset_color}")
        print(f"{'='*60}\n")

        if self.csv_writer:
            self.csv_writer.writerow([
                timestamp,
                f"{signal.binance_price:.2f}",
                f"{signal.binance_velocity:.2f}",
                f"{self.current_yes_price:.4f}",
                f"{self.current_no_price:.4f}",
                signal.signal_type,
                f"{signal.true_prob:.4f}",
                f"{signal.market_prob:.4f}",
                f"{signal.theoretical_profit:.2f}",
                f"{time_to_expiry:.1f}",
            ])
            self.csv_file.flush()

    async def _poll_limitless(self) -> None:
        """Poll Limitless for current prices."""
        poll_count = 0

        while self.running:
            try:
                yes_price, no_price = await self.client.get_market_prices(
                    self.market_slug
                )

                if yes_price > 0:
                    self.current_yes_price = yes_price
                if no_price > 0:
                    self.current_no_price = no_price

                # Also try orderbook
                try:
                    orderbook = await self.client.get_orderbook(
                        self.market_slug
                    )
                    bids = orderbook.get("bids", [])
                    asks = orderbook.get("asks", [])

                    if asks:
                        self.current_yes_price = float(asks[0].get("price", self.current_yes_price))
                    if bids:
                        self.current_no_price = float(bids[0].get("price", self.current_no_price))
                except Exception:
                    pass  # Orderbook might not be available for simple markets

                poll_count += 1
                if poll_count % 4 == 0:  # Output every 4 polls (20 seconds)
                    print(f"[POLL] Limitless - YES: {self.current_yes_price:.4f}, NO: {self.current_no_price:.4f}")

            except Exception as e:
                print(f"[ERROR] Limitless poll failed: {e}")

            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _stream_binance(self) -> None:
        """Stream BTC prices via WebSocket (uses Kraken due to Binance geo-restrictions)."""
        # Use Kraken instead of Binance due to geo-restrictions
        exchange = ccxtpro.kraken({
            "enableRateLimit": True,
        })

        symbol = "BTC/USD"  # Kraken uses BTC/USD
        print(f"[STREAM] Connecting to Kraken: {symbol}")

        tick_count = 0
        try:
            while self.running:
                ticker = await exchange.watch_ticker(symbol)
                price = ticker.get("last")

                if price is None:
                    continue

                self.current_binance_price = price
                tick_count += 1

                # Log price every 50 ticks (~10-20 seconds)
                if tick_count % 50 == 0:
                    print(f"[PRICE] Kraken BTC: ${price:,.2f}")

                # Check for velocity trigger
                signal_result = self.velocity_detector.add_price(price)

                if signal_result:
                    velocity, direction = signal_result
                    print(f"[TRIGGER] {direction} velocity detected: ${velocity:+,.2f}/sec")

                    # Check expiry safety
                    is_safe, time_to_expiry = self._check_expiry_safety()
                    if not is_safe:
                        print(f"[SKIP] Too close to expiry ({time_to_expiry:.1f} min)")
                        continue

                    # Analyze opportunity
                    # For simple markets: YES price = ask, NO price = bid
                    if self.analyzer and self.current_yes_price > 0:
                        signal = self.analyzer.analyze_opportunity(
                            binance_price=self.current_binance_price,
                            binance_velocity=velocity,
                            limitless_best_ask=self.current_yes_price,
                            limitless_best_bid=self.current_no_price,
                        )

                        if signal:
                            self._log_signal(signal, time_to_expiry)

        except Exception as e:
            print(f"[ERROR] Binance stream error: {e}")
        finally:
            await exchange.close()

    async def run(self) -> None:
        """Main run loop."""
        if not await self.initialize():
            print("[FATAL] Initialization failed. Exiting.")
            return

        self.running = True
        print("\n[START] Paper Sniper running... Press Ctrl+C to stop.\n")

        try:
            await asyncio.gather(
                self._stream_binance(),
                self._poll_limitless(),
            )
        except KeyboardInterrupt:
            print("\n[STOP] Shutting down...")
        finally:
            self.running = False
            if self.csv_file:
                self.csv_file.close()
                print(f"[STOP] Log saved to: {self.config.csv_filename}")
            if self.client:
                await self.client.close()


async def main():
    sniper = PaperSniper()
    await sniper.run()


if __name__ == "__main__":
    asyncio.run(main())
