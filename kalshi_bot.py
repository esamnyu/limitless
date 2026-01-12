#!/usr/bin/env python3
"""
Kalshi Latency Arbitrage Bot - Same strategy as Atlas, legal for US/NYC.

Monitors CEX prices (Kraken) and exploits lag in Kalshi BTC markets.

Usage:
    # Paper trading (default):
    python3 kalshi_bot.py

    # Live trading (requires API keys):
    python3 kalshi_bot.py --live

Requirements:
    - Kalshi account (kalshi.com)
    - API key + private key (for live trading)
    - Set in .env: KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH
"""

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import ccxt.pro as ccxtpro
from dotenv import load_dotenv

from kalshi_client import KalshiClient
from alerts import AlertManager

load_dotenv()

SIGNALS_FILE = Path("kalshi_signals.jsonl")


@dataclass
class KalshiMarket:
    """A Kalshi prediction market."""
    ticker: str
    title: str
    strike: float
    expiry_ts: float
    yes_price: float = 0.0
    no_price: float = 0.0
    volume: int = 0
    updated_at: float = 0.0

    @property
    def ttl_min(self) -> float:
        return (self.expiry_ts - time.time()) / 60

    @property
    def tradeable(self) -> bool:
        return self.ttl_min > 5 and self.yes_price > 0


@dataclass
class Signal:
    """An arbitrage signal."""
    market: KalshiMarket
    action: str  # BUY_YES or BUY_NO
    cex_price: float
    velocity: float
    expected_prob: float
    market_prob: float
    edge: float
    profit_est: float
    ts: float = field(default_factory=time.time)


class KalshiBot:
    """
    Kalshi Latency Arbitrage Bot.

    Same strategy as Atlas (Limitless), adapted for Kalshi.
    Legal for US/NYC users.
    """

    VERSION = "1.0.0"

    def __init__(self, live_mode: bool = False):
        self.live_mode = live_mode
        self.monitor_only = not live_mode

        # Kalshi API credentials
        self.api_key_id = os.getenv("KALSHI_API_KEY_ID", "")
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

        # Strategy params (same as Atlas, but higher min_edge due to fees)
        self.velocity_threshold = 25.0  # $/sec trigger
        self.min_edge = 0.20  # 20% minimum (higher than Limitless due to ~1.2% fees)
        self.position_size = 10.0  # $10 trades
        self.momentum_confirm_sec = 2.0  # seconds to confirm momentum

        # State
        self.markets: dict[str, KalshiMarket] = {}
        self.cex_price = 0.0
        self.cex_velocity = 0.0
        self.price_history: list[tuple[float, float]] = []

        # Momentum confirmation
        self.momentum_start_time = 0.0
        self.momentum_direction = 0  # 1=bullish, -1=bearish, 0=none
        self.last_trigger_time = 0.0

        # Components
        self.client: Optional[KalshiClient] = None
        self.alerts = AlertManager()

        # Stats
        self.signals_detected = 0
        self.start_time = 0.0
        self.running = False

    async def start(self):
        """Initialize and start the bot."""
        self._print_banner()

        # Initialize Kalshi client
        self.client = KalshiClient(
            api_key_id=self.api_key_id,
            private_key_path=self.private_key_path,
            demo_mode=not self.live_mode,
        )
        await self.client.start()

        # Check exchange status
        status = await self.client.get_exchange_status()
        if not status:
            print("[ERROR] Could not connect to Kalshi API")
            return

        print(f"[INIT] Kalshi exchange status: {status.get('exchange_active', 'unknown')}")

        # Discover markets
        await self._discover_markets()
        if not self.markets:
            print("[WARN] No BTC markets found - will keep checking")

        print(f"[INIT] Tracking {len(self.markets)} markets")
        print(f"[INIT] Mode: {'LIVE' if self.live_mode else 'PAPER'}")
        print(f"[INIT] Min edge: {self.min_edge:.0%} (higher due to Kalshi fees)")

        # Check balance for live mode
        if self.live_mode:
            balance = await self.client.get_balance()
            if balance < self.position_size:
                print(f"[ERROR] Insufficient balance: ${balance:.2f}")
                return
            print(f"[INIT] Kalshi balance: ${balance:.2f}")

        self.running = True
        self.start_time = time.time()

        print("\n[START] Kalshi bot running. Press Ctrl+C to stop.\n")
        print("=" * 70)

        # Send Discord alert
        mode = "LIVE" if self.live_mode else "PAPER"
        await self.alerts.bot_started(len(self.markets), f"Kalshi {mode}")

        await self._run()

    async def stop(self):
        """Graceful shutdown."""
        self.running = False

        await self.alerts.bot_stopped("Manual stop")

        if self.client:
            await self.client.stop()

        self._print_summary()

    def _print_banner(self):
        """Print startup banner."""
        print("\n" + "=" * 70)
        print("""
    ██╗  ██╗ █████╗ ██╗     ███████╗██╗  ██╗██╗
    ██║ ██╔╝██╔══██╗██║     ██╔════╝██║  ██║██║
    █████╔╝ ███████║██║     ███████╗███████║██║
    ██╔═██╗ ██╔══██║██║     ╚════██║██╔══██║██║
    ██║  ██╗██║  ██║███████╗███████║██║  ██║██║
    ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝
                    ARBITRAGE BOT v{version}
              US-Regulated Prediction Markets
        """.format(version=self.VERSION))
        print("=" * 70)

    def _print_summary(self):
        """Print session summary."""
        runtime = time.time() - self.start_time

        print("\n" + "=" * 70)
        print("                      SESSION SUMMARY")
        print("=" * 70)
        print(f"  Runtime:           {runtime/60:.1f} minutes")
        print(f"  Markets tracked:   {len(self.markets)}")
        print(f"  Signals detected:  {self.signals_detected}")
        print("=" * 70)

    async def _discover_markets(self):
        """Find BTC price prediction markets on Kalshi."""
        print("[SCAN] Discovering Kalshi BTC markets...")

        try:
            markets = await self.client.get_btc_markets()

            for m in markets:
                parsed = self.client.parse_market(m)

                if parsed["strike"] <= 0:
                    continue

                # Skip expired or soon-to-expire
                ttl_min = (parsed["expiry_ts"] - time.time()) / 60
                if ttl_min < 5:
                    continue

                ticker = parsed["ticker"]
                if ticker in self.markets:
                    continue

                market = KalshiMarket(
                    ticker=ticker,
                    title=parsed["title"],
                    strike=parsed["strike"],
                    expiry_ts=parsed["expiry_ts"],
                    yes_price=parsed["yes_price"],
                    no_price=parsed["no_price"],
                    volume=parsed["volume"],
                    updated_at=time.time(),
                )

                self.markets[ticker] = market
                print(f"  [+] {ticker}: ${market.strike:,.0f} | "
                      f"YES:{market.yes_price:.0%} NO:{market.no_price:.0%} | "
                      f"{market.ttl_min:.0f}min")

        except Exception as e:
            print(f"[ERROR] Market discovery failed: {e}")

    async def _run(self):
        """Main run loop."""
        tasks = [
            asyncio.create_task(self._cex_stream()),
            asyncio.create_task(self._market_updater()),
            asyncio.create_task(self._status_display()),
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    async def _cex_stream(self):
        """Stream BTC price from Kraken and detect velocity triggers."""
        exchange = ccxtpro.kraken({"enableRateLimit": True})
        print("[CEX] Streaming BTC/USD from Kraken...")

        try:
            while self.running:
                ticker = await exchange.watch_ticker("BTC/USD")
                now = time.time()
                price = ticker.get("last", 0)

                if price > 0:
                    self.cex_price = price
                    self.price_history.append((now, price))

                    # Keep last 5 seconds
                    self.price_history = [
                        (t, p) for t, p in self.price_history if now - t < 5
                    ]

                    # Calculate velocity
                    if len(self.price_history) >= 2:
                        old_t, old_p = self.price_history[0]
                        dt = now - old_t
                        if dt > 0.3:
                            self.cex_velocity = (price - old_p) / dt
                            await self._check_momentum(now)

        except Exception as e:
            print(f"[CEX ERROR] {e}")
        finally:
            await exchange.close()

    async def _check_momentum(self, now: float):
        """Check and confirm momentum before triggering."""
        velocity = self.cex_velocity
        current_direction = 1 if velocity > 0 else -1 if velocity < 0 else 0
        velocity_exceeds = abs(velocity) >= self.velocity_threshold

        # Cooldown: don't trigger too frequently
        if now - self.last_trigger_time < 10:
            return

        if velocity_exceeds:
            if self.momentum_direction == 0:
                self.momentum_start_time = now
                self.momentum_direction = current_direction
                print(f"[MOMENTUM] {'BULLISH' if current_direction > 0 else 'BEARISH'} "
                      f"(${velocity:+.1f}/s)")

            elif self.momentum_direction == current_direction:
                elapsed = now - self.momentum_start_time
                if elapsed >= self.momentum_confirm_sec:
                    print(f"[MOMENTUM] Confirmed after {elapsed:.1f}s")
                    self.last_trigger_time = now
                    self.momentum_direction = 0
                    await self._on_trigger()
            else:
                self.momentum_direction = 0
        else:
            if self.momentum_direction != 0:
                self.momentum_direction = 0

    async def _on_trigger(self):
        """Handle velocity trigger."""
        velocity = self.cex_velocity
        price = self.cex_price
        direction = "BULLISH" if velocity > 0 else "BEARISH"
        print(f"\n[TRIGGER] {direction} ${velocity:+.1f}/s @ ${price:,.2f}")

        # Find best opportunity
        best: Optional[Signal] = None

        for market in self.markets.values():
            if not market.tradeable:
                continue

            signal = self._analyze(market)
            if signal and signal.edge >= self.min_edge:
                if not best or signal.edge > best.edge:
                    best = signal

        if not best:
            print("[TRIGGER] No opportunities above threshold")
            return

        self.signals_detected += 1

        print(f"\n[SIGNAL #{self.signals_detected}] {best.action}")
        print(f"  Market: {best.market.title[:45]}...")
        print(f"  Strike: ${best.market.strike:,.2f}")
        print(f"  Expected: {best.expected_prob:.0%} | Actual: {best.market_prob:.0%}")
        print(f"  Edge: {best.edge:+.1%}")
        print(f"  Est. Profit: ${best.profit_est:+.2f}")

        # Send Discord alert
        await self.alerts.signal_alert(
            action=best.action,
            asset="BTC",
            edge=best.edge,
            profit_est=best.profit_est,
            velocity=best.velocity,
        )

        # Log signal
        self._log_signal(best)

        if not self.monitor_only:
            await self._execute(best)

    def _analyze(self, market: KalshiMarket) -> Optional[Signal]:
        """Analyze market for opportunity."""
        if self.cex_price <= 0:
            return None

        diff = self.cex_price - market.strike
        pct_diff = diff / self.cex_price

        # Expected probability based on price vs strike
        if pct_diff > 0.01:
            exp = 0.98
        elif pct_diff > 0.005:
            exp = 0.85
        elif pct_diff > 0.002:
            exp = 0.70
        elif pct_diff > 0.0005:
            exp = 0.58
        elif pct_diff > -0.0005:
            exp = 0.50
        elif pct_diff > -0.002:
            exp = 0.42
        elif pct_diff > -0.005:
            exp = 0.30
        elif pct_diff > -0.01:
            exp = 0.15
        else:
            exp = 0.02

        if self.cex_velocity > 0:  # Bullish -> buy YES
            edge = exp - market.yes_price
            if edge > 0:
                profit = self.position_size * edge - 1.5  # Account for ~1.2% fees
                return Signal(
                    market=market,
                    action="BUY_YES",
                    cex_price=self.cex_price,
                    velocity=self.cex_velocity,
                    expected_prob=exp,
                    market_prob=market.yes_price,
                    edge=edge,
                    profit_est=profit,
                )
        else:  # Bearish -> buy NO
            edge = (1 - exp) - market.no_price
            if edge > 0:
                profit = self.position_size * edge - 1.5
                return Signal(
                    market=market,
                    action="BUY_NO",
                    cex_price=self.cex_price,
                    velocity=self.cex_velocity,
                    expected_prob=1 - exp,
                    market_prob=market.no_price,
                    edge=edge,
                    profit_est=profit,
                )

        return None

    async def _execute(self, signal: Signal):
        """Execute trade on Kalshi."""
        print("[EXECUTE] Placing order...")

        try:
            # Determine side and price
            if signal.action == "BUY_YES":
                side = "yes"
                price = int(signal.market_prob * 100)  # Convert to cents
            else:
                side = "no"
                price = int(signal.market.no_price * 100)

            # Calculate contracts
            contracts = int(self.position_size / (price / 100))

            result = await self.client.place_order(
                ticker=signal.market.ticker,
                side=side,
                action="buy",
                count=contracts,
                price=price,
                order_type="limit",
            )

            if result:
                print(f"[EXECUTE] Order placed: {result.get('order_id', 'unknown')}")
            else:
                print("[EXECUTE] Order failed")

        except Exception as e:
            print(f"[EXECUTE ERROR] {e}")

    def _log_signal(self, signal: Signal):
        """Log signal to file."""
        with open(SIGNALS_FILE, "a") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(),
                "platform": "kalshi",
                "market": signal.market.ticker,
                "action": signal.action,
                "cex_price": signal.cex_price,
                "velocity": signal.velocity,
                "edge": signal.edge,
                "profit_est": signal.profit_est,
            }) + "\n")

    async def _market_updater(self):
        """Update market prices periodically."""
        while self.running:
            await asyncio.sleep(2)  # Poll every 2 seconds

            for ticker, market in list(self.markets.items()):
                try:
                    # Get orderbook for current prices
                    orderbook = await self.client.get_orderbook(ticker)

                    yes_bids = orderbook.get("yes", [])
                    no_bids = orderbook.get("no", [])

                    if yes_bids:
                        market.yes_price = yes_bids[0][0] / 100.0
                    if no_bids:
                        market.no_price = no_bids[0][0] / 100.0

                    market.updated_at = time.time()

                except Exception:
                    pass

            # Remove expired markets
            expired = [t for t, m in self.markets.items() if m.ttl_min < 5]
            for t in expired:
                del self.markets[t]

            # Discover new markets periodically
            if len(self.markets) < 3:
                await self._discover_markets()

    async def _status_display(self):
        """Display status periodically."""
        while self.running:
            await asyncio.sleep(15)

            if self.cex_price > 0:
                print(f"\n[STATUS] BTC: ${self.cex_price:,.0f} | "
                      f"Vel: ${self.cex_velocity:+.1f}/s | "
                      f"Markets: {len(self.markets)} | "
                      f"Signals: {self.signals_detected}")

                for m in list(self.markets.values())[:3]:
                    diff = self.cex_price - m.strike
                    print(f"  ${m.strike:,.0f} ({diff:+,.0f}) "
                          f"YES:{m.yes_price:.0%} NO:{m.no_price:.0%} "
                          f"[{m.ttl_min:.0f}m]")


async def main():
    parser = argparse.ArgumentParser(description="Kalshi Latency Arbitrage Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    args = parser.parse_args()

    bot = KalshiBot(live_mode=args.live)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
