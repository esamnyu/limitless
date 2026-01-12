#!/usr/bin/env python3
"""
Weather Trading Bot - High Win Rate Strategy for Kalshi NYC Weather Markets.

Compares GFS and ECMWF model forecasts to Kalshi market prices to find mispricings.
When models agree but market prices a different bracket, we have an edge.

Win Rate Target: 75-85%

Usage:
    # Paper trading (default) - logs trades without execution
    python3 weather_bot.py

    # Live trading
    python3 weather_bot.py --live

    # One-time scan
    python3 weather_bot.py --once
"""

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from weather_client import WeatherClient, Forecast
from kalshi_client import KalshiClient
from alerts import AlertManager

load_dotenv()

# Log files
PAPER_TRADES_LOG = Path("weather_paper_trades.jsonl")
RESULTS_LOG = Path("weather_results.jsonl")


@dataclass
class WeatherOpportunity:
    """A weather trading opportunity."""

    date: str  # Settlement date (YYYY-MM-DD)
    ticker: str  # Kalshi ticker

    # Model predictions
    gfs_temp: float
    ecmwf_temp: float
    model_avg: float
    model_bracket: str  # e.g., "B39.5"
    model_bracket_range: str  # e.g., "39-40°F"
    model_confidence: float

    # Market state
    market_yes_price: float  # Current YES price for model bracket
    market_favorite: str  # Bracket market favors
    market_favorite_price: float

    # Edge calculation
    edge: float  # model_confidence - market_yes_price
    expected_value: float  # edge * potential_win

    # Timing
    hours_to_settlement: float
    ts: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())

    @property
    def potential_return(self) -> float:
        """Potential return if we win (assuming $1 payout)."""
        if self.market_yes_price > 0:
            return (1.0 / self.market_yes_price) - 1.0
        return 0.0

    @property
    def is_tradeable(self) -> bool:
        """Check if this opportunity meets all entry criteria."""
        return (
            self.edge >= 0.20  # 20% minimum edge
            and self.hours_to_settlement >= 6  # Not too close to settlement
            and self.model_bracket != self.market_favorite  # Market disagrees
        )


@dataclass
class PaperTrade:
    """A paper trade for backtesting."""

    opportunity: WeatherOpportunity
    contracts: int
    entry_price: float
    entry_ts: float

    # Settlement (filled in later)
    settled: bool = False
    actual_temp: Optional[float] = None
    actual_bracket: Optional[str] = None
    pnl: float = 0.0

    def settle(self, actual_temp: float, actual_bracket: str):
        """Settle the paper trade with actual NWS data."""
        self.settled = True
        self.actual_temp = actual_temp
        self.actual_bracket = actual_bracket

        won = actual_bracket == self.opportunity.model_bracket
        if won:
            self.pnl = self.contracts * (1.0 - self.entry_price)
        else:
            self.pnl = -self.contracts * self.entry_price


class WeatherBot:
    """
    Weather trading bot for Kalshi KXHIGHNY markets.

    Strategy:
    1. Get GFS and ECMWF forecasts for NYC
    2. Check if models agree (within 2°F)
    3. Get Kalshi market prices for matching date
    4. If model bracket != market favorite and edge > 20%: TRADE
    """

    VERSION = "1.0.0"

    # Kalshi KXHIGHNY bracket mappings (NOTE: brackets are dynamic per day)
    # These are just examples - actual brackets are parsed from market data
    BRACKETS = {
        "T39": {"range": "<39°F", "min": float("-inf"), "max": 39},
        "B39.5": {"range": "39-40°F", "min": 39, "max": 41},
        "B41.5": {"range": "41-42°F", "min": 41, "max": 43},
        "B43.5": {"range": "43-44°F", "min": 43, "max": 45},
        "B45.5": {"range": "45-46°F", "min": 45, "max": 47},
        "T46": {"range": ">46°F", "min": 47, "max": float("inf")},
    }

    # Month abbreviations for ticker parsing
    MONTHS = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }

    # Trading parameters
    MIN_EDGE = 0.20  # 20% minimum edge
    MIN_HOURS_TO_SETTLEMENT = 6
    MAX_POSITION_SIZE = 25.0  # $25 max per trade
    MAX_DIVERGENCE = 2.0  # Max °F difference between models

    def __init__(self, live_mode: bool = False):
        self.live_mode = live_mode
        self.paper_mode = not live_mode

        # API credentials
        self.kalshi_key = os.getenv("KALSHI_API_KEY_ID", "")
        self.kalshi_pem = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

        # Clients
        self.weather: Optional[WeatherClient] = None
        self.kalshi: Optional[KalshiClient] = None
        self.alerts = AlertManager()

        # State
        self.running = False
        self.start_time = 0.0

        # Paper trading
        self.paper_trades: list[PaperTrade] = []
        self.paper_wins = 0
        self.paper_losses = 0
        self.paper_pnl = 0.0

        # Stats
        self.opportunities_found = 0
        self.trades_executed = 0

    async def start(self):
        """Initialize and start the bot."""
        self._print_banner()

        # Initialize clients
        self.weather = WeatherClient()
        await self.weather.start()

        self.kalshi = KalshiClient(
            api_key_id=self.kalshi_key,
            private_key_path=self.kalshi_pem,
            demo_mode=False,
        )
        await self.kalshi.start()

        # Check Kalshi balance
        if self.live_mode:
            balance = await self.kalshi.get_balance()
            print(f"[INIT] Kalshi balance: ${balance:.2f}")
            if balance < 10:
                print("[WARN] Low balance - switching to paper mode")
                self.live_mode = False
                self.paper_mode = True

        mode = "LIVE" if self.live_mode else "PAPER"
        print(f"[INIT] Mode: {mode}")
        print(f"[INIT] Min edge: {self.MIN_EDGE:.0%}")
        print(f"[INIT] Max position: ${self.MAX_POSITION_SIZE:.2f}")

        self.running = True
        self.start_time = datetime.now(timezone.utc).timestamp()

        # Load existing paper trades
        self._load_paper_trades()

        print("\n" + "=" * 70)
        print("[START] Weather trading bot running...")
        print("=" * 70)

        await self.alerts.send(
            "Weather Bot Started",
            f"Mode: {mode}\nMin Edge: {self.MIN_EDGE:.0%}",
            color=0x00FF00,
        )

    async def stop(self):
        """Graceful shutdown."""
        self.running = False

        runtime = datetime.now(timezone.utc).timestamp() - self.start_time

        # Save paper trades
        self._save_paper_trades()

        await self.alerts.send(
            "Weather Bot Stopped",
            f"Runtime: {runtime/60:.1f}min\n"
            f"Opportunities: {self.opportunities_found}\n"
            f"Paper P&L: ${self.paper_pnl:.2f}",
            color=0xFF0000,
        )

        if self.weather:
            await self.weather.stop()
        if self.kalshi:
            await self.kalshi.stop()

        self._print_summary()

    def _print_banner(self):
        """Print startup banner."""
        print("\n" + "=" * 70)
        print("""
    ██╗    ██╗███████╗ █████╗ ████████╗██╗  ██╗███████╗██████╗
    ██║    ██║██╔════╝██╔══██╗╚══██╔══╝██║  ██║██╔════╝██╔══██╗
    ██║ █╗ ██║█████╗  ███████║   ██║   ███████║█████╗  ██████╔╝
    ██║███╗██║██╔══╝  ██╔══██║   ██║   ██╔══██║██╔══╝  ██╔══██╗
    ╚███╔███╔╝███████╗██║  ██║   ██║   ██║  ██║███████╗██║  ██║
     ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
               KALSHI WEATHER BOT v{version}
                  NYC High Temp Markets
                   Target: 75-85% Win Rate
        """.format(version=self.VERSION))
        print("=" * 70)

    def _print_summary(self):
        """Print session summary."""
        runtime = datetime.now(timezone.utc).timestamp() - self.start_time

        print("\n" + "=" * 70)
        print("                    SESSION SUMMARY")
        print("=" * 70)
        print(f"  Runtime:              {runtime/60:.1f} minutes")
        print(f"  Mode:                 {'LIVE' if self.live_mode else 'PAPER'}")
        print(f"  Opportunities found:  {self.opportunities_found}")
        print(f"  Trades executed:      {self.trades_executed}")

        if self.paper_trades:
            settled = [t for t in self.paper_trades if t.settled]
            print(f"\n  [PAPER TRADING RESULTS]")
            print(f"    Total trades:  {len(self.paper_trades)}")
            print(f"    Settled:       {len(settled)}")
            print(f"    Wins:          {self.paper_wins}")
            print(f"    Losses:        {self.paper_losses}")
            if self.paper_wins + self.paper_losses > 0:
                win_rate = self.paper_wins / (self.paper_wins + self.paper_losses)
                print(f"    Win Rate:      {win_rate:.1%}")
            print(f"    Paper P&L:     ${self.paper_pnl:.2f}")

        print("=" * 70)

    def _parse_ticker_date(self, ticker: str) -> Optional[str]:
        """
        Parse date from Kalshi ticker.

        Example: KXHIGHNY-26JAN12-B39.5 -> 2026-01-12
        """
        import re
        # Match pattern like 26JAN12
        match = re.search(r"(\d{2})([A-Z]{3})(\d{2})", ticker)
        if not match:
            return None

        year = int("20" + match.group(1))
        month_abbr = match.group(2)
        day = int(match.group(3))

        month = self.MONTHS.get(month_abbr)
        if not month:
            return None

        return f"{year}-{month:02d}-{day:02d}"

    def _date_to_ticker_format(self, date_str: str) -> str:
        """
        Convert date to ticker format.

        Example: 2026-01-12 -> 26JAN12
        """
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        month_abbrs = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                       "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        return f"{dt.year % 100:02d}{month_abbrs[dt.month-1]}{dt.day:02d}"

    def _parse_bracket_from_subtitle(self, subtitle: str) -> dict:
        """
        Parse bracket info from market subtitle.

        Examples:
            "39° to 40°" -> {"min": 39, "max": 40, "range": "39-40°F"}
            "47° or above" -> {"min": 47, "max": inf, "range": ">47°F"}
            "38° or below" -> {"min": -inf, "max": 38, "range": "<38°F"}
        """
        import re

        # Range: "39° to 40°"
        range_match = re.search(r"(\d+)°?\s*to\s*(\d+)°?", subtitle)
        if range_match:
            min_temp = int(range_match.group(1))
            max_temp = int(range_match.group(2))
            return {
                "min": min_temp,
                "max": max_temp + 1,  # Bracket is inclusive
                "range": f"{min_temp}-{max_temp}°F"
            }

        # Above: "47° or above"
        above_match = re.search(r"(\d+)°?\s*or\s*above", subtitle)
        if above_match:
            min_temp = int(above_match.group(1))
            return {
                "min": min_temp,
                "max": float("inf"),
                "range": f">{min_temp}°F"
            }

        # Below: "38° or below"
        below_match = re.search(r"(\d+)°?\s*or\s*below", subtitle)
        if below_match:
            max_temp = int(below_match.group(1))
            return {
                "min": float("-inf"),
                "max": max_temp + 1,  # Include the boundary
                "range": f"<{max_temp+1}°F"
            }

        return {"min": 0, "max": 0, "range": "Unknown"}

    async def scan_once(self):
        """Run a single scan for opportunities."""
        print("\n[SCAN] Checking weather forecasts and markets...")

        # Get weather forecasts
        gfs_forecasts, ecmwf_forecasts = await self.weather.get_nyc_forecasts()

        if not gfs_forecasts or not ecmwf_forecasts:
            print("[SCAN] Failed to get forecasts")
            return []

        # Index by date
        gfs_by_date = {f.date: f for f in gfs_forecasts}
        ecmwf_by_date = {f.date: f for f in ecmwf_forecasts}

        # Get Kalshi weather markets
        kalshi_markets = await self._get_weather_markets()

        if not kalshi_markets:
            print("[SCAN] No Kalshi weather markets found")
            return []

        # Group markets by date
        markets_by_date = {}
        for m in kalshi_markets:
            ticker = m.get("ticker", "")
            date = self._parse_ticker_date(ticker)
            if date:
                if date not in markets_by_date:
                    markets_by_date[date] = []
                markets_by_date[date].append(m)

        print(f"[SCAN] Market dates available: {sorted(markets_by_date.keys())}")

        opportunities = []

        # Check each date with both forecasts
        for date in sorted(set(gfs_by_date.keys()) & set(ecmwf_by_date.keys())):
            gfs = gfs_by_date[date]
            ecmwf = ecmwf_by_date[date]

            # Check model consensus
            consensus = self.weather.get_model_consensus(gfs, ecmwf, self.MAX_DIVERGENCE)
            if not consensus:
                continue

            # Find matching Kalshi markets for this date
            matching_markets = markets_by_date.get(date, [])

            if not matching_markets:
                print(f"[SCAN] No markets for {date}")
                continue

            print(f"[SCAN] {date}: Models agree at {consensus['avg_temp']:.1f}°F ({len(matching_markets)} brackets)")

            # Check for opportunities
            opp = await self._check_opportunity(consensus, matching_markets)
            if opp:
                if opp.is_tradeable:
                    opportunities.append(opp)
                    self.opportunities_found += 1
                    await self._handle_opportunity(opp)
                else:
                    print(f"[SCAN] {date}: Opportunity found but not tradeable (edge={opp.edge:.0%})")

        return opportunities

    async def _get_weather_markets(self) -> list:
        """Get Kalshi KXHIGHNY markets."""
        try:
            # Search for NYC high temp markets
            markets = await self.kalshi.get_markets(
                series_ticker="KXHIGHNY",
                limit=100
            )

            print(f"[KALSHI] Found {len(markets)} KXHIGHNY markets")
            return markets

        except Exception as e:
            print(f"[KALSHI] Error fetching markets: {e}")
            return []

    async def _check_opportunity(
        self,
        consensus: dict,
        markets: list
    ) -> Optional[WeatherOpportunity]:
        """Check if there's a trading opportunity."""
        model_temp = consensus["avg_temp"]

        # Parse each market's bracket and find the one our temp falls into
        target_market = None
        target_bracket_info = None
        all_markets_for_date = []

        for m in markets:
            subtitle = m.get("subtitle", "")
            bracket_info = self._parse_bracket_from_subtitle(subtitle)

            # Check if model temp falls in this bracket
            if bracket_info["min"] <= model_temp < bracket_info["max"]:
                target_market = m
                target_bracket_info = bracket_info

            all_markets_for_date.append({
                "market": m,
                "bracket": bracket_info
            })

        if not target_market:
            print(f"[SCAN] No bracket found for {model_temp:.1f}°F")
            return None

        # Get prices for target bracket
        yes_price = float(target_market.get("yes_bid", 0) or 0) / 100
        if yes_price == 0:
            yes_price = float(target_market.get("last_price", 0) or 0) / 100

        # Find market favorite (highest priced bracket)
        favorite = None
        favorite_bracket_info = None
        favorite_price = 0
        for item in all_markets_for_date:
            m = item["market"]
            price = float(m.get("yes_bid", 0) or 0) / 100
            if price == 0:
                price = float(m.get("last_price", 0) or 0) / 100
            if price > favorite_price:
                favorite = m
                favorite_bracket_info = item["bracket"]
                favorite_price = price

        # Get favorite bracket range
        favorite_bracket_range = favorite_bracket_info["range"] if favorite_bracket_info else "Unknown"

        # Calculate hours to settlement
        close_time = target_market.get("close_time", "")
        try:
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            hours_to_settlement = (close_dt - datetime.now(timezone.utc)).total_seconds() / 3600
        except:
            hours_to_settlement = 24  # Default

        # Calculate edge
        edge = consensus["confidence"] - yes_price

        # Create opportunity
        return WeatherOpportunity(
            date=consensus["date"],
            ticker=target_market.get("ticker", ""),
            gfs_temp=consensus["gfs_temp"],
            ecmwf_temp=consensus["ecmwf_temp"],
            model_avg=consensus["avg_temp"],
            model_bracket=target_bracket_info["range"],
            model_bracket_range=target_bracket_info["range"],
            model_confidence=consensus["confidence"],
            market_yes_price=yes_price,
            market_favorite=favorite_bracket_range,
            market_favorite_price=favorite_price,
            edge=edge,
            expected_value=edge * (1.0 - yes_price) if yes_price < 1 else 0,
            hours_to_settlement=hours_to_settlement,
        )

    async def _handle_opportunity(self, opp: WeatherOpportunity):
        """Handle a discovered opportunity."""
        print(f"\n{'='*70}")
        print(f"[OPPORTUNITY #{self.opportunities_found}] {opp.date}")
        print(f"{'='*70}")
        print(f"  Models:     GFS={opp.gfs_temp:.1f}°F  ECMWF={opp.ecmwf_temp:.1f}°F")
        print(f"  Average:    {opp.model_avg:.1f}°F → {opp.model_bracket_range}")
        print(f"  Confidence: {opp.model_confidence:.0%}")
        print(f"")
        print(f"  Market:     {opp.model_bracket_range} @ {opp.market_yes_price:.0%}")
        print(f"  Favorite:   {self.BRACKETS.get(opp.market_favorite, {}).get('range', '?')} @ {opp.market_favorite_price:.0%}")
        print(f"")
        print(f"  Edge:       {opp.edge:.0%}")
        print(f"  Potential:  {opp.potential_return:.1f}x return")
        print(f"  EV:         ${opp.expected_value:.2f} per contract")
        print(f"  Settlement: {opp.hours_to_settlement:.1f} hours")
        print(f"{'='*70}")

        # Send alert
        await self.alerts.send(
            f"Weather Opportunity: {opp.date}",
            f"Models: GFS={opp.gfs_temp:.1f}°F ECMWF={opp.ecmwf_temp:.1f}°F\n"
            f"Bracket: {opp.model_bracket_range} @ {opp.market_yes_price:.0%}\n"
            f"Edge: {opp.edge:.0%}\n"
            f"Potential: {opp.potential_return:.1f}x",
            color=0x00FF00,
        )

        # Execute or paper trade
        if self.live_mode:
            await self._execute_trade(opp)
        elif self.paper_mode:
            self._paper_trade(opp)

    async def _execute_trade(self, opp: WeatherOpportunity):
        """Execute a live trade on Kalshi."""
        print("\n[EXECUTE] Placing live trade...")

        # Calculate position size
        price_cents = int(opp.market_yes_price * 100)
        contracts = int(self.MAX_POSITION_SIZE / opp.market_yes_price)

        try:
            result = await self.kalshi.place_order(
                ticker=opp.ticker,
                side="yes",
                action="buy",
                count=contracts,
                price=price_cents,
                order_type="limit",
            )

            self.trades_executed += 1
            print(f"  [SUCCESS] Bought {contracts} YES @ {opp.market_yes_price:.0%}")
            print(f"  Order ID: {result.get('order', {}).get('order_id', 'N/A')}")

            # Log trade
            self._log_trade(opp, contracts, is_paper=False)

        except Exception as e:
            print(f"  [ERROR] Trade failed: {e}")

    def _paper_trade(self, opp: WeatherOpportunity):
        """Record a paper trade."""
        contracts = int(self.MAX_POSITION_SIZE / max(opp.market_yes_price, 0.01))

        trade = PaperTrade(
            opportunity=opp,
            contracts=contracts,
            entry_price=opp.market_yes_price,
            entry_ts=datetime.now(timezone.utc).timestamp(),
        )

        self.paper_trades.append(trade)
        self._log_trade(opp, contracts, is_paper=True)

        print(f"\n[PAPER] Recorded paper trade:")
        print(f"  {contracts} contracts @ {opp.market_yes_price:.0%}")
        print(f"  Max profit: ${contracts * (1 - opp.market_yes_price):.2f}")
        print(f"  Max loss: ${contracts * opp.market_yes_price:.2f}")

    def _log_trade(self, opp: WeatherOpportunity, contracts: int, is_paper: bool):
        """Log trade to file."""
        log_file = PAPER_TRADES_LOG if is_paper else Path("weather_live_trades.jsonl")

        with open(log_file, "a") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "date": opp.date,
                "ticker": opp.ticker,
                "gfs_temp": opp.gfs_temp,
                "ecmwf_temp": opp.ecmwf_temp,
                "model_bracket": opp.model_bracket,
                "market_price": opp.market_yes_price,
                "edge": opp.edge,
                "contracts": contracts,
                "is_paper": is_paper,
            }) + "\n")

    def _load_paper_trades(self):
        """Load existing paper trades from log."""
        if PAPER_TRADES_LOG.exists():
            # Count existing trades
            with open(PAPER_TRADES_LOG) as f:
                lines = f.readlines()
                print(f"[PAPER] Loaded {len(lines)} historical paper trades")

    def _save_paper_trades(self):
        """Save paper trade results."""
        pass  # Already logged in real-time

    async def run_loop(self, interval: int = 300):
        """Run continuous scanning loop."""
        while self.running:
            try:
                await self.scan_once()
                print(f"\n[WAIT] Next scan in {interval}s...")
                await asyncio.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] Scan failed: {e}")
                await asyncio.sleep(60)

        await self.stop()


async def main():
    parser = argparse.ArgumentParser(description="Kalshi Weather Trading Bot")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (default: paper)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single scan and exit",
    )
    args = parser.parse_args()

    bot = WeatherBot(live_mode=args.live)
    await bot.start()

    if args.once:
        await bot.scan_once()
        await bot.stop()
    else:
        await bot.run_loop()


if __name__ == "__main__":
    asyncio.run(main())
