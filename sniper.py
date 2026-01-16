#!/usr/bin/env python3
"""
NYC SNIPER - Strategy-Based Weather Trading Bot
Implements Wind Penalty + Midnight High strategies with human-in-the-loop execution.
"""

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv

from kalshi_client import KalshiClient
from alerts import AlertManager

load_dotenv()

ET = ZoneInfo("America/New_York")
TRADES_LOG = Path("sniper_trades.jsonl")


@dataclass
class HourlyForecast:
    """Hourly forecast data from NWS."""
    time: datetime
    temp_f: float
    wind_speed_mph: float
    wind_gust_mph: float
    short_forecast: str
    is_daytime: bool


@dataclass
class TradeTicket:
    """Trade recommendation with all analysis data."""
    nws_forecast_high: float
    physics_high: float
    wind_penalty: float
    wind_gust: float
    is_midnight_risk: bool
    midnight_temp: Optional[float]
    afternoon_temp: Optional[float]
    target_bracket_low: int
    target_bracket_high: int
    target_ticker: str
    current_price_cents: int
    implied_odds: float
    estimated_edge: float
    recommendation: str  # "BUY", "PASS", "HEDGE"
    confidence: int  # 1-10
    rationale: str


@dataclass
class ExitSignal:
    """Exit recommendation for position management."""
    ticker: str
    signal_type: str  # "TAKE_PROFIT", "BAIL_OUT", "HOLD"
    contracts_held: int
    avg_entry_cents: int
    current_bid_cents: int
    roi_percent: float
    target_bracket: tuple[int, int]  # (low, high)
    nws_forecast_high: float
    thesis_valid: bool
    sell_qty: int  # 0 for HOLD
    sell_price_cents: int
    rationale: str


class NWSClient:
    """NWS API client for observations and hourly forecasts."""
    OBS_URL = "https://api.weather.gov/stations/KNYC/observations/latest"
    HOURLY_URL = "https://api.weather.gov/gridpoints/OKX/33,37/forecast/hourly"

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": "NYC_Sniper/4.0", "Accept": "application/geo+json"},
            timeout=aiohttp.ClientTimeout(total=15, connect=5),
        )

    async def stop(self):
        if self.session:
            await self.session.close()

    async def get_current_temp(self) -> Optional[float]:
        """Get current temperature from KNYC station."""
        try:
            async with self.session.get(self.OBS_URL) as resp:
                if resp.status != 200:
                    return None
                props = (await resp.json()).get("properties", {})
                temp_c = props.get("temperature", {}).get("value")
                if temp_c is None:
                    return None
                return round((temp_c * 1.8) + 32, 1)
        except Exception as e:
            print(f"[ERR] Current temp fetch failed: {e}")
            return None

    async def get_hourly_forecast(self) -> list[HourlyForecast]:
        """Get hourly forecast including wind data."""
        try:
            async with self.session.get(self.HOURLY_URL) as resp:
                if resp.status != 200:
                    print(f"[ERR] Hourly forecast returned {resp.status}")
                    return []
                data = await resp.json()
                periods = data.get("properties", {}).get("periods", [])

                forecasts = []
                for p in periods[:48]:  # Next 48 hours
                    try:
                        time = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00"))
                        temp_f = float(p.get("temperature", 0))

                        # Parse wind speed (e.g., "10 mph" or "5 to 10 mph")
                        wind_str = p.get("windSpeed", "0 mph")
                        wind_match = re.search(r"(\d+)\s*(?:to\s*(\d+))?\s*mph", wind_str, re.I)
                        if wind_match:
                            wind_speed = float(wind_match.group(2) or wind_match.group(1))
                        else:
                            wind_speed = 0.0

                        # Estimate gusts (typically 1.5x sustained in strong wind)
                        wind_gust = wind_speed * 1.5 if wind_speed > 10 else wind_speed

                        forecasts.append(HourlyForecast(
                            time=time,
                            temp_f=temp_f,
                            wind_speed_mph=wind_speed,
                            wind_gust_mph=wind_gust,
                            short_forecast=p.get("shortForecast", ""),
                            is_daytime=p.get("isDaytime", False),
                        ))
                    except Exception:
                        continue
                return forecasts
        except Exception as e:
            print(f"[ERR] Hourly forecast fetch failed: {e}")
            return []


class NYCSniper:
    """Strategy-based weather trading bot with human-in-the-loop execution."""
    VERSION = "4.0.0"
    MAX_POSITION_PCT = 0.15  # 15% of NLV per trade

    def __init__(self, live_mode: bool = False):
        self.live_mode = live_mode
        self.nws: Optional[NWSClient] = None
        self.kalshi: Optional[KalshiClient] = None
        self.alerts: Optional[AlertManager] = None
        self.balance = 0.0

    async def start(self):
        print(f"\n{'='*60}")
        print(f"  NYC SNIPER v{self.VERSION}")
        print(f"  Strategy: Wind Penalty + Midnight High")
        print(f"{'='*60}")

        self.nws = NWSClient()
        await self.nws.start()

        self.kalshi = KalshiClient(
            api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
            private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
            demo_mode=False,
        )
        await self.kalshi.start()
        self.alerts = AlertManager()

        self.balance = await self.kalshi.get_balance()
        print(f"\n[INIT] Mode: {'LIVE' if self.live_mode else 'ANALYSIS ONLY'}")
        print(f"[INIT] Balance: ${self.balance:.2f}")
        print(f"[INIT] Max Position: ${self.balance * self.MAX_POSITION_PCT:.2f} (15% of NLV)")

    async def stop(self):
        if self.nws:
            await self.nws.stop()
        if self.kalshi:
            await self.kalshi.stop()

    def calculate_wind_penalty(self, wind_gust_mph: float) -> float:
        """
        Strategy B: Wind Mixing Penalty
        - Gusts > 15 mph: -1.0F penalty
        - Gusts > 25 mph: -2.0F penalty
        """
        if wind_gust_mph > 25:
            return 2.0
        elif wind_gust_mph > 15:
            return 1.0
        return 0.0

    def check_midnight_high(self, forecasts: list[HourlyForecast]) -> tuple[bool, Optional[float], Optional[float]]:
        """
        Strategy A: Midnight High Detection
        Returns (is_midnight_high, midnight_temp, afternoon_temp)
        """
        now = datetime.now(ET)
        tomorrow = now.date() + timedelta(days=1)

        midnight_temp = None
        afternoon_temp = None

        for f in forecasts:
            f_local = f.time.astimezone(ET)
            f_date = f_local.date()
            f_hour = f_local.hour

            # Find midnight (12:00-1:00 AM) temp for tomorrow
            if f_date == tomorrow and 0 <= f_hour <= 1:
                midnight_temp = f.temp_f

            # Find afternoon (2:00-4:00 PM) temp for tomorrow
            if f_date == tomorrow and 14 <= f_hour <= 16:
                afternoon_temp = f.temp_f

        # Midnight High: if midnight temp > afternoon temp, the high is locked at midnight
        is_midnight = False
        if midnight_temp is not None and afternoon_temp is not None:
            is_midnight = midnight_temp > afternoon_temp

        return is_midnight, midnight_temp, afternoon_temp

    def get_nws_forecast_high(self, forecasts: list[HourlyForecast]) -> tuple[float, float]:
        """Get NWS forecast high and max wind gust for tomorrow."""
        now = datetime.now(ET)
        tomorrow = now.date() + timedelta(days=1)

        max_temp = 0.0
        max_gust = 0.0

        for f in forecasts:
            f_local = f.time.astimezone(ET)
            if f_local.date() == tomorrow:
                if f.temp_f > max_temp:
                    max_temp = f.temp_f
                if f.wind_gust_mph > max_gust:
                    max_gust = f.wind_gust_mph

        return max_temp, max_gust

    def temp_to_bracket(self, temp_f: float) -> tuple[int, int]:
        """Convert temperature to bracket bounds (low, high)."""
        # NWS rounds: x.49 down, x.50 up
        rounded = int(Decimal(str(temp_f)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        # Brackets are typically 2-degree ranges
        low = (rounded // 2) * 2 - 1
        high = low + 2
        return low, high

    async def get_kalshi_markets(self) -> list[dict]:
        """Fetch today's and tomorrow's KXHIGHNY markets."""
        try:
            markets = await self.kalshi.get_markets(series_ticker="KXHIGHNY", status="open", limit=100)
            return markets
        except Exception as e:
            print(f"[ERR] Market fetch failed: {e}")
            return []

    def find_target_market(self, markets: list[dict], target_temp: float) -> Optional[dict]:
        """Find the market bracket containing the target temperature."""
        now = datetime.now(ET)
        tomorrow = now + timedelta(days=1)
        tomorrow_str = f"{tomorrow.year % 100:02d}{['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'][tomorrow.month-1]}{tomorrow.day:02d}"

        for m in markets:
            ticker = m.get("ticker", "")
            if tomorrow_str not in ticker:
                continue

            subtitle = m.get("subtitle", "").lower()

            # Parse bracket range from subtitle
            # e.g., "33 to 34" or "35 or above" or "32 or below"
            if "to" in subtitle:
                match = re.search(r"(\d+)\s*(?:°|degrees?)?\s*to\s*(\d+)", subtitle)
                if match:
                    low, high = int(match.group(1)), int(match.group(2))
                    if low <= target_temp <= high:
                        return m
            elif "above" in subtitle:
                match = re.search(r"(\d+)\s*(?:°|degrees?)?\s*or\s*above", subtitle)
                if match:
                    threshold = int(match.group(1))
                    if target_temp >= threshold:
                        return m
            elif "below" in subtitle:
                match = re.search(r"(\d+)\s*(?:°|degrees?)?\s*or\s*below", subtitle)
                if match:
                    threshold = int(match.group(1))
                    if target_temp < threshold:
                        return m

        return None

    def generate_trade_ticket(
        self,
        nws_high: float,
        wind_gust: float,
        is_midnight: bool,
        midnight_temp: Optional[float],
        afternoon_temp: Optional[float],
        market: Optional[dict],
    ) -> TradeTicket:
        """Generate a trade ticket with full analysis."""

        # Calculate physics adjustments
        wind_penalty = self.calculate_wind_penalty(wind_gust)
        physics_high = nws_high - wind_penalty

        # If midnight high scenario, use midnight temp as the physics high
        if is_midnight and midnight_temp:
            physics_high = midnight_temp

        # Determine target bracket
        bracket_low, bracket_high = self.temp_to_bracket(physics_high)

        # Get market data
        if market:
            ticker = market.get("ticker", "")
            yes_bid = market.get("yes_bid", 0)
            yes_ask = market.get("yes_ask", 0)
            price = yes_ask if yes_ask else yes_bid
            implied_odds = price / 100 if price else 0.5
        else:
            ticker = "NO_MARKET_FOUND"
            price = 0
            implied_odds = 0.5

        # Calculate edge (our confidence - market implied probability)
        # Base confidence: 70% for wind penalty, 80% for midnight high
        our_confidence = 0.80 if is_midnight else 0.70
        edge = our_confidence - implied_odds

        # Determine recommendation
        if edge > 0.20 and price > 0 and price < 80:
            recommendation = "BUY"
            confidence = 8 if edge > 0.30 else 7
        elif edge > 0.10:
            recommendation = "PASS"  # Edge exists but too thin
            confidence = 5
        else:
            recommendation = "PASS"
            confidence = 3

        # Build rationale
        rationale_parts = []
        if wind_penalty > 0:
            rationale_parts.append(f"Wind Penalty: -{wind_penalty}F (gusts {wind_gust:.0f}mph)")
        if is_midnight:
            rationale_parts.append(f"Midnight High: {midnight_temp}F > Afternoon {afternoon_temp}F")
        if not rationale_parts:
            rationale_parts.append("No significant weather anomaly detected")

        return TradeTicket(
            nws_forecast_high=nws_high,
            physics_high=physics_high,
            wind_penalty=wind_penalty,
            wind_gust=wind_gust,
            is_midnight_risk=is_midnight,
            midnight_temp=midnight_temp,
            afternoon_temp=afternoon_temp,
            target_bracket_low=bracket_low,
            target_bracket_high=bracket_high,
            target_ticker=ticker,
            current_price_cents=price,
            implied_odds=implied_odds,
            estimated_edge=edge,
            recommendation=recommendation,
            confidence=confidence,
            rationale=" | ".join(rationale_parts),
        )

    def print_trade_ticket(self, ticket: TradeTicket):
        """Print formatted trade ticket."""
        print("\n" + "="*50)
        print("         SNIPER ANALYSIS")
        print("="*50)
        print(f"* NWS Forecast High:  {ticket.nws_forecast_high:.0f}F")
        print(f"* Physics High:       {ticket.physics_high:.0f}F (Wind Penalty: -{ticket.wind_penalty:.0f}F)")
        print(f"* Max Wind Gust:      {ticket.wind_gust:.0f} mph")
        print(f"* Midnight Risk:      {'YES' if ticket.is_midnight_risk else 'No'}")
        if ticket.is_midnight_risk:
            print(f"  - Midnight Temp:    {ticket.midnight_temp:.0f}F")
            print(f"  - Afternoon Temp:   {ticket.afternoon_temp:.0f}F")
        print("-"*50)
        print(f"TARGET BRACKET:    {ticket.target_bracket_low}F to {ticket.target_bracket_high}F")
        print(f"TICKER:            {ticket.target_ticker}")
        print(f"CURRENT PRICE:     {ticket.current_price_cents}c (Implied: {ticket.implied_odds:.0%})")
        print(f"ESTIMATED EDGE:    {'+' if ticket.estimated_edge > 0 else ''}{ticket.estimated_edge:.0%}")
        print(f"CONFIDENCE:        {ticket.confidence}/10")
        print("-"*50)
        print(f"RATIONALE: {ticket.rationale}")
        print("-"*50)
        print(f">>> RECOMMENDATION: {ticket.recommendation} <<<")
        print("="*50)

    async def execute_trade(self, ticket: TradeTicket) -> bool:
        """Execute trade with human confirmation."""
        if ticket.recommendation != "BUY":
            print("\n[SKIP] No trade recommended.")
            return False

        if ticket.current_price_cents == 0:
            print("\n[SKIP] No valid market price.")
            return False

        # Calculate position size (15% of NLV)
        max_cost = self.balance * self.MAX_POSITION_PCT
        contracts = int(max_cost / (ticket.current_price_cents / 100))
        total_cost = contracts * ticket.current_price_cents / 100
        potential_profit = contracts * (100 - ticket.current_price_cents) / 100

        print(f"\n[TRADE SETUP]")
        print(f"  Contracts: {contracts}")
        print(f"  Entry:     {ticket.current_price_cents}c")
        print(f"  Cost:      ${total_cost:.2f}")
        print(f"  Max Profit: ${potential_profit:.2f}")

        if not self.live_mode:
            print("\n[ANALYSIS MODE] No trade executed. Use --live for real trades.")
            return False

        # Human-in-the-loop confirmation
        response = input(f"\nExecute trade? (y/n): ").strip().lower()

        if response != "y":
            print("[CANCELLED] Trade not executed.")
            return False

        # Execute the trade
        try:
            result = await self.kalshi.place_order(
                ticker=ticket.target_ticker,
                side="yes",
                action="buy",
                count=contracts,
                price=ticket.current_price_cents,
                order_type="limit"
            )
            order_id = result.get("order", {}).get("order_id", "N/A")
            print(f"\n[EXECUTED] Order ID: {order_id}")

            # Log the trade
            with open(TRADES_LOG, "a") as f:
                f.write(json.dumps({
                    "ts": datetime.now(ET).isoformat(),
                    "ticker": ticket.target_ticker,
                    "side": "yes",
                    "contracts": contracts,
                    "price": ticket.current_price_cents,
                    "nws_high": ticket.nws_forecast_high,
                    "physics_high": ticket.physics_high,
                    "wind_penalty": ticket.wind_penalty,
                    "midnight_risk": ticket.is_midnight_risk,
                    "edge": ticket.estimated_edge,
                    "order_id": order_id,
                }) + "\n")

            return True
        except Exception as e:
            print(f"\n[ERROR] Trade failed: {e}")
            return False

    # =========================================================================
    # PORTFOLIO MANAGER MODE (--manage)
    # =========================================================================

    def parse_bracket_from_ticker(self, ticker: str) -> tuple[int, int]:
        """Parse bracket range from ticker (e.g., B33.5 -> (33, 35))."""
        match = re.search(r"-([TB])(\d+(?:\.\d+)?)$", ticker)
        if not match:
            return (0, 0)

        prefix, value = match.group(1), float(match.group(2))

        if prefix == "T":
            # T39 means >= 39 (above) or < 39 (below)
            # Need to check market subtitle for actual meaning
            return (int(value), int(value) + 2)
        else:
            # B33.5 means between bracket, typically 33-35
            low = int(value)
            return (low, low + 2)

    async def get_avg_entry_from_fills(self, ticker: str) -> tuple[int, float]:
        """Calculate average entry price from fills for a ticker."""
        resp = await self.kalshi._req("GET", "/portfolio/fills?limit=200", auth=True)
        fills = resp.get("fills", [])

        total_cost = 0
        total_contracts = 0

        for f in fills:
            if f.get("ticker") != ticker:
                continue

            side = f.get("side")
            action = f.get("action")
            count = f.get("count", 0)
            price = f.get("yes_price") or f.get("no_price") or 0

            # Calculate effective YES cost
            if side == "yes" and action == "buy":
                total_contracts += count
                total_cost += count * price
            elif side == "no" and action == "sell":
                # Sell NO = equivalent to Buy YES at (100 - no_price)
                total_contracts += count
                total_cost += count * (100 - price)

        avg_entry = total_cost / total_contracts if total_contracts > 0 else 0
        return total_contracts, avg_entry

    async def evaluate_position_from_api(self, position: dict, nws_high: float) -> Optional[ExitSignal]:
        """Evaluate a position from the positions API endpoint."""
        ticker = position.get("ticker", "")
        contracts = abs(position.get("position", 0))  # Can be negative for short

        if contracts <= 0:
            return None

        # Calculate avg entry from total_traded / position size isn't accurate
        # Use fills to calculate true average entry price
        _, avg_entry = await self.get_avg_entry_from_fills(ticker)

        # Get current bid from orderbook
        orderbook = await self.kalshi.get_orderbook(ticker)
        yes_bids = orderbook.get("yes", [])
        current_bid = yes_bids[0][0] if yes_bids else 0

        if current_bid == 0:
            print(f"  [WARN] No bids for {ticker}")
            return None

        # Calculate ROI
        roi = ((current_bid - avg_entry) / avg_entry * 100) if avg_entry > 0 else 0

        # Parse target bracket from ticker
        bracket = self.parse_bracket_from_ticker(ticker)

        # Check thesis validity: is forecast still in our bracket?
        thesis_valid = bracket[0] <= nws_high <= bracket[1]

        # Determine signal
        if roi >= 100:
            signal_type = "TAKE_PROFIT"
            sell_qty = contracts // 2
            rationale = f"ROI {roi:.0f}% >= 100%. Sell half to lock in risk-free profit."
        elif not thesis_valid:
            signal_type = "BAIL_OUT"
            sell_qty = contracts
            rationale = f"NWS forecast {nws_high:.0f}F OUTSIDE bracket {bracket[0]}-{bracket[1]}F. Thesis INVALID."
        else:
            signal_type = "HOLD"
            sell_qty = 0
            rationale = f"Thesis valid ({bracket[0]}-{bracket[1]}F). ROI {roi:.0f}%. Continue holding."

        return ExitSignal(
            ticker=ticker,
            signal_type=signal_type,
            contracts_held=contracts,
            avg_entry_cents=int(avg_entry),
            current_bid_cents=current_bid,
            roi_percent=roi,
            target_bracket=bracket,
            nws_forecast_high=nws_high,
            thesis_valid=thesis_valid,
            sell_qty=sell_qty,
            sell_price_cents=current_bid,
            rationale=rationale,
        )

    async def evaluate_position(self, ticker: str, nws_high: float) -> Optional[ExitSignal]:
        """Evaluate a single position and generate exit signal."""

        # Get position data from fills (more accurate than positions endpoint)
        contracts, avg_entry = await self.get_avg_entry_from_fills(ticker)

        if contracts <= 0:
            return None

        # Get current bid from orderbook
        # Format: {"yes": [[price, qty], ...], "no": [[price, qty], ...]}
        orderbook = await self.kalshi.get_orderbook(ticker)
        yes_bids = orderbook.get("yes", [])
        # First element is [price, quantity] - price is in cents
        current_bid = yes_bids[0][0] if yes_bids else 0

        if current_bid == 0:
            print(f"  [WARN] No bids for {ticker}")
            return None

        # Calculate ROI
        roi = ((current_bid - avg_entry) / avg_entry * 100) if avg_entry > 0 else 0

        # Parse target bracket from ticker
        bracket = self.parse_bracket_from_ticker(ticker)

        # Check thesis validity: is forecast still in our bracket?
        thesis_valid = bracket[0] <= nws_high <= bracket[1]

        # Determine signal
        if roi >= 100:
            signal_type = "TAKE_PROFIT"
            sell_qty = contracts // 2
            rationale = f"ROI {roi:.0f}% >= 100%. Sell half to lock in risk-free profit."
        elif not thesis_valid:
            signal_type = "BAIL_OUT"
            sell_qty = contracts
            rationale = f"NWS forecast {nws_high:.0f}F OUTSIDE bracket {bracket[0]}-{bracket[1]}F. Thesis INVALID."
        else:
            signal_type = "HOLD"
            sell_qty = 0
            rationale = f"Thesis valid ({bracket[0]}-{bracket[1]}F). ROI {roi:.0f}%. Continue holding."

        return ExitSignal(
            ticker=ticker,
            signal_type=signal_type,
            contracts_held=contracts,
            avg_entry_cents=int(avg_entry),
            current_bid_cents=current_bid,
            roi_percent=roi,
            target_bracket=bracket,
            nws_forecast_high=nws_high,
            thesis_valid=thesis_valid,
            sell_qty=sell_qty,
            sell_price_cents=current_bid,
            rationale=rationale,
        )

    def print_exit_signal(self, signal: ExitSignal, position_num: int):
        """Print formatted exit signal."""
        print(f"\n[POSITION {position_num}] {signal.ticker}")
        print("-" * 50)
        print(f"* Contracts:     {signal.contracts_held}")
        print(f"* Avg Entry:     {signal.avg_entry_cents}c")
        print(f"* Current Bid:   {signal.current_bid_cents}c")
        print(f"* ROI:           {'+' if signal.roi_percent >= 0 else ''}{signal.roi_percent:.0f}%")
        print(f"* Target:        {signal.target_bracket[0]}-{signal.target_bracket[1]}F")
        print(f"* NWS Forecast:  {signal.nws_forecast_high:.0f}F")
        print(f"* Thesis:        {'VALID' if signal.thesis_valid else 'INVALID'}")
        print("-" * 50)

        # Color-code signal type
        if signal.signal_type == "TAKE_PROFIT":
            print(f">>> SIGNAL: TAKE PROFIT <<<")
        elif signal.signal_type == "BAIL_OUT":
            print(f">>> SIGNAL: BAIL OUT <<<")
        else:
            print(f">>> SIGNAL: HOLD <<<")

        print(f">>> Rationale: {signal.rationale}")

        if signal.sell_qty > 0:
            proceeds = signal.sell_qty * signal.sell_price_cents / 100
            print(f">>> SELL {signal.sell_qty} @ {signal.sell_price_cents}c = ${proceeds:.2f}")

        print("=" * 50)

    async def execute_exit(self, signal: ExitSignal) -> bool:
        """Execute exit with human confirmation."""
        if signal.sell_qty == 0:
            return False

        proceeds = signal.sell_qty * signal.sell_price_cents / 100

        print(f"\n[EXIT ORDER]")
        print(f"  Ticker:    {signal.ticker}")
        print(f"  Action:    SELL {signal.sell_qty} YES")
        print(f"  Price:     {signal.sell_price_cents}c (LIMIT at bid)")
        print(f"  Proceeds:  ${proceeds:.2f}")

        if not self.live_mode:
            print("\n[ANALYSIS MODE] No trade executed. Use --live for real trades.")
            return False

        # Human-in-the-loop confirmation
        response = input(f"\nExecute sell? (y/n): ").strip().lower()

        if response != "y":
            print("[CANCELLED] Exit not executed.")
            return False

        # Execute the sell order
        try:
            result = await self.kalshi.place_order(
                ticker=signal.ticker,
                side="yes",
                action="sell",
                count=signal.sell_qty,
                price=signal.sell_price_cents,
                order_type="limit"
            )
            order_id = result.get("order", {}).get("order_id", "N/A")
            print(f"\n[EXECUTED] Sell Order ID: {order_id}")

            # Log the exit
            with open(TRADES_LOG, "a") as f:
                f.write(json.dumps({
                    "ts": datetime.now(ET).isoformat(),
                    "ticker": signal.ticker,
                    "side": "yes",
                    "action": "sell",
                    "contracts": signal.sell_qty,
                    "price": signal.sell_price_cents,
                    "signal_type": signal.signal_type,
                    "roi_percent": signal.roi_percent,
                    "order_id": order_id,
                }) + "\n")

            return True
        except Exception as e:
            print(f"\n[ERROR] Exit failed: {e}")
            return False

    async def manage_positions(self):
        """Portfolio manager mode - check positions and generate exit signals."""
        await self.start()

        try:
            print(f"\n{'='*60}")
            print(f"  NYC SNIPER - PORTFOLIO MANAGER")
            print(f"{'='*60}")

            # 1. Get current NWS forecast
            print("\n[1/3] Fetching NWS forecast...")
            forecasts = await self.nws.get_hourly_forecast()

            # Get forecast for TODAY (not tomorrow) since we're managing existing positions
            now = datetime.now(ET)
            today = now.date()

            max_temp_today = 0.0
            for f in forecasts:
                f_local = f.time.astimezone(ET)
                if f_local.date() == today and f.temp_f > max_temp_today:
                    max_temp_today = f.temp_f

            # Also get current observation
            current_temp = await self.nws.get_current_temp()
            if current_temp and current_temp > max_temp_today:
                max_temp_today = current_temp

            print(f"  NWS Forecast High (Today): {max_temp_today:.0f}F")
            if current_temp:
                print(f"  Current Temp: {current_temp:.0f}F")

            # 2. Get positions from positions endpoint (authoritative source)
            print("\n[2/3] Fetching positions...")
            positions = await self.kalshi.get_positions()

            # Filter for active KXHIGHNY positions with non-zero position
            active_positions = [
                p for p in positions
                if "KXHIGHNY" in p.get("ticker", "") and p.get("position", 0) != 0
            ]

            if not active_positions:
                print("\n[INFO] No active weather positions.")
                return

            print(f"  Found {len(active_positions)} active position(s)")

            # 3. Evaluate each position
            print("\n[3/3] Generating exit signals...")

            position_num = 0
            for pos in active_positions:
                ticker = pos.get("ticker", "")
                position_num += 1
                signal = await self.evaluate_position_from_api(pos, max_temp_today)

                if signal:
                    self.print_exit_signal(signal, position_num)

                    if signal.signal_type != "HOLD":
                        await self.execute_exit(signal)

            print(f"\n{'='*60}")
            print(f"  Portfolio review complete.")
            print(f"{'='*60}")

        finally:
            await self.stop()

    async def run(self):
        """Main analysis and trading workflow."""
        await self.start()

        try:
            print("\n[1/4] Fetching NWS hourly forecast...")
            forecasts = await self.nws.get_hourly_forecast()
            if not forecasts:
                print("[ERR] No forecast data available.")
                return

            print("[2/4] Analyzing weather patterns...")
            nws_high, max_gust = self.get_nws_forecast_high(forecasts)
            is_midnight, midnight_temp, afternoon_temp = self.check_midnight_high(forecasts)

            print(f"  NWS Forecast High: {nws_high:.0f}F")
            print(f"  Max Wind Gust:     {max_gust:.0f} mph")
            print(f"  Midnight High:     {'YES' if is_midnight else 'No'}")

            print("[3/4] Fetching Kalshi markets...")
            markets = await self.get_kalshi_markets()

            # Calculate physics high to find target market
            wind_penalty = self.calculate_wind_penalty(max_gust)
            physics_high = nws_high - wind_penalty
            if is_midnight and midnight_temp:
                physics_high = midnight_temp

            target_market = self.find_target_market(markets, physics_high)

            print("[4/4] Generating trade ticket...")
            ticket = self.generate_trade_ticket(
                nws_high, max_gust, is_midnight, midnight_temp, afternoon_temp, target_market
            )

            self.print_trade_ticket(ticket)

            await self.execute_trade(ticket)

        finally:
            await self.stop()


async def main():
    parser = argparse.ArgumentParser(description="NYC Sniper - Weather Trading Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (requires confirmation)")
    parser.add_argument("--manage", action="store_true", help="Portfolio manager mode - check positions for exit signals")
    args = parser.parse_args()

    bot = NYCSniper(live_mode=args.live)

    if args.manage:
        await bot.manage_positions()
    else:
        await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
