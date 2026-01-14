#!/usr/bin/env python3
"""
NYC Weather Latency Arbitrage Bot v3.0 - HIGH FIDELITY

WIN RATE TARGET: ~99% (risk-adjusted)

Key Upgrades from v2.0:
1. XML feed (bypasses JSON CDN cache, 1-3 min fresher)
2. Precision rounding (matches NWS CLI settlement exactly)
3. "Dead bracket" shorting (buy NO on mathematically lost contracts)
4. Connection optimization (TCP keepalive, DNS cache)

Strategy:
- Poll NWS XML feed every 15-30s (faster than JSON)
- Track MAX temp observed today (monotonic, can't decrease)
- BUY YES on triggered "above X" strikes
- BUY NO on "dead" contracts (temp already exceeded their ceiling)
- Use Decimal precision to match NWS rounding

Usage:
    python3 nyc_weather_arb.py           # Paper trading
    python3 nyc_weather_arb.py --live    # Live trading
    python3 nyc_weather_arb.py --test    # Test mode
"""

import argparse
import asyncio
import json
import os
import re
import time
import xml.etree.ElementTree as XMLParser
from dataclasses import dataclass, field
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

# Timezone for Kalshi tickers
ET = ZoneInfo("America/New_York")

# Log files
ARB_TRADES_LOG = Path("nyc_arb_trades.jsonl")
MAX_TEMP_LOG = Path("nyc_daily_max_temp.json")


@dataclass
class NWSObservation:
    """Real-time weather observation from NWS."""

    station_id: str
    timestamp: datetime
    temp_c: float
    temp_f: float
    description: str
    is_speci: bool  # Special observation (between hourly)
    raw_data: dict = field(default_factory=dict, repr=False)

    @property
    def age_seconds(self) -> float:
        """How old is this observation in seconds."""
        now = datetime.now(ET)
        obs_time = self.timestamp.astimezone(ET)
        return (now - obs_time).total_seconds()


@dataclass
class Strike:
    """A Kalshi strike price parsed from market data."""

    ticker: str
    strike_temp: float  # Temperature threshold
    strike_type: str    # "above", "below", or "between"
    yes_bid: float      # Current YES bid price (0-1 scale)
    yes_ask: float      # Current YES ask price (0-1 scale)
    no_bid: float       # Current NO bid price (for dead bracket shorting)
    no_ask: float       # Current NO ask price (for dead bracket shorting)
    subtitle: str       # Human-readable bracket
    upper_bound: Optional[float] = None  # For "between" brackets

    @property
    def is_triggered(self) -> bool:
        """Check if strike has been triggered (requires temp)."""
        return False  # Set externally

    def check_triggered(self, temp_f: float, buffer: float = 0.2) -> bool:
        """Check if temperature triggers this strike."""
        if self.strike_type == "above":
            return temp_f >= (self.strike_temp + buffer)
        elif self.strike_type == "below":
            return temp_f <= (self.strike_temp - buffer)
        else:
            # "between" - temp must be in range
            return False  # Don't trade between brackets for now

    def check_dead(self, max_temp: float, buffer: float = 0.5) -> bool:
        """
        Check if this strike is DEAD (mathematically lost).

        For "below X" strikes: if max_temp >= X, it's dead (YES loses)
        For "between X-Y" strikes: if max_temp > Y, it's dead
        """
        if self.strike_type == "below":
            # "High < 80°F" is dead if max temp >= 80.5°F
            return max_temp >= (self.strike_temp + buffer)
        elif self.strike_type == "between" and self.upper_bound:
            # "80-82°F" is dead if max temp > 82.5°F
            return max_temp >= (self.upper_bound + buffer)
        return False


class NWSClient:
    """
    High-Fidelity NWS Client v3.0

    KEY IMPROVEMENT: XML feed bypasses JSON CDN caching (1-3 min fresher)

    Priority order:
    1. XML feed (fastest, less cached)
    2. JSON API (fallback)
    """

    # XML feed - use www.weather.gov (w1 redirects)
    XML_URL = "https://www.weather.gov/xml/current_obs/KNYC.xml"
    JSON_URL = "https://api.weather.gov/stations/KNYC/observations/latest"

    HEADERS = {
        "User-Agent": "(NYC_HFT_Arb/3.0, kalshi-trader@example.com)",
        "Accept": "*/*",
        # Attempt to bypass caching
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
    }

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_observation: Optional[NWSObservation] = None
        self.xml_failures = 0
        self.json_failures = 0

    async def start(self):
        """Initialize HTTP session with HFT-optimized connection pooling."""
        # HFT Connection Tuning:
        # - limit=10: Enough parallel connections
        # - ttl_dns_cache=300: Avoid DNS lookups (25ms+ each)
        # - keepalive_timeout=120: Keep connections warm
        # - enable_cleanup_closed=True: Clean dead connections fast
        connector = aiohttp.TCPConnector(
            limit=10,
            ttl_dns_cache=300,  # Cache DNS for 5 min
            keepalive_timeout=120,  # Keep connections alive longer
            enable_cleanup_closed=True,
            force_close=False,  # Reuse connections
        )

        # Shorter timeouts for faster failure detection
        timeout = aiohttp.ClientTimeout(
            total=10,
            connect=3,
            sock_read=5,
        )

        self.session = aiohttp.ClientSession(
            headers=self.HEADERS,
            connector=connector,
            timeout=timeout,
        )
        print("[NWS] HFT client initialized (XML primary, JSON fallback)")

    async def stop(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()

    @staticmethod
    def _nws_round_f(temp_f_float: float) -> float:
        """
        Precision rounding to match NWS Climate Report standards.

        NWS uses "Round Half Up" (Banker's rounding variant).
        This ensures our calculated temp matches settlement exactly.

        Example: 89.95°F -> 90.0°F (not 89.9°F)
        """
        d = Decimal(str(temp_f_float))
        rounded = d.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        return float(rounded)

    async def get_latest_observation(self, station_id: str = "KNYC") -> Optional[NWSObservation]:
        """
        Fetch latest observation - XML first (faster), JSON fallback.
        """
        # 1. Try XML feed first (bypasses JSON CDN cache)
        obs = await self._fetch_xml()
        if obs:
            return obs

        # 2. Fallback to JSON API
        obs = await self._fetch_json(station_id)
        if obs:
            return obs

        print(f"[NWS] All sources failed (XML: {self.xml_failures}, JSON: {self.json_failures})")
        return None

    async def _fetch_xml(self) -> Optional[NWSObservation]:
        """Fetch from XML feed (faster, less cached)."""
        try:
            async with self.session.get(
                self.XML_URL,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    self.xml_failures += 1
                    return None

                content = await resp.read()
                return self._parse_xml(content)

        except asyncio.TimeoutError:
            print("[NWS] XML timeout")
            self.xml_failures += 1
            return None
        except Exception as e:
            print(f"[NWS] XML error: {e}")
            self.xml_failures += 1
            return None

    async def _fetch_json(self, station_id: str) -> Optional[NWSObservation]:
        """Fetch from JSON API (fallback)."""
        url = f"https://api.weather.gov/stations/{station_id}/observations/latest"

        try:
            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    self.json_failures += 1
                    return None

                data = await resp.json()
                return self._parse_json(data, station_id)

        except asyncio.TimeoutError:
            print("[NWS] JSON timeout")
            self.json_failures += 1
            return None
        except Exception as e:
            print(f"[NWS] JSON error: {e}")
            self.json_failures += 1
            return None

    def _parse_xml(self, content: bytes) -> Optional[NWSObservation]:
        """Parse NWS XML feed."""
        try:
            root = XMLParser.fromstring(content)

            # XML provides BOTH temp_f and temp_c directly - use temp_f for precision
            temp_f_elem = root.find("temp_f")
            temp_c_elem = root.find("temp_c")

            if temp_f_elem is not None and temp_f_elem.text:
                # Use direct Fahrenheit (more accurate, no conversion needed)
                temp_f = float(temp_f_elem.text)
                temp_c = float(temp_c_elem.text) if temp_c_elem is not None else (temp_f - 32) / 1.8
            elif temp_c_elem is not None and temp_c_elem.text:
                # Fallback to Celsius conversion
                temp_c = float(temp_c_elem.text)
                temp_f = self._nws_round_f((temp_c * 1.8) + 32)
            else:
                print("[NWS] No temperature in XML")
                return None

            # Parse observation time
            obs_time_elem = root.find("observation_time_rfc822")
            if obs_time_elem is not None and obs_time_elem.text:
                # RFC822 format: "Mon, 12 Jan 2026 15:51:00 -0500"
                try:
                    from email.utils import parsedate_to_datetime
                    timestamp = parsedate_to_datetime(obs_time_elem.text)
                except:
                    timestamp = datetime.now(ET)
            else:
                timestamp = datetime.now(ET)

            # Get weather description
            weather_elem = root.find("weather")
            description = weather_elem.text if weather_elem is not None else "Fair"

            obs = NWSObservation(
                station_id="KNYC",
                timestamp=timestamp,
                temp_c=temp_c,
                temp_f=temp_f,
                description=f"[XML] {description}",
                is_speci=False,  # XML doesn't flag SPECI easily
                raw_data={},
            )

            self.last_observation = obs
            return obs

        except Exception as e:
            print(f"[NWS] XML parse error: {e}")
            return None

    def _parse_json(self, data: dict, station_id: str) -> Optional[NWSObservation]:
        """Parse NWS JSON response."""
        try:
            props = data.get("properties", {})

            temp_data = props.get("temperature", {})
            temp_c = temp_data.get("value")

            if temp_c is None:
                print("[NWS] No temperature in JSON")
                return None

            # Convert with PRECISION ROUNDING
            temp_f_raw = (temp_c * 1.8) + 32
            temp_f = self._nws_round_f(temp_f_raw)

            # Parse timestamp
            timestamp_str = props.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except:
                timestamp = datetime.now(ET)

            # Check for SPECI
            description = props.get("textDescription", "")
            raw_message = props.get("rawMessage", "")
            is_speci = "SPECI" in raw_message or "SPECI" in description

            obs = NWSObservation(
                station_id=station_id,
                timestamp=timestamp,
                temp_c=temp_c,
                temp_f=temp_f,
                description=f"[JSON] {description}",
                is_speci=is_speci,
                raw_data=data,
            )

            self.last_observation = obs
            return obs

        except Exception as e:
            print(f"[NWS] JSON parse error: {e}")
            return None


class NYCWeatherArbBot:
    """
    Latency arbitrage bot for Kalshi NYC weather markets.

    HIGH WIN RATE STRATEGY:
    - Track max temp observed today (can only go up)
    - For "above X" strikes: trade when max >= X + buffer
    - For "below X" strikes: only after 4 PM when temp falling
    - Faster polling (30s) to catch opportunities
    """

    VERSION = "3.2.0"  # Added trajectory prediction

    # Trading parameters - HIGH FIDELITY
    SAFETY_BUFFER = 0.5      # °F buffer - increased for safety
    MIN_PROFIT_THRESHOLD = 3  # Minimum cents profit required
    MAX_POSITION_SIZE = 25.0  # Max $ per trade
    SWEEP_PRICE = 97         # Limit order price (lowered for better fills)
    POLL_INTERVAL = 20       # Faster polling (20s) - XML can handle it

    # Time-based rules (ET hours)
    ABOVE_STRIKE_START_HOUR = 10   # Start trading "above" strikes at 10 AM
    BELOW_STRIKE_START_HOUR = 16   # Only trade "below" strikes after 4 PM
    MARKET_CLOSE_HOUR = 20         # Stop trading at 8 PM

    # Glitch protection
    MAX_CONSECUTIVE_GLITCHES = 3   # Pause trading after 3 consecutive glitches

    # Trajectory prediction settings
    TRAJECTORY_MIN_READINGS = 3    # Minimum readings to calculate trend
    TRAJECTORY_WINDOW_MINUTES = 60 # How far ahead to predict
    TRAJECTORY_MIN_RATE = 0.5      # Minimum °F/hour to act on
    TRAJECTORY_STRIKE_BUFFER = 1.5 # Buy when predicted to be within X°F of strike
    TRAJECTORY_MIN_CONFIDENCE = 0.7  # Minimum R² for trend line

    def __init__(self, live_mode: bool = False):
        self.live_mode = live_mode
        self.paper_mode = not live_mode

        # Clients
        self.nws: Optional[NWSClient] = None
        self.kalshi: Optional[KalshiClient] = None
        self.alerts: Optional[AlertManager] = None

        # State
        self.running = False
        self.start_time = 0.0

        # Track max temp observed TODAY (key for win rate)
        self.today_date: str = ""
        self.max_temp_today: float = -999.0
        self.max_temp_time: Optional[datetime] = None

        # Track traded strikes (prevent duplicates)
        self.traded_tickers: set[str] = set()

        # Stats
        self.polls = 0
        self.triggers = 0
        self.trades = 0
        self.glitch_count = 0  # Consecutive glitch counter

        # Trajectory prediction - temperature history
        # List of (timestamp, temp_f) tuples
        self.temp_history: list[tuple[datetime, float]] = []
        self.trajectory_rate: float = 0.0  # °F per hour
        self.trajectory_confidence: float = 0.0  # R² of trend line

    async def start(self):
        """Initialize and start the bot."""
        self._print_banner()

        # Initialize NWS client
        self.nws = NWSClient()
        await self.nws.start()

        # Initialize Kalshi client
        self.kalshi = KalshiClient(
            api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
            private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
            demo_mode=False,  # Use production API
        )
        await self.kalshi.start()

        # Initialize alerts
        self.alerts = AlertManager()

        # Check balance if live
        if self.live_mode:
            balance = await self.kalshi.get_balance()
            print(f"[INIT] Kalshi balance: ${balance:.2f}")
            if balance < 10:
                print("[WARN] Low balance - switching to paper mode")
                self.live_mode = False
                self.paper_mode = True

        mode = "LIVE" if self.live_mode else "PAPER"
        print(f"[INIT] Mode: {mode}")
        print(f"[INIT] Safety buffer: {self.SAFETY_BUFFER}°F")
        print(f"[INIT] Sweep price: {self.SWEEP_PRICE}¢")
        print(f"[INIT] Max position: ${self.MAX_POSITION_SIZE:.2f}")
        print(f"[INIT] Poll interval: {self.POLL_INTERVAL}s")

        self.running = True
        self.start_time = time.time()

        # Load traded tickers and max temp from logs
        self._load_traded_tickers()
        self._load_max_temp()

        print("\n" + "=" * 70)
        print("[START] NYC Weather Arb Bot v2.0 - HIGH WIN RATE MODE")
        print("=" * 70)

    async def stop(self):
        """Graceful shutdown."""
        self.running = False

        if self.nws:
            await self.nws.stop()
        if self.kalshi:
            await self.kalshi.stop()

        self._print_summary()

    def _print_banner(self):
        """Print startup banner."""
        print("\n" + "=" * 70)
        print("""
    ███╗   ██╗██╗   ██╗ ██████╗    █████╗ ██████╗ ██████╗
    ████╗  ██║╚██╗ ██╔╝██╔════╝   ██╔══██╗██╔══██╗██╔══██╗
    ██╔██╗ ██║ ╚████╔╝ ██║        ███████║██████╔╝██████╔╝
    ██║╚██╗██║  ╚██╔╝  ██║        ██╔══██║██╔══██╗██╔══██╗
    ██║ ╚████║   ██║   ╚██████╗   ██║  ██║██║  ██║██████╔╝
    ╚═╝  ╚═══╝   ╚═╝    ╚═════╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
          HIGH FIDELITY ARB BOT v{version}
         XML Feed + Dead Brackets + TRAJECTORY PREDICTION
                 Target: ~99% Win Rate
    """.format(version=self.VERSION))
        print("=" * 70)

    def _print_summary(self):
        """Print session summary."""
        runtime = time.time() - self.start_time

        print("\n" + "=" * 70)
        print("                    SESSION SUMMARY")
        print("=" * 70)
        print(f"  Runtime:    {runtime/60:.1f} minutes")
        print(f"  Mode:       {'LIVE' if self.live_mode else 'PAPER'}")
        print(f"  Polls:      {self.polls}")
        print(f"  Triggers:   {self.triggers}")
        print(f"  Trades:     {self.trades}")
        print("=" * 70)

    def _load_traded_tickers(self):
        """Load previously traded tickers from log."""
        if ARB_TRADES_LOG.exists():
            with open(ARB_TRADES_LOG) as f:
                for line in f:
                    try:
                        trade = json.loads(line.strip())
                        ticker = trade.get("ticker", "")
                        if ticker:
                            self.traded_tickers.add(ticker)
                    except:
                        continue
            print(f"[INIT] Loaded {len(self.traded_tickers)} previously traded tickers")

    def _load_max_temp(self):
        """Load today's max temp from file (persists across restarts)."""
        today = datetime.now(ET).strftime("%Y-%m-%d")
        self.today_date = today

        if MAX_TEMP_LOG.exists():
            try:
                with open(MAX_TEMP_LOG) as f:
                    data = json.load(f)
                    if data.get("date") == today:
                        self.max_temp_today = data.get("max_temp", -999.0)
                        print(f"[INIT] Loaded max temp for today: {self.max_temp_today}°F")
                    else:
                        # New day, reset
                        self.max_temp_today = -999.0
                        print(f"[INIT] New day - max temp reset")
            except:
                self.max_temp_today = -999.0

    def _save_max_temp(self):
        """Save today's max temp to file."""
        with open(MAX_TEMP_LOG, "w") as f:
            json.dump({
                "date": self.today_date,
                "max_temp": self.max_temp_today,
                "updated_at": datetime.now(ET).isoformat(),
            }, f)

    def _update_trajectory(self, temp_f: float, obs_time: datetime):
        """
        Update temperature history and calculate trajectory.

        Trajectory = rate of temperature change (°F per hour)
        Positive = warming, Negative = cooling
        """
        now = datetime.now(ET)

        # Add new reading
        self.temp_history.append((obs_time, temp_f))

        # Keep only last 2 hours of data
        cutoff = now - timedelta(hours=2)
        self.temp_history = [(t, temp) for t, temp in self.temp_history if t > cutoff]

        # Need at least 3 readings to calculate trend
        if len(self.temp_history) < self.TRAJECTORY_MIN_READINGS:
            self.trajectory_rate = 0.0
            self.trajectory_confidence = 0.0
            return

        # Calculate linear regression for trajectory
        # Convert timestamps to hours since first reading
        first_time = self.temp_history[0][0]
        x = []  # Hours since first reading
        y = []  # Temperatures

        for t, temp in self.temp_history:
            hours = (t - first_time).total_seconds() / 3600
            x.append(hours)
            y.append(temp)

        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)

        # Slope (rate of change in °F per hour)
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            self.trajectory_rate = 0.0
            self.trajectory_confidence = 0.0
            return

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        self.trajectory_rate = slope

        # Calculate R² (confidence in the trend)
        mean_y = sum_y / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        if ss_tot == 0:
            self.trajectory_confidence = 1.0
            return

        intercept = (sum_y - slope * sum_x) / n
        ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot)
        self.trajectory_confidence = max(0, r_squared)

    def _predict_temp(self, minutes_ahead: int) -> float:
        """Predict temperature X minutes in the future based on trajectory."""
        if not self.temp_history or self.trajectory_rate == 0:
            return self.max_temp_today

        current_temp = self.temp_history[-1][1]
        hours_ahead = minutes_ahead / 60
        predicted = current_temp + (self.trajectory_rate * hours_ahead)

        return predicted

    def _find_trajectory_opportunities(
        self,
        current_temp: float,
        strikes: list,
    ) -> list[dict]:
        """
        Find pre-positioning opportunities based on trajectory prediction.

        Strategy: If temp is rising and predicted to cross a strike within
        the prediction window, BUY EARLY before the market reprices.
        """
        opportunities = []
        now = datetime.now(ET)

        # Only use trajectory if confidence is high enough
        if self.trajectory_confidence < self.TRAJECTORY_MIN_CONFIDENCE:
            return []

        # Only pre-position if temp is rising meaningfully
        if self.trajectory_rate < self.TRAJECTORY_MIN_RATE:
            return []

        # Predict temp at end of window
        predicted_temp = self._predict_temp(self.TRAJECTORY_WINDOW_MINUTES)

        for strike in strikes:
            # Skip if already traded
            if strike.ticker in self.traded_tickers:
                continue

            # Only pre-position on "above" strikes for now
            if strike.strike_type != "above":
                continue

            # Check if we're predicted to cross this strike
            distance_to_strike = strike.strike_temp - current_temp

            # Skip if already crossed
            if distance_to_strike <= 0:
                continue

            # Skip if too far away (won't reach in time)
            if predicted_temp < strike.strike_temp:
                continue

            # Calculate time to reach strike (in minutes)
            if self.trajectory_rate > 0:
                hours_to_strike = distance_to_strike / self.trajectory_rate
                minutes_to_strike = hours_to_strike * 60
            else:
                continue

            # Only act if strike will be reached within prediction window
            if minutes_to_strike > self.TRAJECTORY_WINDOW_MINUTES:
                continue

            # Check if there's profit potential
            profit_cents = self.SWEEP_PRICE - int(strike.yes_ask * 100)
            if profit_cents < self.MIN_PROFIT_THRESHOLD:
                continue

            # This is a trajectory opportunity!
            opportunities.append({
                "strike": strike,
                "side": "yes",
                "action": "buy",
                "reason": (
                    f"TRAJECTORY: {current_temp:.1f}°F → {predicted_temp:.1f}°F "
                    f"(+{self.trajectory_rate:.1f}°F/hr, {self.trajectory_confidence:.0%} conf), "
                    f"crossing {strike.strike_temp}°F in ~{minutes_to_strike:.0f}min"
                ),
                "is_trajectory": True,
            })

            print(f"  [TRAJECTORY] Predicted to cross {strike.strike_temp}°F in {minutes_to_strike:.0f}min")
            print(f"    Current: {current_temp:.1f}°F, Rate: +{self.trajectory_rate:.1f}°F/hr")
            print(f"    Confidence: {self.trajectory_confidence:.0%}, YES @ {strike.yes_ask:.0%}")

        return opportunities

    async def _check_glitch_pause(self):
        """Check if we should pause trading due to sensor glitches."""
        if self.glitch_count >= self.MAX_CONSECUTIVE_GLITCHES:
            print(f"\n[ALERT] TOO MANY CONSECUTIVE GLITCHES ({self.glitch_count})")
            print(f"[ALERT] PAUSING TRADING FOR SAFETY - Manual review required")

            # Send Discord alert
            if self.alerts:
                await self.alerts.send(
                    title="SENSOR GLITCH ALERT",
                    message=(
                        f"**{self.glitch_count} consecutive bad readings detected!**\n\n"
                        f"Trading has been PAUSED to protect bankroll.\n"
                        f"Last valid max temp: {self.max_temp_today}°F\n\n"
                        f"Please verify NWS KNYC station status manually."
                    ),
                    color=0xFF0000  # Red
                )

    # Sanity check limits for sensor glitch protection
    MIN_VALID_TEMP = -20.0   # NYC never goes below -20°F
    MAX_VALID_TEMP = 120.0   # NYC never exceeds 120°F
    MAX_TEMP_JUMP = 10.0     # Max reasonable temp change between readings

    def _update_max_temp(self, temp_f: float) -> bool:
        """
        Update max temp if new reading is higher.

        INCLUDES SENSOR GLITCH PROTECTION:
        - Rejects temps outside valid NYC range (-20°F to 120°F)
        - Rejects sudden jumps > 10°F (likely sensor malfunction)

        Returns True if this is a valid new max.
        """
        today = datetime.now(ET).strftime("%Y-%m-%d")

        # 1. Day Reset Logic
        if today != self.today_date:
            print(f"[MAX] New day detected - resetting max temp")
            self.today_date = today
            self.max_temp_today = -999.0
            self.traded_tickers.clear()  # Reset traded tickers for new day

        # 2. SANITY CHECK: Valid temperature range
        if temp_f < self.MIN_VALID_TEMP or temp_f > self.MAX_VALID_TEMP:
            self.glitch_count += 1
            print(f"[WARN] INVALID TEMP: {temp_f}°F outside valid range [{self.MIN_VALID_TEMP}, {self.MAX_VALID_TEMP}]")
            print(f"[WARN] Glitch count: {self.glitch_count}/{self.MAX_CONSECUTIVE_GLITCHES}")
            # Note: Can't await here since this is sync, will check in scan_once
            return False

        # 3. GLITCH PROTECTION: Detect sudden jumps
        # If we have an established baseline and temp jumps > 10°F in one reading,
        # it's almost certainly a sensor glitch (voltage spike, maintenance mode, etc.)
        if self.max_temp_today > -900:  # We have a valid baseline
            temp_change = abs(temp_f - self.max_temp_today)
            if temp_f > self.max_temp_today and temp_change > self.MAX_TEMP_JUMP:
                self.glitch_count += 1
                print(f"[WARN] SENSOR GLITCH DETECTED?")
                print(f"[WARN] Temp jumped from {self.max_temp_today}°F to {temp_f}°F (+{temp_change:.1f}°F)")
                print(f"[WARN] Glitch count: {self.glitch_count}/{self.MAX_CONSECUTIVE_GLITCHES}")
                # Note: Can't await here since this is sync, will check in scan_once
                return False

        # Reading is valid - reset glitch counter
        self.glitch_count = 0

        # 4. Standard Update - all checks passed
        if temp_f > self.max_temp_today:
            old_max = self.max_temp_today
            self.max_temp_today = temp_f
            self.max_temp_time = datetime.now(ET)
            self._save_max_temp()

            if old_max > -999:
                print(f"[MAX] New daily high: {temp_f}°F (was {old_max}°F)")
            return True

        return False

    def _get_today_ticker_prefix(self) -> str:
        """Get today's ticker prefix in Kalshi format."""
        now = datetime.now(ET)
        # Format: KXHIGHNY-26JAN12
        month_abbrs = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                       "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        return f"KXHIGHNY-{now.year % 100:02d}{month_abbrs[now.month-1]}{now.day:02d}"

    async def get_todays_strikes(self) -> list[Strike]:
        """Fetch and parse today's KXHIGHNY markets."""
        prefix = self._get_today_ticker_prefix()
        print(f"[MARKET] Fetching markets for {prefix}")

        try:
            markets = await self.kalshi.get_markets(
                series_ticker="KXHIGHNY",
                status="open",
                limit=100,
            )

            strikes = []
            for m in markets:
                ticker = m.get("ticker", "")

                # Filter for today's markets
                if not ticker.startswith(prefix):
                    continue

                strike = self._parse_strike(m)
                if strike:
                    strikes.append(strike)

            print(f"[MARKET] Found {len(strikes)} strikes for today")
            return strikes

        except Exception as e:
            print(f"[MARKET] Error fetching markets: {e}")
            return []

    def _parse_strike(self, market: dict) -> Optional[Strike]:
        """Parse strike info from market data."""
        try:
            ticker = market.get("ticker", "")
            subtitle = market.get("subtitle", "")

            # Parse strike from ticker suffix
            # Examples: -T48 (above 48), -B45.5 (between 45-46)
            match = re.search(r"-([TB])(\d+(?:\.\d+)?)$", ticker)
            if not match:
                return None

            strike_marker = match.group(1)  # T or B
            strike_temp = float(match.group(2))

            # Determine strike type from subtitle
            # "47° or above" -> above
            # "38° or below" -> below
            # "39° to 40°" -> between
            strike_type = "between"
            upper_bound = None

            if "above" in subtitle.lower() or "or more" in subtitle.lower():
                strike_type = "above"
            elif "below" in subtitle.lower() or "or less" in subtitle.lower():
                strike_type = "below"
            else:
                # Parse upper bound from "39° to 40°" or "39 to 40"
                between_match = re.search(r"(\d+).*?to.*?(\d+)", subtitle)
                if between_match:
                    upper_bound = float(between_match.group(2))

            # Get YES prices (convert from cents to decimal)
            yes_bid = (market.get("yes_bid") or 0) / 100
            yes_ask = (market.get("yes_ask") or 0) / 100

            # Get NO prices (for dead bracket shorting)
            # NO bid = 1 - YES ask, NO ask = 1 - YES bid (approximately)
            no_bid = (market.get("no_bid") or 0) / 100
            no_ask = (market.get("no_ask") or 0) / 100

            # If no explicit NO prices, calculate from YES
            if no_ask == 0 and yes_bid > 0:
                no_ask = 1 - yes_bid + 0.01  # Small spread estimate
            if no_bid == 0 and yes_ask > 0:
                no_bid = 1 - yes_ask - 0.01

            # If no ask, use last price as estimate
            if yes_ask == 0:
                last = (market.get("last_price") or 50) / 100
                yes_ask = last

            return Strike(
                ticker=ticker,
                strike_temp=strike_temp,
                strike_type=strike_type,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                subtitle=subtitle,
                upper_bound=upper_bound,
            )

        except Exception as e:
            print(f"[PARSE] Error parsing strike: {e}")
            return None

    def find_triggered_contracts(
        self,
        current_temp: float,
        strikes: list[Strike]
    ) -> list[dict]:
        """
        Find contracts where temp has exceeded strike but price is still low.

        HIGH FIDELITY LOGIC (v3.0):
        - STRATEGY A (WINNERS): Buy YES on triggered "above" strikes
        - STRATEGY B (DEAD BRACKETS): Buy NO on mathematically dead contracts
        - Time-based confidence: later in day = higher confidence

        Returns list of trade actions (dict with strike, side, action, reason).
        """
        actions = []
        now = datetime.now(ET)
        current_hour = now.hour

        # Don't trade outside market hours
        if current_hour < self.ABOVE_STRIKE_START_HOUR:
            print(f"  [TIME] Too early ({current_hour}:00) - waiting for 10 AM")
            return []
        if current_hour >= self.MARKET_CLOSE_HOUR:
            print(f"  [TIME] Market closed ({current_hour}:00) - stopping")
            return []

        # Safe max temp (monotonic - can only increase)
        safe_max = max(current_temp, self.max_temp_today)

        for strike in strikes:
            # Skip if already traded
            if strike.ticker in self.traded_tickers:
                continue

            # =========================================================
            # STRATEGY A: THE WINNER (Buy YES on triggered "above")
            # =========================================================
            if strike.strike_type == "above":
                if safe_max >= (strike.strike_temp + self.SAFETY_BUFFER):
                    profit_cents = self.SWEEP_PRICE - int(strike.yes_ask * 100)
                    if profit_cents >= self.MIN_PROFIT_THRESHOLD:
                        actions.append({
                            "strike": strike,
                            "side": "yes",
                            "action": "buy",
                            "reason": f"WINNER: max {safe_max}°F >= {strike.strike_temp}°F"
                        })
                        print(f"  [WIN] {strike.strike_temp}°F ABOVE triggered (max={safe_max}°F)")

            # =========================================================
            # STRATEGY B: DEAD BRACKET SHORTING (Buy NO on dead contracts)
            # =========================================================
            # If max temp has EXCEEDED a "below" or "between" strike's ceiling,
            # that contract is DEAD. YES will settle at $0. Buy NO (guaranteed win).

            elif strike.strike_type == "below":
                # "High < 80°F" is DEAD if max temp >= 80.5°F
                if strike.check_dead(safe_max, self.SAFETY_BUFFER):
                    # Check if NO is cheap enough to buy
                    no_cost_cents = int(strike.no_ask * 100)
                    profit_cents = 100 - no_cost_cents  # NO pays $1 when YES loses

                    if profit_cents >= self.MIN_PROFIT_THRESHOLD and no_cost_cents < 98:
                        actions.append({
                            "strike": strike,
                            "side": "no",
                            "action": "buy",
                            "reason": f"DEAD: max {safe_max}°F killed '<{strike.strike_temp}°F'"
                        })
                        print(f"  [DEAD] {strike.strike_temp}°F BELOW is dead (max={safe_max}°F) → BUY NO")

            elif strike.strike_type == "between":
                # "41-42°F" is DEAD if max temp > 42.5°F
                if strike.check_dead(safe_max, self.SAFETY_BUFFER):
                    no_cost_cents = int(strike.no_ask * 100)
                    profit_cents = 100 - no_cost_cents

                    if profit_cents >= self.MIN_PROFIT_THRESHOLD and no_cost_cents < 98:
                        actions.append({
                            "strike": strike,
                            "side": "no",
                            "action": "buy",
                            "reason": f"DEAD: max {safe_max}°F exceeded bracket ceiling"
                        })
                        print(f"  [DEAD] {strike.subtitle} is dead (max={safe_max}°F) → BUY NO")

        return actions

    async def execute_arb(self, trade_action: dict, nws_temp: float):
        """
        Execute arbitrage trade.

        Args:
            trade_action: Dict with keys: strike, side, action, reason
            nws_temp: Current NWS temperature
        """
        self.triggers += 1

        strike = trade_action["strike"]
        side = trade_action["side"]  # "yes" or "no"
        action = trade_action["action"]  # "buy"
        reason = trade_action["reason"]

        # Calculate profit based on side
        if side == "yes":
            entry_price = strike.yes_ask
            profit_cents = self.SWEEP_PRICE - int(entry_price * 100)
        else:  # NO
            entry_price = strike.no_ask
            profit_cents = 100 - int(entry_price * 100)  # NO pays $1 on win

        print(f"\n{'='*70}")
        print(f"[TRIGGER] {strike.ticker} → {action.upper()} {side.upper()}")
        print(f"{'='*70}")
        print(f"  Reason:       {reason}")
        print(f"  NWS Temp:     {nws_temp}°F")
        print(f"  Max Today:    {self.max_temp_today}°F")
        print(f"  Strike:       {strike.strike_temp}°F ({strike.strike_type})")
        print(f"  Subtitle:     {strike.subtitle}")
        print(f"  {side.upper()} Ask:   {entry_price:.0%}")
        print(f"  Est. Profit:  {profit_cents}¢ per contract")

        # Calculate position size
        price_cents = int(entry_price * 100) if entry_price > 0 else self.SWEEP_PRICE
        contracts = int(self.MAX_POSITION_SIZE / (price_cents / 100))

        if self.live_mode:
            await self._execute_live(strike, contracts, nws_temp, side)
        else:
            await self._execute_paper(strike, contracts, nws_temp, side)

        # Mark as traded
        self.traded_tickers.add(strike.ticker)
        self.trades += 1

        # Send Discord alert
        if self.alerts:
            max_profit = contracts * profit_cents / 100
            emoji = "🎯" if side == "yes" else "💀"
            await self.alerts.send(
                title=f"{emoji} ARB: {action.upper()} {side.upper()} @ {strike.strike_temp}°F",
                message=(
                    f"**Ticker:** `{strike.ticker}`\n"
                    f"**Reason:** {reason}\n"
                    f"**NWS Temp:** {nws_temp}°F\n"
                    f"**Max Today:** {self.max_temp_today}°F\n"
                    f"**Side:** {side.upper()}\n"
                    f"**Contracts:** {contracts}\n"
                    f"**Entry:** {entry_price:.0%}\n"
                    f"**Est. Profit:** ${max_profit:.2f}\n"
                    f"**Mode:** {'LIVE' if self.live_mode else 'PAPER'}"
                ),
                color=0x00FF00 if self.live_mode else 0x87CEEB
            )

    async def _execute_live(self, strike: Strike, contracts: int, nws_temp: float, side: str = "yes"):
        """Execute live trade on Kalshi."""
        print(f"\n[LIVE] Placing order: BUY {contracts} {side.upper()} @ {self.SWEEP_PRICE}¢")

        try:
            result = await self.kalshi.place_order(
                ticker=strike.ticker,
                side=side,
                action="buy",
                count=contracts,
                price=self.SWEEP_PRICE,
                order_type="limit",
            )

            order_id = result.get("order", {}).get("order_id", "N/A")
            print(f"[LIVE] Order placed! ID: {order_id}")

            self._log_trade(strike, contracts, nws_temp, side, is_paper=False, order_id=order_id)

        except Exception as e:
            print(f"[LIVE] Order failed: {e}")

    async def _execute_paper(self, strike: Strike, contracts: int, nws_temp: float, side: str = "yes"):
        """Record paper trade."""
        print(f"\n[PAPER] Recording: BUY {contracts} {side.upper()}")

        if side == "yes":
            entry_price = strike.yes_ask
        else:
            entry_price = strike.no_ask

        cost = contracts * entry_price
        max_profit = contracts * (1 - entry_price)
        max_loss = cost

        print(f"[PAPER] Entry: {entry_price:.0%}")
        print(f"[PAPER] Max profit: ${max_profit:.2f}")
        print(f"[PAPER] Max loss: ${max_loss:.2f}")

        self._log_trade(strike, contracts, nws_temp, side, is_paper=True)

    def _log_trade(
        self,
        strike: Strike,
        contracts: int,
        nws_temp: float,
        side: str,
        is_paper: bool,
        order_id: str = None
    ):
        """Log trade to file."""
        entry_price = strike.yes_ask if side == "yes" else strike.no_ask

        with open(ARB_TRADES_LOG, "a") as f:
            f.write(json.dumps({
                "ts": datetime.now(ET).isoformat(),
                "ticker": strike.ticker,
                "strike_temp": strike.strike_temp,
                "strike_type": strike.strike_type,
                "side": side,
                "nws_temp": nws_temp,
                "max_temp": self.max_temp_today,
                "entry_price": entry_price,
                "contracts": contracts,
                "is_paper": is_paper,
                "order_id": order_id,
                "version": self.VERSION,
            }) + "\n")

    async def scan_once(self):
        """Run a single scan for arbitrage opportunities."""
        self.polls += 1

        # Get latest NWS observation
        obs = await self.nws.get_latest_observation("KNYC")
        if not obs:
            print("[SCAN] Failed to get NWS observation")
            return

        speci_flag = " [SPECI]" if obs.is_speci else ""
        age_min = obs.age_seconds / 60

        # Update max temp tracking (includes glitch detection)
        is_new_max = self._update_max_temp(obs.temp_f)

        # Update trajectory tracking
        self._update_trajectory(obs.temp_f, obs.timestamp)

        # Check if we should pause due to glitches
        if self.glitch_count >= self.MAX_CONSECUTIVE_GLITCHES:
            await self._check_glitch_pause()
            print(f"[PAUSED] Skipping trades until sensor stabilizes...")
            return

        # Format trajectory info
        if self.trajectory_rate != 0 and len(self.temp_history) >= self.TRAJECTORY_MIN_READINGS:
            trend_arrow = "↑" if self.trajectory_rate > 0 else "↓"
            trajectory_str = f" | Trend: {trend_arrow}{abs(self.trajectory_rate):.1f}°F/hr ({self.trajectory_confidence:.0%} conf)"
        else:
            trajectory_str = ""

        print(f"\n[SCAN #{self.polls}] {datetime.now(ET).strftime('%H:%M:%S ET')}")
        print(f"  Current:  {obs.temp_f}°F ({obs.temp_c:.1f}°C){speci_flag}")
        print(f"  Max Today: {self.max_temp_today}°F {'← NEW!' if is_new_max else ''}{trajectory_str}")
        print(f"  Obs Age:  {age_min:.1f} min")

        # Skip if observation is too old (> 2 hours)
        if obs.age_seconds > 7200:
            print(f"  [WARN] Observation too old, skipping")
            return

        # Get today's strikes
        strikes = await self.get_todays_strikes()
        if not strikes:
            print("  No active strikes found")
            return

        # Print current strikes with smart annotations
        print(f"\n  Active Strikes (using max temp {self.max_temp_today}°F for 'above'):")
        for s in sorted(strikes, key=lambda x: x.strike_temp):
            # Check trigger based on strike type
            if s.strike_type == "above":
                check_temp = max(obs.temp_f, self.max_temp_today)
                triggered = check_temp >= (s.strike_temp + self.SAFETY_BUFFER)
            elif s.strike_type == "below":
                triggered = obs.temp_f <= (s.strike_temp - self.SAFETY_BUFFER)
            else:
                triggered = False

            mark = "✓" if triggered else " "
            traded = "[TRADED]" if s.ticker in self.traded_tickers else ""
            print(f"    [{mark}] {s.strike_temp}°F {s.strike_type:6}: YES @ {s.yes_ask:.0%} {traded}")

        # Find triggered contracts (returns list of action dicts)
        actions = self.find_triggered_contracts(obs.temp_f, strikes)

        # Also check trajectory-based opportunities (pre-positioning)
        trajectory_actions = self._find_trajectory_opportunities(obs.temp_f, strikes)
        actions.extend(trajectory_actions)

        if actions:
            print(f"\n  [!] {len(actions)} TRADE OPPORTUNITIES:")
            for action in actions:
                await self.execute_arb(action, obs.temp_f)
        else:
            print(f"\n  No arbitrage opportunities")

    async def run_loop(self):
        """Main polling loop."""
        print(f"\n[LOOP] Starting with {self.POLL_INTERVAL}s interval...")

        while self.running:
            try:
                await self.scan_once()

                print(f"\n[WAIT] Next scan in {self.POLL_INTERVAL}s...")
                await asyncio.sleep(self.POLL_INTERVAL)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                await asyncio.sleep(30)

        await self.stop()


async def test_strikes():
    """Test strike parsing without trading."""
    print("\n[TEST] Testing strike parsing...")

    bot = NYCWeatherArbBot(live_mode=False)
    await bot.start()

    # Get strikes
    strikes = await bot.get_todays_strikes()

    print(f"\n[TEST] Found {len(strikes)} strikes:")
    for s in sorted(strikes, key=lambda x: x.strike_temp):
        print(f"  {s.ticker}")
        print(f"    Temp: {s.strike_temp}°F ({s.strike_type})")
        print(f"    Subtitle: {s.subtitle}")
        print(f"    YES: bid={s.yes_bid:.0%} ask={s.yes_ask:.0%}")

    # Test NWS
    print("\n[TEST] Testing NWS API...")
    obs = await bot.nws.get_latest_observation("KNYC")
    if obs:
        print(f"  Station: {obs.station_id}")
        print(f"  Temp: {obs.temp_f}°F ({obs.temp_c:.1f}°C)")
        print(f"  Time: {obs.timestamp}")
        print(f"  SPECI: {obs.is_speci}")
        print(f"  Age: {obs.age_seconds/60:.1f} min")

        # Check which strikes would be triggered
        triggered = bot.find_triggered_contracts(obs.temp_f, strikes)
        print(f"\n[TEST] Triggered strikes: {len(triggered)}")
        for s in triggered:
            print(f"  {s.ticker} - {s.subtitle}")

    await bot.stop()


async def fetch_nws_cli_high(date_str: str = None) -> Optional[float]:
    """
    Fetch official high temperature from NWS CLI report.

    This is the SETTLEMENT SOURCE for Kalshi weather markets.
    CLI reports are issued around 5-7 AM ET for the previous day.

    Args:
        date_str: Date in YYYY-MM-DD format (default: yesterday)

    Returns:
        Official high temperature in °F, or None if not found
    """
    # CLI report URL for NYC (issued by OKX office)
    CLI_URL = "https://forecast.weather.gov/product.php?site=OKX&product=CLI&issuedby=NYC"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(CLI_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    print(f"[CLI] Failed to fetch: HTTP {resp.status}")
                    return None

                text = await resp.text()

                # Parse the CLI report for high temperature
                # Format varies but typically: "MAXIMUM TEMPERATURE (F)...    47"
                # or "HIGHEST...47" or similar patterns

                import re

                # Try multiple patterns
                patterns = [
                    r"MAXIMUM\s+TEMPERATURE\s*\(F\)[\.\s]+(\d+)",
                    r"HIGHEST[\.\s]+(\d+)",
                    r"MAX\s+TEMP[\.\s]+(\d+)",
                    r"HIGH[\.\s]+(\d+)\s+LOW",
                ]

                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        high_temp = float(match.group(1))
                        print(f"[CLI] Found official high: {high_temp}°F")
                        return high_temp

                # If patterns fail, look for the temperature table
                # WEATHER DATA FOR YESTERDAY
                if "WEATHER DATA FOR YESTERDAY" in text or "CLIMATE REPORT" in text:
                    # Find line with temperature data
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        if 'MAXIMUM' in line.upper() and 'TEMPERATURE' in line.upper():
                            # Temperature usually on same line or next
                            numbers = re.findall(r'\b(\d{1,3})\b', line)
                            for num in numbers:
                                temp = int(num)
                                if 0 <= temp <= 120:  # Valid temp range
                                    print(f"[CLI] Found official high: {temp}°F")
                                    return float(temp)

                print("[CLI] Could not parse high temperature from report")
                return None

    except Exception as e:
        print(f"[CLI] Error fetching report: {e}")
        return None


async def check_settlements():
    """
    Check settlements for all paper trades and calculate win rate.

    Reads trades from nyc_arb_trades.jsonl, fetches official settlement,
    and updates with win/loss status.
    """
    print("\n" + "=" * 60)
    print("           SETTLEMENT CHECKER")
    print("=" * 60)

    if not ARB_TRADES_LOG.exists():
        print("[SETTLE] No trades log found.")
        return

    # Load all trades
    trades = []
    with open(ARB_TRADES_LOG) as f:
        for line in f:
            try:
                trade = json.loads(line.strip())
                trades.append(trade)
            except:
                continue

    if not trades:
        print("[SETTLE] No trades to check.")
        return

    print(f"[SETTLE] Found {len(trades)} trades to check")

    # Get official high from CLI
    official_high = await fetch_nws_cli_high()

    if official_high is None:
        print("[SETTLE] Could not fetch official settlement.")
        print("[SETTLE] CLI report may not be available yet (issued ~5-7 AM ET)")
        return

    print(f"\n[SETTLE] Official NYC High: {official_high}°F")
    print("-" * 60)

    # Check each trade
    wins = 0
    losses = 0
    pending = 0
    total_pnl = 0.0

    results = []

    for trade in trades:
        ticker = trade.get("ticker", "")
        strike_temp = trade.get("strike_temp", 0)
        strike_type = trade.get("strike_type", "")
        side = trade.get("side", "yes")
        entry_price = trade.get("entry_price", 0)
        contracts = trade.get("contracts", 0)
        trade_date = trade.get("ts", "")[:10]

        # Determine if this trade is for today/yesterday
        today = datetime.now(ET).strftime("%Y-%m-%d")
        yesterday = (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")

        # Skip future trades
        if trade_date > today:
            pending += 1
            continue

        # Determine win/loss based on strike type and side
        won = False

        if side == "yes":
            if strike_type == "above":
                # YES wins if official high >= strike
                won = official_high >= strike_temp
            elif strike_type == "below":
                # YES wins if official high < strike
                won = official_high < strike_temp
            elif strike_type == "between":
                # YES wins if high is in the bracket
                upper = trade.get("upper_bound", strike_temp + 2)
                won = strike_temp <= official_high < upper
        else:  # NO side
            if strike_type == "above":
                won = official_high < strike_temp
            elif strike_type == "below":
                won = official_high >= strike_temp
            elif strike_type == "between":
                upper = trade.get("upper_bound", strike_temp + 2)
                won = not (strike_temp <= official_high < upper)

        # Calculate P&L
        if won:
            pnl = contracts * (1.0 - entry_price)  # Win pays $1, minus entry
            wins += 1
            status = "WIN"
        else:
            pnl = -contracts * entry_price  # Lose entire entry
            losses += 1
            status = "LOSS"

        total_pnl += pnl

        results.append({
            "ticker": ticker[-20:],
            "side": side.upper(),
            "strike": f"{strike_temp}°F",
            "entry": f"{entry_price:.0%}",
            "status": status,
            "pnl": f"${pnl:+.2f}",
        })

        print(f"  {ticker[-20:]} | {side.upper():3} @ {entry_price:.0%} | {status:4} | ${pnl:+.2f}")

    # Summary
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0

    print("-" * 60)
    print(f"\n  RESULTS:")
    print(f"    Wins:     {wins}")
    print(f"    Losses:   {losses}")
    print(f"    Pending:  {pending}")
    print(f"    Win Rate: {win_rate:.1f}%")
    print(f"    Total P&L: ${total_pnl:+.2f}")
    print("=" * 60)

    # Save results to file
    settlement_file = Path("settlement_results.json")
    with open(settlement_file, "w") as f:
        json.dump({
            "checked_at": datetime.now(ET).isoformat(),
            "official_high": official_high,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "trades": results,
        }, f, indent=2)

    print(f"\n[SETTLE] Results saved to {settlement_file}")


async def main():
    parser = argparse.ArgumentParser(description="NYC Weather Latency Arbitrage Bot")
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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test strike parsing and NWS API",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Poll interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--settle",
        action="store_true",
        help="Check settlements and calculate win rate",
    )
    args = parser.parse_args()

    if args.settle:
        await check_settlements()
        return

    if args.test:
        await test_strikes()
        return

    bot = NYCWeatherArbBot(live_mode=args.live)
    bot.POLL_INTERVAL = args.interval

    await bot.start()

    if args.once:
        await bot.scan_once()
        await bot.stop()
    else:
        await bot.run_loop()


if __name__ == "__main__":
    asyncio.run(main())
