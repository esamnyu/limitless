#!/usr/bin/env python3
"""
WEATHER SNIPER v3.0 - Multi-City Predictive Weather Trading Bot

Supports: NYC (New York), CHI (Chicago)

Strategies:
  A. Midnight High - Post-frontal cold advection detection
  B. Wind Mixing Penalty - Mechanical mixing suppresses heating
  C. Rounding Arbitrage - NWS rounds x.50 up, x.49 down
  D. Wet Bulb Protocol - Evaporative cooling from rain into dry air
  E. MOS Consensus - Fade NWS when models disagree

Execution:
  - Smart Pegging - Bid+1 instead of hitting the Ask
  - Human-in-the-loop confirmation for all trades

Usage:
  python3 sniper.py                    # NYC (default)
  python3 sniper.py --city CHI         # Chicago
  python3 sniper.py --city NYC --live  # NYC with live trading
"""

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import aiofiles
import aiohttp
from dotenv import load_dotenv

from kalshi_client import KalshiClient
from config import (
    # City configuration
    StationConfig,
    get_station_config,
    DEFAULT_CITY,
    STATIONS,
    # Trading parameters
    MAX_POSITION_PCT,
    EDGE_THRESHOLD_BUY,
    MAX_ENTRY_PRICE_CENTS,
    TAKE_PROFIT_ROI_PCT,
    CAPITAL_EFFICIENCY_THRESHOLD_CENTS,
    # Smart Pegging
    MAX_SPREAD_TO_CROSS_CENTS,
    PEG_OFFSET_CENTS,
    MIN_BID_CENTS,
    # Weather strategy parameters
    WIND_PENALTY_LIGHT_THRESHOLD_MPH,
    WIND_PENALTY_HEAVY_THRESHOLD_MPH,
    WIND_PENALTY_LIGHT_DEGREES,
    WIND_PENALTY_HEAVY_DEGREES,
    WIND_GUST_MULTIPLIER,
    WIND_GUST_THRESHOLD_MPH,
    MIDNIGHT_HOUR_START,
    MIDNIGHT_HOUR_END,
    AFTERNOON_HOUR_START,
    AFTERNOON_HOUR_END,
    # Wet Bulb parameters
    WET_BULB_PRECIP_THRESHOLD_PCT,
    WET_BULB_DEPRESSION_MIN_F,
    WET_BULB_FACTOR_LIGHT,
    WET_BULB_FACTOR_HEAVY,
    WET_BULB_HEAVY_PRECIP_THRESHOLD,
    # MOS parameters
    MOS_DIVERGENCE_THRESHOLD_F,
    # Confidence levels
    CONFIDENCE_MIDNIGHT_HIGH,
    CONFIDENCE_WIND_PENALTY,
    CONFIDENCE_WET_BULB,
    CONFIDENCE_MOS_FADE,
    # API settings
    NWS_TIMEOUT_TOTAL_SEC,
    NWS_TIMEOUT_CONNECT_SEC,
    FORECAST_HOURS_AHEAD,
    FILLS_FETCH_LIMIT,
    # File paths
    TRADES_LOG_FILE,
    # Logging
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


def validate_credentials() -> tuple[str, str]:
    """Validate Kalshi API credentials exist and are accessible."""
    api_key = os.getenv("KALSHI_API_KEY_ID")
    if not api_key:
        raise ConfigurationError(
            "KALSHI_API_KEY_ID not set in environment. "
            "Please set this in your .env file or environment variables."
        )

    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not private_key_path:
        raise ConfigurationError(
            "KALSHI_PRIVATE_KEY_PATH not set in environment. "
            "Please set this in your .env file or environment variables."
        )

    key_path = Path(private_key_path)
    if not key_path.exists():
        raise ConfigurationError(f"Private key file not found at: {private_key_path}")

    if not key_path.is_file():
        raise ConfigurationError(f"Private key path is not a file: {private_key_path}")

    try:
        key_path.read_bytes()
    except PermissionError:
        raise ConfigurationError(f"Cannot read private key file (permission denied): {private_key_path}")
    except Exception as e:
        raise ConfigurationError(f"Cannot read private key file: {private_key_path} - {e}")

    return api_key, private_key_path


@dataclass
class HourlyForecast:
    """Hourly forecast data from NWS."""
    time: datetime
    temp_f: float
    wind_speed_mph: float
    wind_gust_mph: float
    short_forecast: str
    is_daytime: bool
    precip_prob: int = 0
    dewpoint_f: float = 0.0


@dataclass
class MOSForecast:
    """MOS (Model Output Statistics) forecast data."""
    source: str  # "MAV" (GFS) or "MET" (NAM)
    valid_date: datetime
    max_temp_f: float
    min_temp_f: float
    precip_prob_12hr: int = 0


@dataclass
class TradeTicket:
    """Trade recommendation with all analysis data."""
    # NWS data
    nws_forecast_high: float
    # Physics adjustments
    physics_high: float
    wind_penalty: float
    wet_bulb_penalty: float
    wind_gust: float
    # Strategy flags
    is_midnight_risk: bool
    midnight_temp: Optional[float]
    afternoon_temp: Optional[float]
    is_wet_bulb_risk: bool
    is_mos_fade: bool
    # MOS data
    mav_high: Optional[float] = None
    met_high: Optional[float] = None
    mos_consensus: Optional[float] = None
    # Target
    target_bracket_low: int = 0
    target_bracket_high: int = 0
    target_ticker: str = ""
    # Market data
    current_bid_cents: int = 0
    current_ask_cents: int = 0
    entry_price_cents: int = 0
    implied_odds: float = 0.0
    spread_cents: int = 0
    # Analysis
    estimated_edge: float = 0.0
    recommendation: str = "PASS"
    confidence: int = 0
    rationale: str = ""


@dataclass
class ExitSignal:
    """Exit recommendation for position management."""
    ticker: str
    signal_type: str
    contracts_held: int
    avg_entry_cents: int
    current_bid_cents: int
    roi_percent: float
    target_bracket: tuple[int, int]
    nws_forecast_high: float
    thesis_valid: bool
    sell_qty: int
    sell_price_cents: int
    rationale: str


class MOSClient:
    """Client for fetching MOS (Model Output Statistics) data."""

    def __init__(self, station_config: StationConfig):
        self.station_config = station_config
        self.session: Optional[aiohttp.ClientSession] = None
        self.tz = ZoneInfo(station_config.timezone)

    async def start(self):
        self.session = aiohttp.ClientSession(
            headers={
                "User-Agent": "WeatherSniper/3.0 (contact: weather-sniper@example.com)",
                "Accept": "text/plain",
            },
            timeout=aiohttp.ClientTimeout(total=NWS_TIMEOUT_TOTAL_SEC, connect=NWS_TIMEOUT_CONNECT_SEC),
        )
        logger.info(f"MOS client initialized for {self.station_config.station_id}")

    async def stop(self):
        if self.session:
            await self.session.close()

    async def fetch_mos(self, url: str, source: str) -> Optional[MOSForecast]:
        """Fetch and parse MOS bulletin."""
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(f"MOS {source} fetch returned {resp.status}")
                    return None
                text = await resp.text()
                return self._parse_mos(text, source)
        except asyncio.TimeoutError:
            logger.error(f"MOS {source} request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"MOS {source} HTTP error: {e}")
            return None
        except Exception as e:
            logger.exception(f"MOS {source} unexpected error: {e}")
            return None

    def _parse_mos(self, text: str, source: str) -> Optional[MOSForecast]:
        """Parse MOS text bulletin to extract max temperature."""
        try:
            lines = text.strip().split('\n')

            dt_line = None
            temp_line = None

            for i, line in enumerate(lines):
                if line.strip().startswith('DT'):
                    dt_line = line
                if line.strip().startswith('X/N') or line.strip().startswith('N/X'):
                    temp_line = line
                    break

            if not temp_line:
                logger.debug(f"Could not find X/N line in {source} MOS")
                return None

            parts = temp_line.split()
            if len(parts) < 2:
                return None

            temps = []
            for p in parts[1:]:
                try:
                    temps.append(int(p))
                except ValueError:
                    continue

            if not temps:
                return None

            max_temp = temps[0] if temps else None
            min_temp = temps[1] if len(temps) > 1 else None

            if max_temp is None:
                return None

            valid_date = datetime.now(self.tz).date() + timedelta(days=1)

            return MOSForecast(
                source=source,
                valid_date=datetime(valid_date.year, valid_date.month, valid_date.day, tzinfo=self.tz),
                max_temp_f=float(max_temp),
                min_temp_f=float(min_temp) if min_temp else 0.0,
            )

        except Exception as e:
            logger.debug(f"Error parsing {source} MOS: {e}")
            return None

    async def get_mav(self) -> Optional[MOSForecast]:
        """Get GFS MOS (MAV) forecast."""
        return await self.fetch_mos(self.station_config.mos_mav_url, "MAV")

    async def get_met(self) -> Optional[MOSForecast]:
        """Get NAM MOS (MET) forecast."""
        return await self.fetch_mos(self.station_config.mos_met_url, "MET")


class NWSClient:
    """NWS API client for observations and hourly forecasts."""

    def __init__(self, station_config: StationConfig):
        self.station_config = station_config
        self.session: Optional[aiohttp.ClientSession] = None
        self.gridpoint_url: str = station_config.nws_hourly_forecast_url
        self.observation_url: str = station_config.nws_observation_url

    async def start(self):
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": f"WeatherSniper/3.0 ({self.station_config.city_code})", "Accept": "application/geo+json"},
            timeout=aiohttp.ClientTimeout(total=NWS_TIMEOUT_TOTAL_SEC, connect=NWS_TIMEOUT_CONNECT_SEC),
        )
        logger.info(f"NWS client initialized for {self.station_config.station_id} (gridpoint: {self.gridpoint_url})")

    async def stop(self):
        if self.session:
            await self.session.close()
            logger.debug("NWS client stopped")

    async def get_current_temp(self) -> Optional[float]:
        """Get current temperature from station."""
        try:
            async with self.session.get(self.observation_url) as resp:
                if resp.status != 200:
                    logger.warning(f"NWS observation returned status {resp.status}")
                    return None
                props = (await resp.json()).get("properties", {})
                temp_c = props.get("temperature", {}).get("value")
                if temp_c is None:
                    return None
                return round((temp_c * 1.8) + 32, 1)
        except asyncio.TimeoutError:
            logger.error("NWS observation request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"NWS observation HTTP error: {e}")
            return None
        except Exception as e:
            logger.exception(f"NWS observation unexpected error: {e}")
            return None

    async def get_hourly_forecast(self) -> list[HourlyForecast]:
        """Get hourly forecast including wind, precip, and dewpoint data."""
        try:
            async with self.session.get(self.gridpoint_url) as resp:
                if resp.status != 200:
                    logger.error(f"NWS hourly forecast returned status {resp.status}")
                    return []
                data = await resp.json()
                periods = data.get("properties", {}).get("periods", [])

                forecasts = []
                for p in periods[:FORECAST_HOURS_AHEAD]:
                    try:
                        time = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00"))
                        temp_f = float(p.get("temperature", 0))

                        wind_str = p.get("windSpeed", "0 mph")
                        wind_match = re.search(r"(\d+)\s*(?:to\s*(\d+))?\s*mph", wind_str, re.I)
                        if wind_match:
                            wind_speed = float(wind_match.group(2) or wind_match.group(1))
                        else:
                            wind_speed = 0.0

                        wind_gust = (
                            wind_speed * WIND_GUST_MULTIPLIER
                            if wind_speed > WIND_GUST_THRESHOLD_MPH
                            else wind_speed
                        )

                        precip_val = p.get("probabilityOfPrecipitation", {}).get("value")
                        precip_prob = int(precip_val) if precip_val is not None else 0

                        dew_val = p.get("dewpoint", {}).get("value")
                        dew_c = float(dew_val) if dew_val is not None else 0.0
                        dew_f = (dew_c * 1.8) + 32

                        forecasts.append(HourlyForecast(
                            time=time,
                            temp_f=temp_f,
                            wind_speed_mph=wind_speed,
                            wind_gust_mph=wind_gust,
                            short_forecast=p.get("shortForecast", ""),
                            is_daytime=p.get("isDaytime", False),
                            precip_prob=precip_prob,
                            dewpoint_f=dew_f,
                        ))
                    except (KeyError, ValueError) as e:
                        logger.debug(f"Skipping malformed forecast period: {e}")
                        continue

                logger.info(f"Fetched {len(forecasts)} hourly forecast periods")
                return forecasts

        except asyncio.TimeoutError:
            logger.error("NWS hourly forecast request timed out")
            return []
        except aiohttp.ClientError as e:
            logger.error(f"NWS hourly forecast HTTP error: {e}")
            return []
        except Exception as e:
            logger.exception(f"NWS hourly forecast unexpected error: {e}")
            return []


class WeatherSniper:
    """Multi-city predictive weather trading bot with multi-strategy analysis."""
    VERSION = "3.0.0"

    def __init__(self, city_code: str = DEFAULT_CITY, live_mode: bool = False):
        self.city_code = city_code.upper()
        self.station_config = get_station_config(self.city_code)
        self.tz = ZoneInfo(self.station_config.timezone)
        self.live_mode = live_mode
        self.nws: Optional[NWSClient] = None
        self.mos: Optional[MOSClient] = None
        self.kalshi: Optional[KalshiClient] = None
        self.balance = 0.0

    async def start(self):
        print(f"\n{'='*60}")
        print(f"  WEATHER SNIPER v{self.VERSION}")
        print(f"  City: {self.station_config.city_name}")
        print(f"  Station: {self.station_config.station_id}")
        print(f"  Strategies: A-Midnight | B-Wind | D-WetBulb | E-MOS")
        print(f"{'='*60}")

        logger.info(f"Starting Weather Sniper v{self.VERSION} for {self.city_code}")

        print("\n[INIT] Validating credentials...")
        try:
            api_key, private_key_path = validate_credentials()
            logger.info("Credentials validated successfully")
            print("[INIT] Credentials validated successfully")
        except ConfigurationError as e:
            logger.critical(f"Configuration error: {e}")
            print(f"\n[FATAL] Configuration Error: {e}")
            raise SystemExit(1)

        # Initialize NWS client with city-specific config
        self.nws = NWSClient(self.station_config)
        await self.nws.start()

        # Initialize MOS client with city-specific config
        self.mos = MOSClient(self.station_config)
        await self.mos.start()

        # Initialize Kalshi client
        self.kalshi = KalshiClient(
            api_key_id=api_key,
            private_key_path=private_key_path,
            demo_mode=False,
        )
        await self.kalshi.start()

        self.balance = await self.kalshi.get_balance()
        if self.balance == 0:
            logger.warning("Balance is $0.00 - check API connection or account funding")
            print("[WARN] Balance is $0.00 - check API connection or account funding")

        mode_str = "LIVE" if self.live_mode else "ANALYSIS ONLY"
        max_position = self.balance * MAX_POSITION_PCT
        logger.info(f"Initialized: mode={mode_str}, balance=${self.balance:.2f}")

        print(f"\n[INIT] Mode: {mode_str}")
        print(f"[INIT] Balance: ${self.balance:.2f}")
        print(f"[INIT] Max Position: ${max_position:.2f} ({MAX_POSITION_PCT:.0%} of NLV)")

    async def stop(self):
        if self.nws:
            await self.nws.stop()
        if self.mos:
            await self.mos.stop()
        if self.kalshi:
            await self.kalshi.stop()
        logger.info("Weather Sniper stopped")

    # =========================================================================
    # STRATEGY CALCULATIONS (Physics logic unchanged)
    # =========================================================================

    def calculate_wind_penalty(self, wind_gust_mph: float) -> float:
        """Strategy B: Wind Mixing Penalty."""
        if wind_gust_mph > WIND_PENALTY_HEAVY_THRESHOLD_MPH:
            return WIND_PENALTY_HEAVY_DEGREES
        elif wind_gust_mph > WIND_PENALTY_LIGHT_THRESHOLD_MPH:
            return WIND_PENALTY_LIGHT_DEGREES
        return 0.0

    def calculate_wet_bulb_penalty(self, temp_f: float, dewpoint_f: float, precip_prob: int) -> float:
        """Strategy D: Wet Bulb / Evaporative Cooling Risk."""
        if precip_prob < WET_BULB_PRECIP_THRESHOLD_PCT:
            return 0.0

        depression = temp_f - dewpoint_f

        if depression < WET_BULB_DEPRESSION_MIN_F:
            return 0.0

        factor = WET_BULB_FACTOR_HEAVY if precip_prob >= WET_BULB_HEAVY_PRECIP_THRESHOLD else WET_BULB_FACTOR_LIGHT

        penalty = depression * factor
        return round(penalty, 1)

    def check_mos_divergence(
        self, nws_high: float, mav_high: Optional[float], met_high: Optional[float]
    ) -> tuple[bool, Optional[float]]:
        """Strategy E: Check if NWS diverges from MOS consensus."""
        mos_values = [v for v in [mav_high, met_high] if v is not None]
        if not mos_values:
            return False, None

        mos_consensus = sum(mos_values) / len(mos_values)

        if nws_high > mos_consensus + MOS_DIVERGENCE_THRESHOLD_F:
            return True, mos_consensus

        return False, mos_consensus

    def check_midnight_high(self, forecasts: list[HourlyForecast]) -> tuple[bool, Optional[float], Optional[float]]:
        """Strategy A: Midnight High Detection."""
        now = datetime.now(self.tz)
        tomorrow = now.date() + timedelta(days=1)

        midnight_temp = None
        afternoon_temp = None

        for f in forecasts:
            f_local = f.time.astimezone(self.tz)
            f_date = f_local.date()
            f_hour = f_local.hour

            if f_date == tomorrow and MIDNIGHT_HOUR_START <= f_hour <= MIDNIGHT_HOUR_END:
                midnight_temp = f.temp_f

            if f_date == tomorrow and AFTERNOON_HOUR_START <= f_hour <= AFTERNOON_HOUR_END:
                afternoon_temp = f.temp_f

        is_midnight = False
        if midnight_temp is not None and afternoon_temp is not None:
            is_midnight = midnight_temp > afternoon_temp

        return is_midnight, midnight_temp, afternoon_temp

    def get_peak_forecast(self, forecasts: list[HourlyForecast]) -> Optional[HourlyForecast]:
        """Get the forecast period with the highest temperature for tomorrow."""
        now = datetime.now(self.tz)
        tomorrow = now.date() + timedelta(days=1)

        tomorrow_forecasts = [
            f for f in forecasts
            if f.time.astimezone(self.tz).date() == tomorrow
        ]

        if not tomorrow_forecasts:
            return None

        return max(tomorrow_forecasts, key=lambda x: x.temp_f)

    def temp_to_bracket(self, temp_f: float) -> tuple[int, int]:
        """Convert temperature to bracket bounds (low, high)."""
        rounded = int(Decimal(str(temp_f)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        low = (rounded // 2) * 2 - 1
        high = low + 2
        return low, high

    # =========================================================================
    # MARKET OPERATIONS
    # =========================================================================

    async def get_kalshi_markets(self) -> list[dict]:
        """Fetch today's and tomorrow's markets for this city."""
        try:
            markets = await self.kalshi.get_markets(
                series_ticker=self.station_config.series_ticker, status="open", limit=100
            )
            logger.debug(f"Fetched {len(markets)} markets for {self.station_config.series_ticker}")
            return markets
        except Exception as e:
            logger.error(f"Market fetch failed: {e}")
            return []

    def find_target_market(self, markets: list[dict], target_temp: float) -> Optional[dict]:
        """Find the market bracket containing the target temperature."""
        now = datetime.now(self.tz)
        tomorrow = now + timedelta(days=1)
        months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
        tomorrow_str = f"{tomorrow.year % 100:02d}{months[tomorrow.month-1]}{tomorrow.day:02d}"

        for m in markets:
            ticker = m.get("ticker", "")
            if tomorrow_str not in ticker:
                continue

            subtitle = m.get("subtitle", "").lower()

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

    def calculate_smart_entry_price(self, bid: int, ask: int) -> tuple[int, str]:
        """Smart Pegging: Calculate optimal entry price."""
        spread = ask - bid

        if bid < MIN_BID_CENTS:
            return 0, "No valid bid"

        if spread <= MAX_SPREAD_TO_CROSS_CENTS:
            return ask, f"Tight spread ({spread}c) - taking ask"
        else:
            entry = bid + PEG_OFFSET_CENTS
            return entry, f"Wide spread ({spread}c) - pegging bid+{PEG_OFFSET_CENTS}"

    # =========================================================================
    # TRADE TICKET GENERATION
    # =========================================================================

    def generate_trade_ticket(
        self,
        peak_forecast: HourlyForecast,
        is_midnight: bool,
        midnight_temp: Optional[float],
        afternoon_temp: Optional[float],
        mav_high: Optional[float],
        met_high: Optional[float],
        market: Optional[dict],
    ) -> TradeTicket:
        """Generate a comprehensive trade ticket with all analysis."""

        nws_high = peak_forecast.temp_f
        wind_gust = peak_forecast.wind_gust_mph
        dewpoint = peak_forecast.dewpoint_f
        precip_prob = peak_forecast.precip_prob

        wind_penalty = self.calculate_wind_penalty(wind_gust)
        wet_bulb_penalty = self.calculate_wet_bulb_penalty(nws_high, dewpoint, precip_prob)

        is_mos_fade, mos_consensus = self.check_mos_divergence(nws_high, mav_high, met_high)

        physics_high = nws_high - wind_penalty - wet_bulb_penalty

        if is_midnight and midnight_temp:
            physics_high = midnight_temp

        if is_mos_fade and mos_consensus:
            physics_high = min(physics_high, mos_consensus)

        bracket_low, bracket_high = self.temp_to_bracket(physics_high)

        if market:
            ticker = market.get("ticker", "")
            bid = market.get("yes_bid", 0)
            ask = market.get("yes_ask", 0)
            entry_price, peg_rationale = self.calculate_smart_entry_price(bid, ask)
            spread = ask - bid if ask and bid else 0
            implied_odds = entry_price / 100 if entry_price else 0.5
        else:
            ticker = "NO_MARKET_FOUND"
            bid, ask, entry_price, spread = 0, 0, 0, 0
            implied_odds = 0.5
            peg_rationale = ""

        base_confidence = CONFIDENCE_WIND_PENALTY
        if is_midnight:
            base_confidence = max(base_confidence, CONFIDENCE_MIDNIGHT_HIGH)
        if wet_bulb_penalty > 0:
            base_confidence = max(base_confidence, CONFIDENCE_WET_BULB)
        if is_mos_fade:
            base_confidence = max(base_confidence, CONFIDENCE_MOS_FADE)

        edge = base_confidence - implied_odds

        if edge > EDGE_THRESHOLD_BUY and entry_price > 0 and entry_price < MAX_ENTRY_PRICE_CENTS:
            recommendation = "BUY"
            confidence = 8 if edge > 0.30 else 7
        elif is_mos_fade:
            recommendation = "FADE_NWS"
            confidence = 7
        elif edge > 0.10:
            recommendation = "PASS"
            confidence = 5
        else:
            recommendation = "PASS"
            confidence = 3

        rationale_parts = []
        if wind_penalty > 0:
            rationale_parts.append(f"Wind: -{wind_penalty:.1f}F")
        if wet_bulb_penalty > 0:
            rationale_parts.append(f"WetBulb: -{wet_bulb_penalty:.1f}F (Precip {precip_prob}%)")
        if is_midnight:
            rationale_parts.append(f"Midnight: {midnight_temp:.0f}F > Afternoon {afternoon_temp:.0f}F")
        if is_mos_fade:
            rationale_parts.append(f"MOS Fade: NWS {nws_high:.0f}F >> Models {mos_consensus:.0f}F")
        if peg_rationale:
            rationale_parts.append(peg_rationale)
        if not rationale_parts:
            rationale_parts.append("No significant weather signals")

        logger.info(f"Trade ticket: {recommendation} {ticker} @ {entry_price}c, edge={edge:.1%}")

        return TradeTicket(
            nws_forecast_high=nws_high,
            physics_high=physics_high,
            wind_penalty=wind_penalty,
            wet_bulb_penalty=wet_bulb_penalty,
            wind_gust=wind_gust,
            is_midnight_risk=is_midnight,
            midnight_temp=midnight_temp,
            afternoon_temp=afternoon_temp,
            is_wet_bulb_risk=wet_bulb_penalty > 0,
            is_mos_fade=is_mos_fade,
            mav_high=mav_high,
            met_high=met_high,
            mos_consensus=mos_consensus,
            target_bracket_low=bracket_low,
            target_bracket_high=bracket_high,
            target_ticker=ticker,
            current_bid_cents=bid,
            current_ask_cents=ask,
            entry_price_cents=entry_price,
            spread_cents=spread,
            implied_odds=implied_odds,
            estimated_edge=edge,
            recommendation=recommendation,
            confidence=confidence,
            rationale=" | ".join(rationale_parts),
        )

    def print_trade_ticket(self, ticket: TradeTicket):
        """Print formatted trade ticket."""
        print("\n" + "="*60)
        print(f"        SNIPER ANALYSIS v{self.VERSION} ({self.city_code})")
        print("="*60)

        print(f"* NWS Forecast High:  {ticket.nws_forecast_high:.0f}F")
        print(f"* Physics High:       {ticket.physics_high:.1f}F")
        print(f"  - Wind Penalty:     -{ticket.wind_penalty:.1f}F (gusts {ticket.wind_gust:.0f}mph)")
        print(f"  - WetBulb Penalty:  -{ticket.wet_bulb_penalty:.1f}F")

        print("-"*60)
        print(f"* Midnight High:      {'YES' if ticket.is_midnight_risk else 'No'}")
        if ticket.is_midnight_risk:
            print(f"  - Midnight:         {ticket.midnight_temp:.0f}F")
            print(f"  - Afternoon:        {ticket.afternoon_temp:.0f}F")
        print(f"* Wet Bulb Risk:      {'YES' if ticket.is_wet_bulb_risk else 'No'}")

        print("-"*60)
        print(f"* MAV (GFS) High:     {ticket.mav_high:.0f}F" if ticket.mav_high else "* MAV (GFS) High:     N/A")
        print(f"* MET (NAM) High:     {ticket.met_high:.0f}F" if ticket.met_high else "* MET (NAM) High:     N/A")
        print(f"* MOS Consensus:      {ticket.mos_consensus:.0f}F" if ticket.mos_consensus else "* MOS Consensus:      N/A")
        print(f"* MOS Fade Signal:    {'YES - NWS running hot' if ticket.is_mos_fade else 'No'}")

        print("-"*60)
        print(f"TARGET BRACKET:    {ticket.target_bracket_low}F to {ticket.target_bracket_high}F")
        print(f"TICKER:            {ticket.target_ticker}")
        print(f"MARKET:            Bid {ticket.current_bid_cents}c / Ask {ticket.current_ask_cents}c (Spread: {ticket.spread_cents}c)")
        print(f"ENTRY PRICE:       {ticket.entry_price_cents}c (Smart Peg)")
        print(f"IMPLIED ODDS:      {ticket.implied_odds:.0%}")
        print(f"ESTIMATED EDGE:    {'+' if ticket.estimated_edge > 0 else ''}{ticket.estimated_edge:.0%}")
        print(f"CONFIDENCE:        {ticket.confidence}/10")

        print("-"*60)
        print(f"RATIONALE: {ticket.rationale}")
        print("-"*60)
        print(f">>> RECOMMENDATION: {ticket.recommendation} <<<")
        print("="*60)

    # =========================================================================
    # TRADE EXECUTION
    # =========================================================================

    async def execute_trade(self, ticket: TradeTicket) -> bool:
        """Execute trade with human confirmation and smart pegging."""
        if ticket.recommendation not in ("BUY", "FADE_NWS"):
            logger.info("No trade recommended - skipping execution")
            print("\n[SKIP] No trade recommended.")
            return False

        if ticket.entry_price_cents == 0:
            logger.warning("No valid entry price - skipping execution")
            print("\n[SKIP] No valid entry price.")
            return False

        max_cost = self.balance * MAX_POSITION_PCT
        contracts = int(max_cost / (ticket.entry_price_cents / 100))
        total_cost = contracts * ticket.entry_price_cents / 100
        potential_profit = contracts * (100 - ticket.entry_price_cents) / 100

        print(f"\n[TRADE SETUP]")
        print(f"  Contracts:   {contracts}")
        print(f"  Entry Price: {ticket.entry_price_cents}c (Smart Peg)")
        print(f"  Cost:        ${total_cost:.2f}")
        print(f"  Max Profit:  ${potential_profit:.2f}")

        if ticket.spread_cents > MAX_SPREAD_TO_CROSS_CENTS:
            print(f"  [NOTE] Wide spread - order may not fill immediately")

        if not self.live_mode:
            logger.info("Analysis mode - trade not executed")
            print("\n[ANALYSIS MODE] No trade executed. Use --live for real trades.")
            return False

        response = input(f"\nExecute trade? (y/n): ").strip().lower()

        if response != "y":
            logger.info("Trade cancelled by user")
            print("[CANCELLED] Trade not executed.")
            return False

        try:
            result = await self.kalshi.place_order(
                ticker=ticket.target_ticker,
                side="yes",
                action="buy",
                count=contracts,
                price=ticket.entry_price_cents,
                order_type="limit"
            )
            order_id = result.get("order", {}).get("order_id", "N/A")
            logger.info(f"Trade executed: order_id={order_id}")
            print(f"\n[EXECUTED] Order ID: {order_id}")

            async with aiofiles.open(TRADES_LOG_FILE, "a") as f:
                await f.write(json.dumps({
                    "ts": datetime.now(self.tz).isoformat(),
                    "version": self.VERSION,
                    "city": self.city_code,
                    "ticker": ticket.target_ticker,
                    "side": "yes",
                    "contracts": contracts,
                    "price": ticket.entry_price_cents,
                    "nws_high": ticket.nws_forecast_high,
                    "physics_high": ticket.physics_high,
                    "wind_penalty": ticket.wind_penalty,
                    "wet_bulb_penalty": ticket.wet_bulb_penalty,
                    "midnight_risk": ticket.is_midnight_risk,
                    "mos_fade": ticket.is_mos_fade,
                    "mav_high": ticket.mav_high,
                    "met_high": ticket.met_high,
                    "edge": ticket.estimated_edge,
                    "order_id": order_id,
                }) + "\n")

            return True
        except Exception as e:
            logger.exception(f"Trade execution failed: {e}")
            print(f"\n[ERROR] Trade failed: {e}")
            return False

    # =========================================================================
    # PORTFOLIO MANAGEMENT
    # =========================================================================

    def parse_bracket_from_ticker(self, ticker: str) -> tuple[int, int]:
        """Parse bracket range from ticker."""
        match = re.search(r"-([TB])(\d+(?:\.\d+)?)$", ticker)
        if not match:
            return (0, 0)

        prefix, value = match.group(1), float(match.group(2))

        if prefix == "T":
            return (int(value), int(value) + 2)
        else:
            low = int(value)
            return (low, low + 2)

    async def get_avg_entry_from_fills(self, ticker: str) -> tuple[int, float]:
        """Calculate average entry price from fills for a ticker."""
        fills = await self.kalshi.get_fills(limit=FILLS_FETCH_LIMIT)

        total_cost = 0
        total_contracts = 0

        for f in fills:
            if f.get("ticker") != ticker:
                continue

            side = f.get("side")
            action = f.get("action")
            count = f.get("count", 0)
            price = f.get("yes_price") or f.get("no_price") or 0

            if side == "yes" and action == "buy":
                total_contracts += count
                total_cost += count * price
            elif side == "no" and action == "sell":
                total_contracts += count
                total_cost += count * (100 - price)

        avg_entry = total_cost / total_contracts if total_contracts > 0 else 0
        return total_contracts, avg_entry

    async def _generate_exit_signal(
        self,
        ticker: str,
        contracts: int,
        nws_high: float,
    ) -> Optional[ExitSignal]:
        """
        Professional exit signal generation with 3-rule logic:

        Rule A: FREEROLL - ROI > 100% → Sell half, ride remainder for free
        Rule B: CAPITAL EFFICIENCY - Price > 90¢ → Sell all, redeploy capital
        Rule C: THESIS BREAK - Model mismatch → Immediate bailout

        Philosophy: Never risk 90¢ to make 10¢. Capital velocity > absolute ROI.
        """
        if contracts <= 0:
            return None

        _, avg_entry = await self.get_avg_entry_from_fills(ticker)

        orderbook = await self.kalshi.get_orderbook(ticker)
        yes_bids = orderbook.get("yes", [])
        current_bid = yes_bids[0][0] if yes_bids else 0

        if current_bid == 0:
            logger.warning(f"No bids for {ticker}")
            print(f"  [WARN] No bids for {ticker}")
            return None

        roi = ((current_bid - avg_entry) / avg_entry * 100) if avg_entry > 0 else 0
        bracket = self.parse_bracket_from_ticker(ticker)
        thesis_valid = bracket[0] <= nws_high <= bracket[1]

        # =====================================================================
        # RULE C: THESIS BREAK (Highest Priority - "Oh Sh*t" Handle)
        # If weather model says we're wrong, exit immediately.
        # =====================================================================
        if not thesis_valid:
            signal_type = "BAIL_OUT"
            sell_qty = contracts
            rationale = f"THESIS BROKEN: NWS {nws_high:.0f}F outside bracket {bracket[0]}-{bracket[1]}F. Dump at market."

        # =====================================================================
        # RULE B: CAPITAL EFFICIENCY ("90-Cent Curse")
        # Price > 90¢ means risking 90 to make 10. Terrible risk/reward.
        # Sell and redeploy into a 30¢ opportunity that can double.
        # =====================================================================
        elif current_bid >= CAPITAL_EFFICIENCY_THRESHOLD_CENTS:
            signal_type = "EFFICIENCY_EXIT"
            sell_qty = contracts
            risk = current_bid
            reward = 100 - current_bid
            rationale = f"CAPITAL EFFICIENCY: Price {current_bid}¢ (Risk {risk}¢ to make {reward}¢). Redeploy capital."

        # =====================================================================
        # RULE A: FREEROLL (House Money)
        # ROI > 100% means we doubled. Sell half to secure principal.
        # Remaining contracts are "free" - zero emotional attachment.
        # =====================================================================
        elif roi >= TAKE_PROFIT_ROI_PCT:
            signal_type = "FREEROLL"
            sell_qty = max(1, contracts // 2)  # At least 1 contract
            profit_locked = (sell_qty * current_bid) / 100
            rationale = f"FREEROLL: ROI {roi:.0f}%. Sell {sell_qty} (${profit_locked:.2f}), ride remainder for free."

        # =====================================================================
        # HOLD: Trade developing, thesis valid, no exit trigger
        # =====================================================================
        else:
            signal_type = "HOLD"
            sell_qty = 0
            upside = 100 - current_bid
            rationale = f"DEVELOPING: Thesis valid. Price {current_bid}¢, upside {upside}¢. ROI {roi:.0f}%."

        logger.info(f"Exit signal for {ticker}: {signal_type}, ROI={roi:.0f}%, Bid={current_bid}¢")

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

    async def evaluate_position_from_api(self, position: dict, nws_high: float) -> Optional[ExitSignal]:
        """Evaluate a position from the positions API endpoint."""
        ticker = position.get("ticker", "")
        contracts = abs(position.get("position", 0))
        return await self._generate_exit_signal(ticker, contracts, nws_high)

    def print_exit_signal(self, signal: ExitSignal, position_num: int):
        """Print formatted exit signal with risk/reward analysis."""
        print(f"\n[POSITION {position_num}] {signal.ticker}")
        print("-" * 55)
        print(f"  Contracts:     {signal.contracts_held}")
        print(f"  Avg Entry:     {signal.avg_entry_cents}c")
        print(f"  Current Bid:   {signal.current_bid_cents}c")
        print(f"  ROI:           {'+' if signal.roi_percent >= 0 else ''}{signal.roi_percent:.0f}%")
        print(f"  Target:        {signal.target_bracket[0]}-{signal.target_bracket[1]}F")
        print(f"  NWS Forecast:  {signal.nws_forecast_high:.0f}F")
        print(f"  Thesis:        {'VALID' if signal.thesis_valid else 'INVALID'}")

        # Risk/Reward analysis
        risk = signal.current_bid_cents
        reward = 100 - signal.current_bid_cents
        print(f"  Risk/Reward:   {risk}c risk / {reward}c reward")

        print("-" * 55)

        # Color-coded signal type
        signal_map = {
            "BAIL_OUT": "BAIL_OUT (Thesis Broken)",
            "EFFICIENCY_EXIT": "EFFICIENCY_EXIT (90c Curse)",
            "FREEROLL": "FREEROLL (House Money)",
            "HOLD": "HOLD (Developing)",
        }
        print(f">>> SIGNAL: {signal_map.get(signal.signal_type, signal.signal_type)} <<<")
        print(f">>> {signal.rationale}")

        if signal.sell_qty > 0:
            proceeds = signal.sell_qty * signal.sell_price_cents / 100
            pct_selling = (signal.sell_qty / signal.contracts_held * 100) if signal.contracts_held > 0 else 0
            print(f">>> ACTION: SELL {signal.sell_qty} ({pct_selling:.0f}%) @ {signal.sell_price_cents}c = ${proceeds:.2f}")

        print("=" * 55)

    async def execute_exit(self, signal: ExitSignal) -> bool:
        """Execute exit with human confirmation."""
        if signal.sell_qty == 0:
            return False

        proceeds = signal.sell_qty * signal.sell_price_cents / 100

        print(f"\n[EXIT ORDER]")
        print(f"  Ticker:    {signal.ticker}")
        print(f"  Action:    SELL {signal.sell_qty} YES")
        print(f"  Price:     {signal.sell_price_cents}c (at bid)")
        print(f"  Proceeds:  ${proceeds:.2f}")

        if not self.live_mode:
            logger.info("Analysis mode - exit not executed")
            print("\n[ANALYSIS MODE] No trade executed. Use --live for real trades.")
            return False

        response = input(f"\nExecute sell? (y/n): ").strip().lower()

        if response != "y":
            logger.info("Exit cancelled by user")
            print("[CANCELLED] Exit not executed.")
            return False

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
            logger.info(f"Exit executed: order_id={order_id}")
            print(f"\n[EXECUTED] Sell Order ID: {order_id}")

            async with aiofiles.open(TRADES_LOG_FILE, "a") as f:
                await f.write(json.dumps({
                    "ts": datetime.now(self.tz).isoformat(),
                    "city": self.city_code,
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
            logger.exception(f"Exit execution failed: {e}")
            print(f"\n[ERROR] Exit failed: {e}")
            return False

    # =========================================================================
    # MAIN WORKFLOWS
    # =========================================================================

    async def manage_positions(self):
        """Portfolio manager mode - check positions and generate exit signals."""
        await self.start()

        try:
            print(f"\n{'='*60}")
            print(f"  WEATHER SNIPER - PORTFOLIO MANAGER v{self.VERSION}")
            print(f"  City: {self.station_config.city_name}")
            print(f"{'='*60}")

            print("\n[1/3] Fetching NWS forecast...")
            forecasts = await self.nws.get_hourly_forecast()

            now = datetime.now(self.tz)
            today = now.date()

            max_temp_today = 0.0
            for f in forecasts:
                f_local = f.time.astimezone(self.tz)
                if f_local.date() == today and f.temp_f > max_temp_today:
                    max_temp_today = f.temp_f

            current_temp = await self.nws.get_current_temp()
            if current_temp and current_temp > max_temp_today:
                max_temp_today = current_temp

            print(f"  NWS Forecast High (Today): {max_temp_today:.0f}F")
            if current_temp:
                print(f"  Current Temp: {current_temp:.0f}F")

            print("\n[2/3] Fetching positions...")
            positions = await self.kalshi.get_positions()

            active_positions = [
                p for p in positions
                if self.station_config.series_ticker in p.get("ticker", "") and p.get("position", 0) != 0
            ]

            if not active_positions:
                logger.info("No active weather positions")
                print(f"\n[INFO] No active {self.city_code} weather positions.")
                return

            logger.info(f"Found {len(active_positions)} active position(s)")
            print(f"  Found {len(active_positions)} active position(s)")

            print("\n[3/3] Generating exit signals...")

            for i, pos in enumerate(active_positions, 1):
                signal = await self.evaluate_position_from_api(pos, max_temp_today)

                if signal:
                    self.print_exit_signal(signal, i)

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
            print("\n[1/5] Fetching NWS hourly forecast...")
            forecasts = await self.nws.get_hourly_forecast()
            if not forecasts:
                logger.error("No forecast data available")
                print("[ERR] No forecast data available.")
                return

            print("[2/5] Fetching MOS model data...")
            mav_forecast = await self.mos.get_mav()
            met_forecast = await self.mos.get_met()

            mav_high = mav_forecast.max_temp_f if mav_forecast else None
            met_high = met_forecast.max_temp_f if met_forecast else None

            if mav_high:
                print(f"  MAV (GFS MOS): {mav_high:.0f}F")
            else:
                print(f"  MAV (GFS MOS): Unavailable")
            if met_high:
                print(f"  MET (NAM MOS): {met_high:.0f}F")
            else:
                print(f"  MET (NAM MOS): Unavailable")

            print("[3/5] Analyzing weather patterns...")

            peak_forecast = self.get_peak_forecast(forecasts)
            if not peak_forecast:
                logger.error("Could not determine peak forecast")
                print("[ERR] Could not determine peak forecast.")
                return

            is_midnight, midnight_temp, afternoon_temp = self.check_midnight_high(forecasts)

            print(f"  NWS Forecast High: {peak_forecast.temp_f:.0f}F")
            print(f"  Peak Hour Wind:    {peak_forecast.wind_gust_mph:.0f} mph gusts")
            print(f"  Peak Hour Precip:  {peak_forecast.precip_prob}%")
            print(f"  Peak Hour Dewpoint: {peak_forecast.dewpoint_f:.0f}F")
            print(f"  Midnight High:     {'YES' if is_midnight else 'No'}")

            print("[4/5] Fetching Kalshi markets...")
            markets = await self.get_kalshi_markets()

            wind_penalty = self.calculate_wind_penalty(peak_forecast.wind_gust_mph)
            wet_bulb_penalty = self.calculate_wet_bulb_penalty(
                peak_forecast.temp_f, peak_forecast.dewpoint_f, peak_forecast.precip_prob
            )
            physics_high = peak_forecast.temp_f - wind_penalty - wet_bulb_penalty

            if is_midnight and midnight_temp:
                physics_high = midnight_temp

            target_market = self.find_target_market(markets, physics_high)

            print("[5/5] Generating trade ticket...")
            ticket = self.generate_trade_ticket(
                peak_forecast=peak_forecast,
                is_midnight=is_midnight,
                midnight_temp=midnight_temp,
                afternoon_temp=afternoon_temp,
                mav_high=mav_high,
                met_high=met_high,
                market=target_market,
            )

            self.print_trade_ticket(ticket)

            await self.execute_trade(ticket)

        finally:
            await self.stop()


# Backward compatibility alias
NYCSniper = WeatherSniper


async def main():
    available_cities = ", ".join(STATIONS.keys())

    parser = argparse.ArgumentParser(
        description=f"Weather Sniper v3.0 - Multi-City Predictive Weather Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python3 sniper.py                    # NYC (default), analysis only
  python3 sniper.py --city CHI         # Chicago, analysis only
  python3 sniper.py --city NYC --live  # NYC with live trading
  python3 sniper.py --manage           # Check positions for exit signals

Available cities: {available_cities}
        """
    )
    parser.add_argument(
        "--city",
        type=str,
        default=DEFAULT_CITY,
        help=f"City code to analyze (default: {DEFAULT_CITY}). Available: {available_cities}"
    )
    parser.add_argument("--live", action="store_true", help="Enable live trading (requires confirmation)")
    parser.add_argument("--manage", action="store_true", help="Portfolio manager mode - check positions for exit signals")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate city code
    try:
        get_station_config(args.city)
    except KeyError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    bot = WeatherSniper(city_code=args.city, live_mode=args.live)

    if args.manage:
        await bot.manage_positions()
    else:
        await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
