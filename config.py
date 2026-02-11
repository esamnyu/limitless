#!/usr/bin/env python3
"""
WEATHER EDGE v4.0 - Configuration Constants

Centralized configuration for all trading parameters, thresholds, and settings.
Single source of truth for city/station configs used by all modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# =============================================================================
# CITY STATION CONFIGURATION
# =============================================================================

@dataclass
class StationConfig:
    """Configuration for a single weather station/city."""
    city_code: str              # Short code (NYC, CHI)
    city_name: str              # Full name for display
    station_id: str             # NWS station ID (KNYC, KMDW)
    series_ticker: str          # Kalshi market series (KXHIGHNY, KXHIGHCHI)
    lat: float                  # Latitude for Open-Meteo ensemble API
    lon: float                  # Longitude for Open-Meteo ensemble API
    nws_station_url: str        # NWS station metadata URL
    nws_observation_url: str    # NWS current observation URL
    nws_hourly_forecast_url: str  # NWS hourly forecast URL
    nws_gridpoint: str          # Gridpoint identifier for fallback
    mos_mav_url: str            # GFS MOS (MAV) URL
    mos_met_url: str            # NAM MOS (MET) URL
    timezone: str               # IANA timezone
    dsm_times_z: List[str] = field(default_factory=list)   # DSM release times (Zulu)
    six_hour_z: List[str] = field(default_factory=lambda: ["23:51", "05:51", "11:51", "17:51"])


def _nws_urls(station_id: str) -> tuple:
    """Generate NWS station and observation URLs from a station ID."""
    base = "https://api.weather.gov/stations"
    return (f"{base}/{station_id}", f"{base}/{station_id}/observations/latest")


def _mos_urls(station_id: str) -> tuple:
    """Generate MOS MAV/MET URLs from a station ID."""
    sid = station_id.lower()
    base = "https://tgftp.nws.noaa.gov/data/forecasts/mos"
    return (f"{base}/gfs/short/mav/{sid}.txt", f"{base}/nam/short/met/{sid}.txt")


# Station configurations for all 5 supported cities
STATIONS: Dict[str, StationConfig] = {
    "NYC": StationConfig(
        city_code="NYC",
        city_name="New York (Central Park)",
        station_id="KNYC",
        series_ticker="KXHIGHNY",
        lat=40.78, lon=-73.97,
        nws_station_url=_nws_urls("KNYC")[0],
        nws_observation_url=_nws_urls("KNYC")[1],
        nws_hourly_forecast_url="https://api.weather.gov/gridpoints/OKX/33,37/forecast/hourly",
        nws_gridpoint="OKX/33,37",
        mos_mav_url=_mos_urls("KNYC")[0],
        mos_met_url=_mos_urls("KNYC")[1],
        timezone="America/New_York",
        dsm_times_z=["20:21", "21:21", "05:17"],
    ),
    "CHI": StationConfig(
        city_code="CHI",
        city_name="Chicago (Midway)",
        station_id="KMDW",
        series_ticker="KXHIGHCHI",
        lat=41.79, lon=-87.75,
        nws_station_url=_nws_urls("KMDW")[0],
        nws_observation_url=_nws_urls("KMDW")[1],
        nws_hourly_forecast_url="https://api.weather.gov/gridpoints/LOT/75,72/forecast/hourly",
        nws_gridpoint="LOT/75,72",
        mos_mav_url=_mos_urls("KMDW")[0],
        mos_met_url=_mos_urls("KMDW")[1],
        timezone="America/Chicago",
        dsm_times_z=["21:00", "22:00", "06:00"],
    ),
    "DEN": StationConfig(
        city_code="DEN",
        city_name="Denver (DIA)",
        station_id="KDEN",
        series_ticker="KXHIGHDEN",
        lat=39.86, lon=-104.67,
        nws_station_url=_nws_urls("KDEN")[0],
        nws_observation_url=_nws_urls("KDEN")[1],
        nws_hourly_forecast_url="https://api.weather.gov/gridpoints/BOU/63,62/forecast/hourly",
        nws_gridpoint="BOU/63,62",
        mos_mav_url=_mos_urls("KDEN")[0],
        mos_met_url=_mos_urls("KDEN")[1],
        timezone="America/Denver",
        dsm_times_z=["22:00", "23:00", "07:00"],
    ),
    "MIA": StationConfig(
        city_code="MIA",
        city_name="Miami (MIA Airport)",
        station_id="KMIA",
        series_ticker="KXHIGHMIA",
        lat=25.79, lon=-80.29,
        nws_station_url=_nws_urls("KMIA")[0],
        nws_observation_url=_nws_urls("KMIA")[1],
        nws_hourly_forecast_url="https://api.weather.gov/gridpoints/MFL/76,50/forecast/hourly",
        nws_gridpoint="MFL/76,50",
        mos_mav_url=_mos_urls("KMIA")[0],
        mos_met_url=_mos_urls("KMIA")[1],
        timezone="America/New_York",
        dsm_times_z=["20:30", "21:30", "05:30"],
    ),
    "LAX": StationConfig(
        city_code="LAX",
        city_name="Los Angeles (LAX)",
        station_id="KLAX",
        series_ticker="KXHIGHLAX",
        lat=33.94, lon=-118.41,
        nws_station_url=_nws_urls("KLAX")[0],
        nws_observation_url=_nws_urls("KLAX")[1],
        nws_hourly_forecast_url="https://api.weather.gov/gridpoints/LOX/150,44/forecast/hourly",
        nws_gridpoint="LOX/150,44",
        mos_mav_url=_mos_urls("KLAX")[0],
        mos_met_url=_mos_urls("KLAX")[1],
        timezone="America/Los_Angeles",
        dsm_times_z=["23:00", "00:00", "08:00"],
    ),
}

# Default city if none specified
DEFAULT_CITY = "NYC"


def get_station_config(city_code: str) -> StationConfig:
    """Get station configuration for a city code. Raises KeyError if not found."""
    city_upper = city_code.upper()
    if city_upper not in STATIONS:
        available = ", ".join(STATIONS.keys())
        raise KeyError(f"Unknown city code: {city_code}. Available: {available}")
    return STATIONS[city_upper]


# =============================================================================
# LEGACY CONSTANTS (for backward compatibility)
# These point to NYC by default. New code should use STATIONS dict.
# =============================================================================

_default_station = STATIONS[DEFAULT_CITY]

# NWS APIs (NYC defaults)
NWS_STATION_URL = _default_station.nws_station_url
NWS_OBSERVATION_URL = _default_station.nws_observation_url
NWS_HOURLY_FORECAST_URL = _default_station.nws_hourly_forecast_url
NWS_GRIDPOINT_FALLBACK = _default_station.nws_gridpoint

# MOS URLs (NYC defaults)
MOS_MAV_URL = _default_station.mos_mav_url
MOS_MET_URL = _default_station.mos_met_url

# Market identifier (NYC default)
NYC_HIGH_SERIES_TICKER = _default_station.series_ticker

# =============================================================================
# KALSHI API ENDPOINTS
# =============================================================================

KALSHI_LIVE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"

# =============================================================================
# TRADING PARAMETERS (v2 — tighter risk controls)
# =============================================================================

# Maximum position size as percentage of Net Liquidation Value
MAX_POSITION_PCT = 0.10  # 10% per trade (conservative)

# Maximum daily exposure across all positions
MAX_DAILY_EXPOSURE = 0.25  # 25% of NLV

# Maximum correlated exposure (similar weather pattern cities)
MAX_CORRELATED_EXPOSURE = 0.15  # 15% across correlated cities

# Edge thresholds for trade recommendations
MIN_EDGE_THRESHOLD = 0.15  # 15% minimum edge after fees
MIN_KDE_PROBABILITY = 0.20  # 20% minimum model probability

# Confidence gate — ONLY trade at this level or above
MIN_CONFIDENCE_TO_TRADE = 90  # 90/100 confidence score required

# Maximum price to consider for entry (never buy YES above this)
# Above 50¢ on YES, risk/reward is worse than 1:1 on weather markets
MAX_ENTRY_PRICE_CENTS = 50

# ROI threshold for taking profit (sell half = freeroll)
FREEROLL_MULTIPLIER = 2.0  # Sell half when price doubles (100% ROI)

# Capital Efficiency threshold - sell when price exceeds this
# Above 90c, you risk 90 to make 10. Terrible risk/reward on weather.
CAPITAL_EFFICIENCY_THRESHOLD_CENTS = 90

# Trailing profit lock — after freeroll, protect gains
TRAILING_OFFSET_CENTS = 8  # Sell if price drops 8¢ from peak

# Near-settlement override — hold for $1 if price > threshold and near settlement
SETTLEMENT_HOLD_THRESHOLD_CENTS = 80
SETTLEMENT_HOUR_ET = 7  # Markets settle ~7 AM ET
SETTLEMENT_WINDOW_HOURS = 2

# =============================================================================
# SMART PEGGING (Order Execution)
# =============================================================================

# Maximum spread to cross (if spread > this, peg Bid+1 instead of hitting Ask)
MAX_SPREAD_TO_CROSS_CENTS = 5

# When pegging, add this to the bid
PEG_OFFSET_CENTS = 1

# Minimum acceptable bid (don't place orders if bid is 0)
MIN_BID_CENTS = 1

# =============================================================================
# WEATHER STRATEGY PARAMETERS
# =============================================================================

# Strategy A: Midnight High detection hours
MIDNIGHT_HOUR_START = 0   # 12:00 AM
MIDNIGHT_HOUR_END = 1     # 1:00 AM
AFTERNOON_HOUR_START = 14 # 2:00 PM
AFTERNOON_HOUR_END = 16   # 4:00 PM

# Strategy B: Wind Mixing Penalty thresholds
WIND_PENALTY_LIGHT_THRESHOLD_MPH = 15   # Gusts > 15mph = -1.0F penalty
WIND_PENALTY_HEAVY_THRESHOLD_MPH = 25   # Gusts > 25mph = -2.0F penalty
WIND_PENALTY_LIGHT_DEGREES = 1.0
WIND_PENALTY_HEAVY_DEGREES = 2.0

# Gust estimation multiplier (when gusts not provided)
WIND_GUST_MULTIPLIER = 1.5
WIND_GUST_THRESHOLD_MPH = 10  # Only apply multiplier above this speed

# Strategy C: Rounding Arbitrage (implicit in temp_to_bracket)

# Strategy D: Wet Bulb / Evaporative Cooling
WET_BULB_PRECIP_THRESHOLD_PCT = 40      # Minimum precip probability to trigger
WET_BULB_DEPRESSION_MIN_F = 5           # Minimum temp-dewpoint spread to consider
WET_BULB_FACTOR_LIGHT = 0.25            # Cooling factor when precip 40-70%
WET_BULB_FACTOR_HEAVY = 0.40            # Cooling factor when precip >= 70%
WET_BULB_HEAVY_PRECIP_THRESHOLD = 70    # Precip % threshold for heavy factor

# Strategy E: MOS Consensus (Model vs Official)
MOS_DIVERGENCE_THRESHOLD_F = 2.0  # If NWS > MOS consensus by this much, fade NWS

# Confidence levels for strategies
CONFIDENCE_MIDNIGHT_HIGH = 0.80  # 80% confidence for midnight high
CONFIDENCE_WIND_PENALTY = 0.70   # 70% confidence for wind penalty
CONFIDENCE_WET_BULB = 0.75       # 75% confidence for wet bulb
CONFIDENCE_MOS_FADE = 0.85       # 85% confidence when fading NWS vs MOS

# =============================================================================
# API RATE LIMITING & RETRY
# =============================================================================

# Minimum seconds between API requests
API_MIN_REQUEST_INTERVAL = 0.1  # 10 requests/sec max

# Retry configuration
API_RETRY_ATTEMPTS = 3
API_RETRY_MIN_WAIT_SEC = 1
API_RETRY_MAX_WAIT_SEC = 10
API_RETRY_MULTIPLIER = 2  # Exponential backoff multiplier

# HTTP timeouts
HTTP_TIMEOUT_TOTAL_SEC = 10
HTTP_TIMEOUT_CONNECT_SEC = 2
NWS_TIMEOUT_TOTAL_SEC = 15
NWS_TIMEOUT_CONNECT_SEC = 5

# Connection pool settings
CONNECTION_POOL_LIMIT = 10
DNS_CACHE_TTL_SEC = 300
KEEPALIVE_TIMEOUT_SEC = 120

# =============================================================================
# FORECAST SETTINGS
# =============================================================================

# Number of hourly forecast periods to fetch
FORECAST_HOURS_AHEAD = 48

# Number of recent fills to fetch for position analysis
FILLS_FETCH_LIMIT = 200

# Orderbook depth for price queries
ORDERBOOK_DEPTH = 10

# =============================================================================
# FILE PATHS
# =============================================================================

TRADES_LOG_FILE = Path("sniper_trades.jsonl")

# =============================================================================
# LOGGING
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"

# =============================================================================
# MIDNIGHT SCANNER CONFIGURATION
# =============================================================================

# Scan schedule (24-hour format, ET timezone)
# 23:00 = 11:00 PM, 23:30 = 11:30 PM, 23:55 = 11:55 PM, 00:05 = 12:05 AM
SCAN_TIMES_ET = ["23:00", "23:30", "23:55", "00:05"]

# Minimum edge to trigger a Discord alert
SCANNER_ALERT_EDGE_THRESHOLD = 0.40  # 40%

# Recommendations that trigger alerts
SCANNER_ALERT_RECOMMENDATIONS = ["BUY", "FADE_NWS"]

# Rate limiting for alerts (minutes between alerts for same ticker)
SCANNER_ALERT_COOLDOWN_MINUTES = 30

# =============================================================================
# DISCORD NOTIFICATIONS
# =============================================================================

# Set DISCORD_WEBHOOK_URL in your .env file:
# DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
#
# To create a webhook:
# 1. Go to your Discord server settings
# 2. Click "Integrations" -> "Webhooks"
# 3. Click "New Webhook", name it "NYC Sniper", copy the URL

# =============================================================================
# LLM ENSEMBLE CONFIDENCE (OpenRouter)
# =============================================================================

# Set OPENROUTER_API_KEY in .env to enable
LLM_CONFIDENCE_ENABLED = False  # Disabled by default; enable after validation
LLM_CONFIDENCE_WEIGHT = 0.15   # 15% of final blended score (statistical = 85%)
LLM_TIMEOUT_SECONDS = 10       # Per-model timeout
LLM_MIN_MODELS_REQUIRED = 2    # Minimum models needed for valid consensus

# Models queried in parallel via OpenRouter
LLM_MODELS = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "deepseek/deepseek-chat",
]
