#!/usr/bin/env python3
"""
WEATHER SNIPER - Shared Strategy Module

Consolidated strategy calculations for all weather trading bots.
Eliminates duplication across sniper.py, nyc_sniper_complete.py, etc.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from zoneinfo import ZoneInfo

from config import (
    # Wind Penalty (Strategy B)
    WIND_PENALTY_LIGHT_THRESHOLD_MPH,
    WIND_PENALTY_HEAVY_THRESHOLD_MPH,
    WIND_PENALTY_LIGHT_DEGREES,
    WIND_PENALTY_HEAVY_DEGREES,
    # Midnight High (Strategy A)
    MIDNIGHT_HOUR_START,
    MIDNIGHT_HOUR_END,
    AFTERNOON_HOUR_START,
    AFTERNOON_HOUR_END,
    # Wet Bulb (Strategy D)
    WET_BULB_PRECIP_THRESHOLD_PCT,
    WET_BULB_DEPRESSION_MIN_F,
    WET_BULB_FACTOR_LIGHT,
    WET_BULB_FACTOR_HEAVY,
    WET_BULB_HEAVY_PRECIP_THRESHOLD,
    # MOS Fade (Strategy E)
    MOS_DIVERGENCE_THRESHOLD_F,
    # Confidence levels
    CONFIDENCE_MIDNIGHT_HIGH,
    CONFIDENCE_WIND_PENALTY,
    CONFIDENCE_WET_BULB,
    CONFIDENCE_MOS_FADE,
    # Trading
    EDGE_THRESHOLD_BUY,
    MAX_ENTRY_PRICE_CENTS,
    MAX_SPREAD_TO_CROSS_CENTS,
    PEG_OFFSET_CENTS,
    MIN_BID_CENTS,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

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


# =============================================================================
# STRATEGY A: MIDNIGHT HIGH DETECTION
# =============================================================================

def check_midnight_high(
    forecasts: list[HourlyForecast],
    tz: ZoneInfo
) -> tuple[bool, Optional[float], Optional[float]]:
    """
    Strategy A: Midnight High Detection.

    During post-frontal cold advection, the daily high temperature is often
    set at 12:01 AM before the cold air settles, not in the afternoon.

    Returns: (is_midnight_high, midnight_temp, afternoon_temp)
    """
    now = datetime.now(tz)
    tomorrow = now.date() + timedelta(days=1)

    midnight_temp = None
    afternoon_temp = None

    for f in forecasts:
        f_local = f.time.astimezone(tz)
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


# =============================================================================
# STRATEGY B: WIND MIXING PENALTY
# =============================================================================

def calculate_wind_penalty(wind_gust_mph: float) -> float:
    """
    Strategy B: Wind Mixing Penalty.

    Mechanical mixing from strong winds prevents the "super-adiabatic"
    surface heating layer that allows temperature maximization.

    Returns: Temperature penalty in degrees F
    """
    if wind_gust_mph > WIND_PENALTY_HEAVY_THRESHOLD_MPH:
        return WIND_PENALTY_HEAVY_DEGREES
    elif wind_gust_mph > WIND_PENALTY_LIGHT_THRESHOLD_MPH:
        return WIND_PENALTY_LIGHT_DEGREES
    return 0.0


# =============================================================================
# STRATEGY C: ROUNDING ARBITRAGE
# =============================================================================

def temp_to_bracket(temp_f: float) -> tuple[int, int]:
    """
    Strategy C: Rounding Arbitrage.

    NWS rounds to nearest whole degree:
    - x.50 and above -> rounds UP
    - x.49 and below -> rounds DOWN

    Returns: (bracket_low, bracket_high)
    """
    rounded = int(Decimal(str(temp_f)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    low = (rounded // 2) * 2 - 1
    high = low + 2
    return low, high


# =============================================================================
# STRATEGY D: WET BULB / EVAPORATIVE COOLING
# =============================================================================

def calculate_wet_bulb_penalty(
    temp_f: float,
    dewpoint_f: float,
    precip_prob: int
) -> float:
    """
    Strategy D: Wet Bulb / Evaporative Cooling Risk.

    Rain falling into unsaturated air evaporates, removing latent heat
    and causing cooling. Larger T-Td spread = more potential cooling.

    Returns: Temperature penalty in degrees F
    """
    if precip_prob < WET_BULB_PRECIP_THRESHOLD_PCT:
        return 0.0

    depression = temp_f - dewpoint_f

    if depression < WET_BULB_DEPRESSION_MIN_F:
        return 0.0

    factor = (
        WET_BULB_FACTOR_HEAVY
        if precip_prob >= WET_BULB_HEAVY_PRECIP_THRESHOLD
        else WET_BULB_FACTOR_LIGHT
    )

    penalty = depression * factor
    return round(penalty, 1)


# =============================================================================
# STRATEGY E: MOS CONSENSUS FADE
# =============================================================================

def check_mos_divergence(
    nws_high: float,
    mav_high: Optional[float],
    met_high: Optional[float]
) -> tuple[bool, Optional[float]]:
    """
    Strategy E: Check if NWS diverges from MOS consensus.

    If the official NWS forecast is significantly hotter than the model
    consensus (GFS MAV + NAM MET), fade the NWS forecast.

    Returns: (is_mos_fade, mos_consensus)
    """
    mos_values = [v for v in [mav_high, met_high] if v is not None]
    if not mos_values:
        return False, None

    mos_consensus = sum(mos_values) / len(mos_values)

    if nws_high > mos_consensus + MOS_DIVERGENCE_THRESHOLD_F:
        return True, mos_consensus

    return False, mos_consensus


# =============================================================================
# MARKET OPERATIONS
# =============================================================================

def calculate_smart_entry_price(bid: int, ask: int) -> tuple[int, str]:
    """
    Smart Pegging: Calculate optimal entry price.

    - Tight spread (<= 5c): Cross the spread, take the ask
    - Wide spread (> 5c): Peg bid+1 to avoid market order slippage

    Returns: (entry_price, rationale)
    """
    spread = ask - bid

    if bid < MIN_BID_CENTS:
        return 0, "No valid bid"

    if spread <= MAX_SPREAD_TO_CROSS_CENTS:
        return ask, f"Tight spread ({spread}c) - taking ask"
    else:
        entry = bid + PEG_OFFSET_CENTS
        return entry, f"Wide spread ({spread}c) - pegging bid+{PEG_OFFSET_CENTS}"


# =============================================================================
# PEAK FORECAST DETECTION
# =============================================================================

def get_peak_forecast(
    forecasts: list[HourlyForecast],
    tz: ZoneInfo
) -> Optional[HourlyForecast]:
    """Get the forecast period with the highest temperature for tomorrow."""
    now = datetime.now(tz)
    tomorrow = now.date() + timedelta(days=1)

    tomorrow_forecasts = [
        f for f in forecasts
        if f.time.astimezone(tz).date() == tomorrow
    ]

    if not tomorrow_forecasts:
        return None

    return max(tomorrow_forecasts, key=lambda x: x.temp_f)


# =============================================================================
# TRADE TICKET GENERATION
# =============================================================================

def generate_trade_ticket(
    peak_forecast: HourlyForecast,
    is_midnight: bool,
    midnight_temp: Optional[float],
    afternoon_temp: Optional[float],
    mav_high: Optional[float],
    met_high: Optional[float],
    market: Optional[dict],
) -> TradeTicket:
    """
    Generate a comprehensive trade ticket with all analysis.

    Combines all strategy signals into a single recommendation.
    """
    nws_high = peak_forecast.temp_f
    wind_gust = peak_forecast.wind_gust_mph
    dewpoint = peak_forecast.dewpoint_f
    precip_prob = peak_forecast.precip_prob

    # Apply strategy calculations
    wind_penalty = calculate_wind_penalty(wind_gust)
    wet_bulb_penalty = calculate_wet_bulb_penalty(nws_high, dewpoint, precip_prob)
    is_mos_fade, mos_consensus = check_mos_divergence(nws_high, mav_high, met_high)

    # Calculate physics high
    physics_high = nws_high - wind_penalty - wet_bulb_penalty

    # Override with midnight temp if applicable
    if is_midnight and midnight_temp:
        physics_high = midnight_temp

    # Cap at MOS consensus if fading NWS
    if is_mos_fade and mos_consensus:
        physics_high = min(physics_high, mos_consensus)

    # Get target bracket
    bracket_low, bracket_high = temp_to_bracket(physics_high)

    # Extract market data
    if market:
        ticker = market.get("ticker", "")
        bid = market.get("yes_bid", 0)
        ask = market.get("yes_ask", 0)
        entry_price, peg_rationale = calculate_smart_entry_price(bid, ask)
        spread = ask - bid if ask and bid else 0
        implied_odds = entry_price / 100 if entry_price else 0.5
    else:
        ticker = "NO_MARKET_FOUND"
        bid, ask, entry_price, spread = 0, 0, 0, 0
        implied_odds = 0.5
        peg_rationale = ""

    # Calculate confidence
    base_confidence = CONFIDENCE_WIND_PENALTY
    if is_midnight:
        base_confidence = max(base_confidence, CONFIDENCE_MIDNIGHT_HIGH)
    if wet_bulb_penalty > 0:
        base_confidence = max(base_confidence, CONFIDENCE_WET_BULB)
    if is_mos_fade:
        base_confidence = max(base_confidence, CONFIDENCE_MOS_FADE)

    # Calculate edge
    edge = base_confidence - implied_odds

    # Determine recommendation
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

    # Build rationale
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
