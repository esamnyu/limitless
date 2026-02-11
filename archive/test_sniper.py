#!/usr/bin/env python3
"""
NYC SNIPER - Test Suite

Tests for weather strategy calculations, bracket parsing, and trade ticket generation.
Run with: pytest tests/ -v
"""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sniper import (
    NYCSniper,
    HourlyForecast,
    TradeTicket,
    ExitSignal,
    MOSForecast,
    MOSClient,
    validate_credentials,
    ConfigurationError,
)
from config import (
    WIND_PENALTY_LIGHT_THRESHOLD_MPH,
    WIND_PENALTY_HEAVY_THRESHOLD_MPH,
    WIND_PENALTY_LIGHT_DEGREES,
    WIND_PENALTY_HEAVY_DEGREES,
    TAKE_PROFIT_ROI_PCT,
    # V2: Wet Bulb
    WET_BULB_PRECIP_THRESHOLD_PCT,
    WET_BULB_DEPRESSION_MIN_F,
    WET_BULB_FACTOR_LIGHT,
    WET_BULB_FACTOR_HEAVY,
    WET_BULB_HEAVY_PRECIP_THRESHOLD,
    # V2: MOS
    MOS_DIVERGENCE_THRESHOLD_F,
    # V2: Smart Pegging
    MAX_SPREAD_TO_CROSS_CENTS,
    PEG_OFFSET_CENTS,
    MIN_BID_CENTS,
)

ET = ZoneInfo("America/New_York")


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sniper():
    """Create a NYCSniper instance for testing."""
    return NYCSniper(live_mode=False)


@pytest.fixture
def sample_forecasts():
    """Create sample forecast data for testing."""
    now = datetime.now(ET)
    tomorrow = now.date() + timedelta(days=1)

    forecasts = []
    for hour in range(24):
        time = datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour, 0, 0, tzinfo=ET)

        # Simulate typical temperature pattern
        if 0 <= hour <= 6:
            temp = 35.0  # Cold early morning
        elif 7 <= hour <= 11:
            temp = 35.0 + (hour - 6) * 2  # Rising
        elif 12 <= hour <= 15:
            temp = 45.0  # Peak afternoon
        else:
            temp = 45.0 - (hour - 15) * 1.5  # Declining

        forecasts.append(HourlyForecast(
            time=time,
            temp_f=temp,
            wind_speed_mph=10.0,
            wind_gust_mph=15.0,
            short_forecast="Partly Cloudy",
            is_daytime=6 <= hour <= 18,
        ))

    return forecasts


@pytest.fixture
def midnight_high_forecasts():
    """Create forecast data with midnight high scenario."""
    now = datetime.now(ET)
    tomorrow = now.date() + timedelta(days=1)

    forecasts = []
    for hour in range(24):
        time = datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour, 0, 0, tzinfo=ET)

        # Midnight high: temp drops throughout day (cold front)
        if 0 <= hour <= 1:
            temp = 50.0  # High at midnight
        elif 2 <= hour <= 8:
            temp = 45.0
        else:
            temp = 35.0  # Afternoon is colder

        forecasts.append(HourlyForecast(
            time=time,
            temp_f=temp,
            wind_speed_mph=5.0,
            wind_gust_mph=8.0,
            short_forecast="Cloudy",
            is_daytime=6 <= hour <= 18,
        ))

    return forecasts


# =============================================================================
# WIND PENALTY TESTS
# =============================================================================

class TestWindPenalty:
    """Tests for Strategy B: Wind Mixing Penalty."""

    def test_no_penalty_calm_wind(self, sniper):
        """No penalty when wind is below threshold."""
        penalty = sniper.calculate_wind_penalty(10.0)
        assert penalty == 0.0

    def test_light_penalty(self, sniper):
        """Light penalty when gusts > 15mph."""
        penalty = sniper.calculate_wind_penalty(20.0)
        assert penalty == WIND_PENALTY_LIGHT_DEGREES

    def test_heavy_penalty(self, sniper):
        """Heavy penalty when gusts > 25mph."""
        penalty = sniper.calculate_wind_penalty(30.0)
        assert penalty == WIND_PENALTY_HEAVY_DEGREES

    def test_boundary_light_threshold(self, sniper):
        """Test boundary at light threshold."""
        # At threshold - no penalty
        assert sniper.calculate_wind_penalty(WIND_PENALTY_LIGHT_THRESHOLD_MPH) == 0.0
        # Just above - light penalty
        assert sniper.calculate_wind_penalty(WIND_PENALTY_LIGHT_THRESHOLD_MPH + 0.1) == WIND_PENALTY_LIGHT_DEGREES

    def test_boundary_heavy_threshold(self, sniper):
        """Test boundary at heavy threshold."""
        # At heavy threshold - still light penalty
        assert sniper.calculate_wind_penalty(WIND_PENALTY_HEAVY_THRESHOLD_MPH) == WIND_PENALTY_LIGHT_DEGREES
        # Just above - heavy penalty
        assert sniper.calculate_wind_penalty(WIND_PENALTY_HEAVY_THRESHOLD_MPH + 0.1) == WIND_PENALTY_HEAVY_DEGREES


# =============================================================================
# MIDNIGHT HIGH TESTS
# =============================================================================

class TestMidnightHigh:
    """Tests for Strategy A: Midnight High Detection."""

    def test_normal_day_no_midnight_high(self, sniper, sample_forecasts):
        """Normal day should not trigger midnight high."""
        is_midnight, midnight_temp, afternoon_temp = sniper.check_midnight_high(sample_forecasts)
        assert is_midnight is False
        assert afternoon_temp > midnight_temp if midnight_temp and afternoon_temp else True

    def test_midnight_high_detection(self, sniper, midnight_high_forecasts):
        """Should detect midnight high when midnight > afternoon."""
        is_midnight, midnight_temp, afternoon_temp = sniper.check_midnight_high(midnight_high_forecasts)
        assert is_midnight is True
        assert midnight_temp == 50.0
        assert afternoon_temp == 35.0

    def test_empty_forecasts(self, sniper):
        """Empty forecast list should return no midnight high."""
        is_midnight, midnight_temp, afternoon_temp = sniper.check_midnight_high([])
        assert is_midnight is False
        assert midnight_temp is None
        assert afternoon_temp is None


# =============================================================================
# BRACKET PARSING TESTS
# =============================================================================

class TestBracketParsing:
    """Tests for ticker bracket parsing."""

    def test_parse_between_bracket(self, sniper):
        """Parse B-style bracket (between)."""
        bracket = sniper.parse_bracket_from_ticker("KXHIGHNY-26JAN17-B33.5")
        assert bracket == (33, 35)

    def test_parse_threshold_bracket(self, sniper):
        """Parse T-style bracket (threshold)."""
        bracket = sniper.parse_bracket_from_ticker("KXHIGHNY-26JAN17-T40")
        assert bracket == (40, 42)

    def test_parse_invalid_ticker(self, sniper):
        """Invalid ticker returns (0, 0)."""
        bracket = sniper.parse_bracket_from_ticker("INVALID-TICKER")
        assert bracket == (0, 0)

    def test_parse_decimal_bracket(self, sniper):
        """Parse bracket with decimal value."""
        bracket = sniper.parse_bracket_from_ticker("KXHIGHNY-26JAN17-B38.5")
        assert bracket == (38, 40)


# =============================================================================
# TEMPERATURE TO BRACKET TESTS
# =============================================================================

class TestTempToBracket:
    """Tests for temperature to bracket conversion."""

    def test_exact_temperature(self, sniper):
        """Exact integer temperature."""
        low, high = sniper.temp_to_bracket(34.0)
        assert isinstance(low, int)
        assert isinstance(high, int)
        assert high == low + 2

    def test_round_up(self, sniper):
        """Temperature .5 rounds up."""
        low1, _ = sniper.temp_to_bracket(34.5)
        low2, _ = sniper.temp_to_bracket(35.0)
        assert low1 == low2  # 34.5 rounds to 35

    def test_round_down(self, sniper):
        """Temperature .49 rounds down."""
        low1, _ = sniper.temp_to_bracket(34.49)
        low2, _ = sniper.temp_to_bracket(34.0)
        assert low1 == low2  # 34.49 rounds to 34


# =============================================================================
# FORECAST HIGH TESTS
# =============================================================================

class TestForecastHigh:
    """Tests for NWS forecast high extraction (V2: get_peak_forecast)."""

    def test_extract_peak_forecast(self, sniper, sample_forecasts):
        """Should find maximum temperature for tomorrow."""
        peak = sniper.get_peak_forecast(sample_forecasts)
        assert peak is not None
        assert peak.temp_f == 45.0  # Peak afternoon temp
        assert peak.wind_gust_mph == 15.0

    def test_empty_forecasts_returns_none(self, sniper):
        """Empty forecast list returns None."""
        peak = sniper.get_peak_forecast([])
        assert peak is None


# =============================================================================
# TRADE TICKET GENERATION TESTS
# =============================================================================

class TestTradeTicket:
    """Tests for trade ticket generation (V2 API)."""

    @pytest.fixture
    def peak_forecast_base(self):
        """Base forecast for testing."""
        return HourlyForecast(
            time=datetime.now(ET),
            temp_f=40.0,
            wind_speed_mph=5.0,
            wind_gust_mph=5.0,
            short_forecast="Clear",
            is_daytime=True,
            precip_prob=0,
            dewpoint_f=30.0,
        )

    def test_generate_pass_no_edge(self, sniper, peak_forecast_base):
        """Should recommend PASS when no edge exists."""
        ticket = sniper.generate_trade_ticket(
            peak_forecast=peak_forecast_base,
            is_midnight=False,
            midnight_temp=None,
            afternoon_temp=None,
            mav_high=None,
            met_high=None,
            market={"ticker": "TEST", "yes_bid": 70, "yes_ask": 72},
        )
        assert ticket.recommendation == "PASS"
        assert ticket.wind_penalty == 0.0

    def test_generate_with_wind_penalty(self, sniper):
        """Should apply wind penalty to physics high."""
        forecast = HourlyForecast(
            time=datetime.now(ET),
            temp_f=40.0,
            wind_speed_mph=15.0,
            wind_gust_mph=20.0,  # Light penalty
            short_forecast="Clear",
            is_daytime=True,
            precip_prob=0,
            dewpoint_f=30.0,
        )
        ticket = sniper.generate_trade_ticket(
            peak_forecast=forecast,
            is_midnight=False,
            midnight_temp=None,
            afternoon_temp=None,
            mav_high=None,
            met_high=None,
            market=None,
        )
        assert ticket.wind_penalty == WIND_PENALTY_LIGHT_DEGREES
        assert ticket.physics_high == 40.0 - WIND_PENALTY_LIGHT_DEGREES

    def test_generate_midnight_high_override(self, sniper):
        """Midnight high should override NWS forecast."""
        forecast = HourlyForecast(
            time=datetime.now(ET),
            temp_f=35.0,
            wind_speed_mph=5.0,
            wind_gust_mph=5.0,
            short_forecast="Cloudy",
            is_daytime=True,
            precip_prob=0,
            dewpoint_f=30.0,
        )
        ticket = sniper.generate_trade_ticket(
            peak_forecast=forecast,
            is_midnight=True,
            midnight_temp=45.0,
            afternoon_temp=35.0,
            mav_high=None,
            met_high=None,
            market=None,
        )
        assert ticket.physics_high == 45.0  # Midnight temp, not NWS
        assert ticket.is_midnight_risk is True

    def test_no_market_found(self, sniper, peak_forecast_base):
        """Should handle missing market gracefully."""
        ticket = sniper.generate_trade_ticket(
            peak_forecast=peak_forecast_base,
            is_midnight=False,
            midnight_temp=None,
            afternoon_temp=None,
            mav_high=None,
            met_high=None,
            market=None,
        )
        assert ticket.target_ticker == "NO_MARKET_FOUND"
        assert ticket.entry_price_cents == 0

    def test_generate_with_mos_fade(self, sniper):
        """Should apply MOS fade when NWS diverges from models."""
        forecast = HourlyForecast(
            time=datetime.now(ET),
            temp_f=50.0,  # NWS says 50
            wind_speed_mph=5.0,
            wind_gust_mph=5.0,
            short_forecast="Clear",
            is_daytime=True,
            precip_prob=0,
            dewpoint_f=30.0,
        )
        ticket = sniper.generate_trade_ticket(
            peak_forecast=forecast,
            is_midnight=False,
            midnight_temp=None,
            afternoon_temp=None,
            mav_high=44.0,  # Models say 45 avg
            met_high=46.0,
            market=None,
        )
        assert ticket.is_mos_fade is True
        assert ticket.mos_consensus == 45.0

    def test_generate_with_wet_bulb(self, sniper):
        """Should apply wet bulb penalty when conditions warrant."""
        forecast = HourlyForecast(
            time=datetime.now(ET),
            temp_f=70.0,
            wind_speed_mph=5.0,
            wind_gust_mph=5.0,
            short_forecast="Rain",
            is_daytime=True,
            precip_prob=80,  # High precip
            dewpoint_f=55.0,  # 15F depression
        )
        ticket = sniper.generate_trade_ticket(
            peak_forecast=forecast,
            is_midnight=False,
            midnight_temp=None,
            afternoon_temp=None,
            mav_high=None,
            met_high=None,
            market=None,
        )
        assert ticket.wet_bulb_penalty > 0
        assert ticket.is_wet_bulb_risk is True


# =============================================================================
# EXIT SIGNAL TESTS
# =============================================================================

class TestExitSignal:
    """Tests for exit signal generation logic."""

    def test_take_profit_signal(self):
        """Should signal take profit at 100% ROI."""
        # ROI >= 100% should trigger TAKE_PROFIT
        signal = ExitSignal(
            ticker="TEST",
            signal_type="TAKE_PROFIT",
            contracts_held=100,
            avg_entry_cents=20,
            current_bid_cents=45,  # 125% ROI
            roi_percent=125.0,
            target_bracket=(35, 37),
            nws_forecast_high=36.0,
            thesis_valid=True,
            sell_qty=50,  # Sell half
            sell_price_cents=45,
            rationale="Test",
        )
        assert signal.signal_type == "TAKE_PROFIT"
        assert signal.sell_qty == 50  # Half position

    def test_bail_out_signal(self):
        """Should signal bail out when thesis invalid."""
        signal = ExitSignal(
            ticker="TEST",
            signal_type="BAIL_OUT",
            contracts_held=100,
            avg_entry_cents=30,
            current_bid_cents=25,
            roi_percent=-16.7,
            target_bracket=(35, 37),
            nws_forecast_high=40.0,  # Outside bracket
            thesis_valid=False,
            sell_qty=100,  # Full position
            sell_price_cents=25,
            rationale="Test",
        )
        assert signal.signal_type == "BAIL_OUT"
        assert signal.thesis_valid is False
        assert signal.sell_qty == 100


# =============================================================================
# V2: WET BULB PENALTY TESTS
# =============================================================================

class TestWetBulbPenalty:
    """Tests for Strategy D: Wet Bulb / Evaporative Cooling."""

    def test_no_penalty_low_precip(self, sniper):
        """No penalty when precip probability is below threshold."""
        penalty = sniper.calculate_wet_bulb_penalty(
            temp_f=70.0, dewpoint_f=50.0, precip_prob=30  # Below 40%
        )
        assert penalty == 0.0

    def test_no_penalty_saturated_air(self, sniper):
        """No penalty when air is already saturated (small depression)."""
        penalty = sniper.calculate_wet_bulb_penalty(
            temp_f=70.0, dewpoint_f=68.0, precip_prob=60  # Depression < 5F
        )
        assert penalty == 0.0

    def test_light_precip_penalty(self, sniper):
        """Light penalty factor for 40-69% precip probability."""
        # temp=70, dewpoint=55 -> depression=15F, factor=0.25, penalty=3.75 -> 3.8
        penalty = sniper.calculate_wet_bulb_penalty(
            temp_f=70.0, dewpoint_f=55.0, precip_prob=50
        )
        expected = round(15.0 * WET_BULB_FACTOR_LIGHT, 1)
        assert penalty == expected

    def test_heavy_precip_penalty(self, sniper):
        """Heavy penalty factor for >= 70% precip probability."""
        # temp=70, dewpoint=55 -> depression=15F, factor=0.40, penalty=6.0
        penalty = sniper.calculate_wet_bulb_penalty(
            temp_f=70.0, dewpoint_f=55.0, precip_prob=80
        )
        expected = round(15.0 * WET_BULB_FACTOR_HEAVY, 1)
        assert penalty == expected

    def test_boundary_precip_threshold(self, sniper):
        """Test boundary at precip threshold."""
        # At threshold (40%) - should apply penalty
        penalty = sniper.calculate_wet_bulb_penalty(
            temp_f=70.0, dewpoint_f=55.0, precip_prob=WET_BULB_PRECIP_THRESHOLD_PCT
        )
        assert penalty > 0.0

        # Below threshold - no penalty
        penalty = sniper.calculate_wet_bulb_penalty(
            temp_f=70.0, dewpoint_f=55.0, precip_prob=WET_BULB_PRECIP_THRESHOLD_PCT - 1
        )
        assert penalty == 0.0

    def test_boundary_depression_threshold(self, sniper):
        """Test boundary at depression threshold."""
        # Just below min depression (e.g., 4.9F spread) - no penalty
        # Code uses `depression < threshold`, so exactly at threshold DOES trigger
        penalty = sniper.calculate_wet_bulb_penalty(
            temp_f=70.0, dewpoint_f=70.0 - WET_BULB_DEPRESSION_MIN_F + 0.1, precip_prob=60
        )
        assert penalty == 0.0

        # At or above min depression - should apply penalty
        penalty = sniper.calculate_wet_bulb_penalty(
            temp_f=70.0, dewpoint_f=70.0 - WET_BULB_DEPRESSION_MIN_F, precip_prob=60
        )
        assert penalty > 0.0


# =============================================================================
# V2: MOS DIVERGENCE TESTS
# =============================================================================

class TestMOSDivergence:
    """Tests for Strategy E: MOS Consensus Fade."""

    def test_no_divergence_single_source(self, sniper):
        """No fade when NWS matches single MOS source."""
        should_fade, consensus = sniper.check_mos_divergence(
            nws_high=45.0, mav_high=44.0, met_high=None
        )
        assert should_fade is False
        assert consensus == 44.0

    def test_no_divergence_both_sources(self, sniper):
        """No fade when NWS within threshold of MOS consensus."""
        should_fade, consensus = sniper.check_mos_divergence(
            nws_high=45.0, mav_high=44.0, met_high=46.0
        )
        assert should_fade is False
        assert consensus == 45.0

    def test_fade_nws_running_hot(self, sniper):
        """Should fade when NWS exceeds MOS consensus by threshold."""
        # NWS=50, MOS consensus=45, divergence=5 > threshold(2)
        should_fade, consensus = sniper.check_mos_divergence(
            nws_high=50.0, mav_high=44.0, met_high=46.0
        )
        assert should_fade is True
        assert consensus == 45.0

    def test_no_fade_nws_running_cold(self, sniper):
        """No fade when NWS is colder than MOS (conservative)."""
        should_fade, consensus = sniper.check_mos_divergence(
            nws_high=40.0, mav_high=44.0, met_high=46.0
        )
        assert should_fade is False
        assert consensus == 45.0

    def test_no_mos_data(self, sniper):
        """No fade when no MOS data available."""
        should_fade, consensus = sniper.check_mos_divergence(
            nws_high=50.0, mav_high=None, met_high=None
        )
        assert should_fade is False
        assert consensus is None

    def test_boundary_divergence_threshold(self, sniper):
        """Test boundary at divergence threshold."""
        # Exactly at threshold - no fade
        threshold_nws = 45.0 + MOS_DIVERGENCE_THRESHOLD_F
        should_fade, _ = sniper.check_mos_divergence(
            nws_high=threshold_nws, mav_high=45.0, met_high=45.0
        )
        assert should_fade is False

        # Just above threshold - should fade
        should_fade, _ = sniper.check_mos_divergence(
            nws_high=threshold_nws + 0.1, mav_high=45.0, met_high=45.0
        )
        assert should_fade is True


# =============================================================================
# V2: SMART PEGGING TESTS
# =============================================================================

class TestSmartPegging:
    """Tests for Smart Pegging order execution."""

    def test_tight_spread_takes_ask(self, sniper):
        """Should take the ask when spread is tight."""
        entry, rationale = sniper.calculate_smart_entry_price(bid=45, ask=47)
        assert entry == 47  # Takes the ask
        assert "Tight spread" in rationale

    def test_wide_spread_pegs_bid(self, sniper):
        """Should peg bid+1 when spread is wide."""
        entry, rationale = sniper.calculate_smart_entry_price(bid=40, ask=55)
        assert entry == 40 + PEG_OFFSET_CENTS
        assert "Wide spread" in rationale

    def test_no_valid_bid(self, sniper):
        """Should return 0 when bid is too low."""
        entry, rationale = sniper.calculate_smart_entry_price(bid=0, ask=50)
        assert entry == 0
        assert "No valid bid" in rationale

    def test_boundary_spread_threshold(self, sniper):
        """Test boundary at spread threshold."""
        # At threshold - takes ask
        entry, _ = sniper.calculate_smart_entry_price(
            bid=45, ask=45 + MAX_SPREAD_TO_CROSS_CENTS
        )
        assert entry == 45 + MAX_SPREAD_TO_CROSS_CENTS

        # Just above threshold - pegs bid
        entry, _ = sniper.calculate_smart_entry_price(
            bid=45, ask=45 + MAX_SPREAD_TO_CROSS_CENTS + 1
        )
        assert entry == 45 + PEG_OFFSET_CENTS

    def test_minimum_bid_boundary(self, sniper):
        """Test boundary at minimum bid threshold."""
        # At minimum - valid
        entry, rationale = sniper.calculate_smart_entry_price(bid=MIN_BID_CENTS, ask=50)
        assert entry > 0

        # Below minimum - invalid
        entry, rationale = sniper.calculate_smart_entry_price(bid=MIN_BID_CENTS - 1, ask=50)
        assert entry == 0


# =============================================================================
# V2: MOS CLIENT PARSING TESTS
# =============================================================================

class TestMOSClientParsing:
    """Tests for MOS bulletin parsing."""

    def test_parse_mos_valid_xn_line(self):
        """Should parse valid X/N line from MOS bulletin."""
        client = MOSClient()
        sample_mos = """KNYC   GFS MOS GUIDANCE   1/17/2026  1200 UTC
DT /JAN 17            /JAN 18            /JAN 19
HR    00 03 06 09 12 15 18 21 00 03 06 09 12 15 18 21
X/N                48    32    50    35    48
TMP    45 42 38 35 36 40 46 42 38 34 32 34 38 44 48
"""
        forecast = client._parse_mos(sample_mos, "MAV")
        assert forecast is not None
        assert forecast.source == "MAV"
        assert forecast.max_temp_f == 48.0
        assert forecast.min_temp_f == 32.0

    def test_parse_mos_missing_xn_line(self):
        """Should return None when X/N line is missing."""
        client = MOSClient()
        sample_mos = """KNYC   GFS MOS GUIDANCE
DT /JAN 17
TMP    45 42 38 35
"""
        forecast = client._parse_mos(sample_mos, "MAV")
        assert forecast is None

    def test_parse_mos_empty_text(self):
        """Should handle empty text gracefully."""
        client = MOSClient()
        forecast = client._parse_mos("", "MAV")
        assert forecast is None

    def test_parse_mos_malformed_xn(self):
        """Should handle malformed X/N line."""
        client = MOSClient()
        sample_mos = """KNYC
X/N   abc def
"""
        forecast = client._parse_mos(sample_mos, "MAV")
        assert forecast is None


# =============================================================================
# CREDENTIAL VALIDATION TESTS
# =============================================================================

class TestCredentialValidation:
    """Tests for credential validation."""

    def test_missing_api_key(self):
        """Should raise error when API key missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                validate_credentials()
            assert "KALSHI_API_KEY_ID" in str(exc_info.value)

    def test_missing_private_key_path(self):
        """Should raise error when private key path missing."""
        with patch.dict(os.environ, {"KALSHI_API_KEY_ID": "test-key"}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                validate_credentials()
            assert "KALSHI_PRIVATE_KEY_PATH" in str(exc_info.value)

    def test_nonexistent_key_file(self):
        """Should raise error when key file doesn't exist."""
        with patch.dict(os.environ, {
            "KALSHI_API_KEY_ID": "test-key",
            "KALSHI_PRIVATE_KEY_PATH": "/nonexistent/path/key.pem"
        }, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                validate_credentials()
            assert "not found" in str(exc_info.value)


# =============================================================================
# INTEGRATION TESTS (with mocks)
# =============================================================================

class TestIntegration:
    """Integration tests with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_analysis_flow(self, sniper, sample_forecasts):
        """Test full analysis workflow with mocked clients (V2 API)."""
        # Mock NWS client
        sniper.nws = Mock()
        sniper.nws.get_hourly_forecast = AsyncMock(return_value=sample_forecasts)
        sniper.nws.stop = AsyncMock()

        # Mock Kalshi client
        sniper.kalshi = Mock()
        sniper.kalshi.get_markets = AsyncMock(return_value=[])
        sniper.kalshi.get_balance = AsyncMock(return_value=100.0)
        sniper.kalshi.stop = AsyncMock()

        # Run forecast high extraction (V2: get_peak_forecast)
        peak = sniper.get_peak_forecast(sample_forecasts)
        assert peak is not None

        # Generate ticket (V2 API)
        ticket = sniper.generate_trade_ticket(
            peak_forecast=peak,
            is_midnight=False,
            midnight_temp=None,
            afternoon_temp=None,
            mav_high=None,
            met_high=None,
            market=None,
        )

        assert ticket is not None
        assert ticket.nws_forecast_high == peak.temp_f


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
