"""Tests for stale_price_detector.py — Ensemble shift tracking."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from stale_price_detector import (
    StaleAlert,
    ScanSnapshot,
    build_snapshot,
    detect_stale_prices,
    format_stale_alerts,
    load_previous_state,
    save_current_state,
)

ET = ZoneInfo("America/New_York")


# ─── Helpers ───────────────────────────────────────────


def _snap(
    mean: float = 73.0,
    std: float = 1.5,
    bids: dict | None = None,
) -> ScanSnapshot:
    """Create a ScanSnapshot with sensible defaults."""
    if bids is None:
        bids = {
            "T-72-73": {"bid": 45, "title": "72° to 73°F"},
            "T-74-75": {"bid": 30, "title": "74° to 75°F"},
            "T-70-71": {"bid": 20, "title": "70° to 71°F"},
        }
    return ScanSnapshot(
        mean=mean,
        std=std,
        timestamp=datetime.now(ET).isoformat(),
        bracket_bids=bids,
    )


# ─── Test: ScanSnapshot serialization ─────────────────


class TestScanSnapshot:
    """ScanSnapshot round-trip serialization."""

    def test_to_dict_and_back(self):
        snap = _snap(mean=74.5, std=1.2)
        d = snap.to_dict()
        restored = ScanSnapshot.from_dict(d)
        assert restored.mean == 74.5
        assert restored.std == 1.2
        assert "T-72-73" in restored.bracket_bids
        assert restored.bracket_bids["T-72-73"]["bid"] == 45

    def test_from_dict_missing_fields(self):
        snap = ScanSnapshot.from_dict({})
        assert snap.mean == 0
        assert snap.std == 0
        assert snap.bracket_bids == {}
        assert snap.timestamp == ""

    def test_from_dict_extra_fields_ignored(self):
        d = {"mean": 70.0, "std": 1.0, "timestamp": "ts", "bracket_bids": {}, "extra": "junk"}
        snap = ScanSnapshot.from_dict(d)
        assert snap.mean == 70.0


# ─── Test: build_snapshot ─────────────────────────────


class TestBuildSnapshot:
    """build_snapshot correctly extracts bracket bids."""

    def test_basic_build(self):
        brackets = [
            {"ticker": "T-72-73", "title": "72° to 73°F", "yes_bid": 45, "yes_ask": 55},
            {"ticker": "T-74-75", "title": "74° to 75°F", "yes_bid": 30, "yes_ask": 40},
        ]
        snap = build_snapshot("NYC", 73.0, 1.5, brackets)
        assert snap.mean == 73.0
        assert snap.std == 1.5
        assert len(snap.bracket_bids) == 2
        assert snap.bracket_bids["T-72-73"]["bid"] == 45
        assert snap.bracket_bids["T-74-75"]["title"] == "74° to 75°F"

    def test_empty_brackets(self):
        snap = build_snapshot("NYC", 73.0, 1.5, [])
        assert snap.bracket_bids == {}

    def test_missing_ticker_skipped(self):
        brackets = [
            {"ticker": "", "title": "bad", "yes_bid": 50},
            {"ticker": "T-72-73", "title": "72° to 73°F", "yes_bid": 45},
        ]
        snap = build_snapshot("NYC", 73.0, 1.5, brackets)
        assert len(snap.bracket_bids) == 1
        assert "T-72-73" in snap.bracket_bids

    def test_uses_subtitle_fallback(self):
        """If title is empty, should use subtitle."""
        brackets = [
            {"ticker": "T-72-73", "title": "", "subtitle": "72-73 subtitle", "yes_bid": 45},
        ]
        snap = build_snapshot("NYC", 73.0, 1.5, brackets)
        assert snap.bracket_bids["T-72-73"]["title"] == "72-73 subtitle"


# ─── Test: detect_stale_prices ─────────────────────────


class TestDetectStalePrices:
    """Core stale price detection logic."""

    def test_no_previous_returns_empty(self):
        current = _snap()
        alerts = detect_stale_prices("NYC", current, None)
        assert alerts == []

    def test_small_shift_returns_empty(self):
        """Shift of 1.0°F < 1.5°F threshold → no alerts."""
        prev = _snap(mean=73.0)
        curr = _snap(mean=74.0)
        alerts = detect_stale_prices("NYC", curr, prev)
        assert alerts == []

    def test_exactly_at_threshold(self):
        """Shift of exactly 1.5°F → triggers detection (>= threshold)."""
        prev = _snap(mean=73.0, bids={
            "T-74-75": {"bid": 30, "title": "74° to 75°F"},
        })
        curr = _snap(mean=74.5, bids={
            "T-74-75": {"bid": 30, "title": "74° to 75°F"},  # bid didn't move!
        })
        alerts = detect_stale_prices("NYC", curr, prev)
        # expected_change = int(1.5 * 2) = 3
        # bid_change = 0
        # check: 0 < 3 - 8 = -5 → 0 < -5 is False → no alert
        # With MIN_GAP=8, the gap check is lenient for small shifts
        assert alerts == []

    def test_large_warmer_shift_stale_bid(self, monkeypatch):
        """Ensemble warms by 3°F but bracket bid barely moved → alert."""
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_ENABLED", True)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_SHIFT_F", 1.5)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_GAP_CENTS", 2)

        prev = _snap(mean=70.0, bids={
            "T-72-73": {"bid": 40, "title": "72° to 73°F"},
        })
        curr = _snap(mean=73.0, bids={
            "T-72-73": {"bid": 41, "title": "72° to 73°F"},  # only +1¢
        })
        alerts = detect_stale_prices("NYC", curr, prev)
        # mean_shift = +3.0°F, expected_change = 6¢
        # bid_change = +1¢
        # check: 1 < 6 - 2 = 4 → True → alert!
        assert len(alerts) == 1
        assert alerts[0].direction == "warmer"
        assert alerts[0].mean_shift_f == 3.0
        assert alerts[0].prev_bid == 40
        assert alerts[0].actual_bid == 41

    def test_large_cooler_shift_stale_bid(self, monkeypatch):
        """Ensemble cools by 3°F but bracket bid barely moved → alert."""
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_ENABLED", True)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_SHIFT_F", 1.5)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_GAP_CENTS", 2)

        prev = _snap(mean=75.0, bids={
            "T-72-73": {"bid": 45, "title": "72° to 73°F"},
        })
        curr = _snap(mean=72.0, bids={
            "T-72-73": {"bid": 44, "title": "72° to 73°F"},  # only -1¢
        })
        alerts = detect_stale_prices("NYC", curr, prev)
        # mean_shift = -3.0°F (cooler), expected_change = 6¢
        # bid_change = -1¢
        # check: -1 > -(6 - 2) = -4 → -1 > -4 → True → alert!
        assert len(alerts) == 1
        assert alerts[0].direction == "cooler"
        assert alerts[0].mean_shift_f == -3.0

    def test_bid_moved_appropriately_no_alert(self, monkeypatch):
        """Bid repriced correctly → no alert."""
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_ENABLED", True)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_SHIFT_F", 1.5)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_GAP_CENTS", 2)

        prev = _snap(mean=70.0, bids={
            "T-72-73": {"bid": 40, "title": "72° to 73°F"},
        })
        curr = _snap(mean=73.0, bids={
            "T-72-73": {"bid": 50, "title": "72° to 73°F"},  # +10¢
        })
        alerts = detect_stale_prices("NYC", curr, prev)
        # bid_change = +10, expected = 6, 10 >= 6-2 = 4 → no alert
        assert alerts == []

    def test_disabled_returns_empty(self, monkeypatch):
        """Feature flag off → no alerts."""
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_ENABLED", False)
        prev = _snap(mean=70.0)
        curr = _snap(mean=75.0)
        alerts = detect_stale_prices("NYC", curr, prev)
        assert alerts == []

    def test_zero_mean_returns_empty(self):
        """Zero mean in either snapshot → skip."""
        prev = _snap(mean=0)
        curr = _snap(mean=73.0)
        alerts = detect_stale_prices("NYC", curr, prev)
        assert alerts == []

        prev2 = _snap(mean=73.0)
        curr2 = _snap(mean=0)
        alerts2 = detect_stale_prices("NYC", curr2, prev2)
        assert alerts2 == []

    def test_new_ticker_not_in_previous(self, monkeypatch):
        """Bracket in current but not in previous → skipped, no crash."""
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_ENABLED", True)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_SHIFT_F", 1.0)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_GAP_CENTS", 2)

        prev = _snap(mean=70.0, bids={
            "T-70-71": {"bid": 40, "title": "70° to 71°F"},
        })
        curr = _snap(mean=72.0, bids={
            "T-72-73": {"bid": 45, "title": "72° to 73°F"},  # new ticker
        })
        alerts = detect_stale_prices("NYC", curr, prev)
        assert alerts == []  # skipped because T-72-73 not in previous

    def test_zero_bid_skipped(self, monkeypatch):
        """Brackets with zero bid are skipped."""
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_ENABLED", True)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_SHIFT_F", 1.0)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_GAP_CENTS", 2)

        prev = _snap(mean=70.0, bids={
            "T-72-73": {"bid": 0, "title": "72° to 73°F"},
        })
        curr = _snap(mean=73.0, bids={
            "T-72-73": {"bid": 40, "title": "72° to 73°F"},
        })
        alerts = detect_stale_prices("NYC", curr, prev)
        assert alerts == []

    def test_bid_outside_tradeable_range_skipped(self, monkeypatch):
        """Bids below 15¢ or above 85¢ are skipped (not tradeable)."""
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_ENABLED", True)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_SHIFT_F", 1.0)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_GAP_CENTS", 0)

        prev = _snap(mean=70.0, bids={
            "T-low": {"bid": 5, "title": "Low bracket"},
            "T-high": {"bid": 90, "title": "High bracket"},
        })
        curr = _snap(mean=73.0, bids={
            "T-low": {"bid": 5, "title": "Low bracket"},
            "T-high": {"bid": 90, "title": "High bracket"},
        })
        alerts = detect_stale_prices("NYC", curr, prev)
        assert alerts == []

    def test_multiple_stale_brackets(self, monkeypatch):
        """Multiple brackets stale in same shift → multiple alerts."""
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_ENABLED", True)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_SHIFT_F", 1.5)
        monkeypatch.setattr("stale_price_detector.STALE_PRICE_MIN_GAP_CENTS", 2)

        prev = _snap(mean=70.0, bids={
            "T-72-73": {"bid": 40, "title": "72° to 73°F"},
            "T-74-75": {"bid": 30, "title": "74° to 75°F"},
        })
        curr = _snap(mean=73.0, bids={
            "T-72-73": {"bid": 40, "title": "72° to 73°F"},  # didn't move
            "T-74-75": {"bid": 30, "title": "74° to 75°F"},  # didn't move
        })
        alerts = detect_stale_prices("NYC", curr, prev)
        assert len(alerts) == 2
        tickers = {a.ticker for a in alerts}
        assert "T-72-73" in tickers
        assert "T-74-75" in tickers


# ─── Test: state persistence ──────────────────────────


class TestStatePersistence:
    """State save/load round-trips correctly."""

    def test_round_trip(self, tmp_path, monkeypatch):
        state_file = tmp_path / "stale_state.json"
        monkeypatch.setattr("stale_price_detector.STATE_PATH", state_file)

        states = {
            "NYC": _snap(mean=73.5, std=1.2),
            "CHI": _snap(mean=65.0, std=2.0),
        }
        save_current_state(states)
        loaded = load_previous_state()

        assert "NYC" in loaded
        assert "CHI" in loaded
        assert loaded["NYC"].mean == 73.5
        assert loaded["CHI"].std == 2.0
        assert "T-72-73" in loaded["NYC"].bracket_bids

    def test_empty_state_file(self, tmp_path, monkeypatch):
        state_file = tmp_path / "stale_state.json"
        monkeypatch.setattr("stale_price_detector.STATE_PATH", state_file)
        loaded = load_previous_state()
        assert loaded == {}

    def test_corrupt_state_file(self, tmp_path, monkeypatch):
        state_file = tmp_path / "stale_state.json"
        state_file.write_text("not json {{{")
        monkeypatch.setattr("stale_price_detector.STATE_PATH", state_file)
        loaded = load_previous_state()
        assert loaded == {}

    def test_overwrite_state(self, tmp_path, monkeypatch):
        """Second save overwrites first."""
        state_file = tmp_path / "stale_state.json"
        monkeypatch.setattr("stale_price_detector.STATE_PATH", state_file)

        save_current_state({"NYC": _snap(mean=70.0)})
        save_current_state({"NYC": _snap(mean=75.0)})
        loaded = load_previous_state()
        assert loaded["NYC"].mean == 75.0


# ─── Test: format_stale_alerts ─────────────────────────


class TestFormatStaleAlerts:
    """Alert formatting for Discord."""

    def test_empty_alerts(self):
        assert format_stale_alerts([]) == ""

    def test_single_warmer_alert(self, monkeypatch):
        # Mock shorten_bracket_title to avoid importing full scanner
        monkeypatch.setattr(
            "stale_price_detector.shorten_bracket_title",
            lambda t: t[:10],
            raising=False,
        )
        # Need to also handle the import inside format_stale_alerts
        import stale_price_detector
        monkeypatch.setattr(
            stale_price_detector, "format_stale_alerts",
            stale_price_detector.format_stale_alerts,
        )
        # Provide the imported function reference
        import edge_scanner_v2
        monkeypatch.setattr(
            edge_scanner_v2, "shorten_bracket_title",
            lambda t: t[:10],
            raising=False,
        )

        alert = StaleAlert(
            city="NYC",
            direction="warmer",
            mean_shift_f=2.5,
            prev_mean=70.0,
            curr_mean=72.5,
            bracket_title="72° to 73°F",
            ticker="T-72-73",
            expected_bid_change=5,
            actual_bid=40,
            prev_bid=38,
        )
        result = format_stale_alerts([alert])
        assert "STALE PRICE" in result
        assert "NYC" in result
        assert "🔴" in result  # warmer
        assert "+2.5°F" in result
        assert "T-72-73" in result

    def test_cooler_alert_uses_blue(self, monkeypatch):
        import edge_scanner_v2
        monkeypatch.setattr(
            edge_scanner_v2, "shorten_bracket_title",
            lambda t: t[:10],
            raising=False,
        )

        alert = StaleAlert(
            city="CHI",
            direction="cooler",
            mean_shift_f=-2.0,
            prev_mean=72.0,
            curr_mean=70.0,
            bracket_title="70° to 71°F",
            ticker="T-70-71",
            expected_bid_change=4,
            actual_bid=35,
            prev_bid=36,
        )
        result = format_stale_alerts([alert])
        assert "🔵" in result  # cooler
        assert "CHI" in result

    def test_caps_at_five_alerts(self, monkeypatch):
        import edge_scanner_v2
        monkeypatch.setattr(
            edge_scanner_v2, "shorten_bracket_title",
            lambda t: "short",
            raising=False,
        )

        alerts = [
            StaleAlert(
                city=f"CITY{i}",
                direction="warmer",
                mean_shift_f=2.0,
                prev_mean=70.0,
                curr_mean=72.0,
                bracket_title=f"Bracket {i}",
                ticker=f"T-{i}",
                expected_bid_change=4,
                actual_bid=40,
                prev_bid=38,
            )
            for i in range(8)
        ]
        result = format_stale_alerts(alerts)
        # Should have "1 bracket(s)" in header → no, 8 alerts but only 5 printed
        assert "8 bracket(s)" in result
        # Count city occurrences — should be capped at 5
        assert result.count("CITY") == 5


# ─── Test: StaleAlert dataclass ────────────────────────


class TestStaleAlert:
    """StaleAlert basic field access."""

    def test_field_access(self):
        alert = StaleAlert(
            city="DEN",
            direction="warmer",
            mean_shift_f=1.8,
            prev_mean=60.0,
            curr_mean=61.8,
            bracket_title="62° to 63°F",
            ticker="T-62-63",
            expected_bid_change=3,
            actual_bid=25,
            prev_bid=23,
        )
        assert alert.city == "DEN"
        assert alert.direction == "warmer"
        assert alert.mean_shift_f == 1.8
        assert alert.prev_mean == 60.0
        assert alert.curr_mean == 61.8
        assert alert.ticker == "T-62-63"
        assert alert.expected_bid_change == 3
        assert alert.actual_bid == 25
        assert alert.prev_bid == 23
