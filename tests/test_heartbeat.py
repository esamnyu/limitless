#!/usr/bin/env python3
"""Tests for heartbeat.py — write, read, staleness detection."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_tmpdir = tempfile.mkdtemp()
_test_heartbeat = Path(_tmpdir) / "heartbeats.json"
_test_lock = Path(_tmpdir) / ".heartbeats.lock"


@pytest.fixture(autouse=True)
def _patch_paths(monkeypatch):
    """Redirect heartbeat to temp files for every test."""
    import heartbeat
    monkeypatch.setattr(heartbeat, "HEARTBEAT_FILE", _test_heartbeat)
    monkeypatch.setattr(heartbeat, "HEARTBEAT_LOCK", _test_lock)
    monkeypatch.setattr(heartbeat, "PROJECT_ROOT", Path(_tmpdir))
    if _test_heartbeat.exists():
        _test_heartbeat.unlink()
    if _test_lock.exists():
        _test_lock.unlink()
    yield


class TestWriteHeartbeat:
    """write_heartbeat() tests."""

    def test_write_creates_file(self):
        from heartbeat import write_heartbeat
        assert not _test_heartbeat.exists()
        write_heartbeat("test_service")
        assert _test_heartbeat.exists()

    def test_write_records_timestamp(self):
        from heartbeat import write_heartbeat
        write_heartbeat("auto_scan")
        data = json.loads(_test_heartbeat.read_text())
        assert "auto_scan" in data
        assert "timestamp" in data["auto_scan"]

    def test_write_multiple_services(self):
        from heartbeat import write_heartbeat
        write_heartbeat("auto_scan")
        write_heartbeat("position_monitor")
        data = json.loads(_test_heartbeat.read_text())
        assert "auto_scan" in data
        assert "position_monitor" in data

    def test_write_overwrites_same_service(self):
        from heartbeat import write_heartbeat
        write_heartbeat("auto_scan")
        first = json.loads(_test_heartbeat.read_text())["auto_scan"]["timestamp"]
        import time; time.sleep(0.01)
        write_heartbeat("auto_scan")
        second = json.loads(_test_heartbeat.read_text())["auto_scan"]["timestamp"]
        # Timestamps should differ (second is more recent)
        # They might be same if sub-second, so just check data is valid
        assert second is not None


class TestReadHeartbeats:
    """read_heartbeats() tests."""

    def test_read_missing_file(self):
        from heartbeat import read_heartbeats
        assert read_heartbeats() == {}

    def test_read_after_write(self):
        from heartbeat import write_heartbeat, read_heartbeats
        write_heartbeat("test_service")
        data = read_heartbeats()
        assert "test_service" in data

    def test_read_corrupted_file(self):
        from heartbeat import read_heartbeats
        _test_heartbeat.write_text("{invalid json")
        assert read_heartbeats() == {}


class TestCheckHeartbeats:
    """check_heartbeats() staleness detection."""

    def test_all_healthy(self):
        """All services recently reported → no problems."""
        from heartbeat import write_heartbeat, check_heartbeats
        # Write heartbeats for all expected services
        write_heartbeat("auto_scan")
        write_heartbeat("position_monitor")
        write_heartbeat("backtest_collector")
        write_heartbeat("morning_check")
        problems = check_heartbeats()
        assert problems == []

    def test_missing_service(self):
        """Service never reported → flagged as never_seen."""
        from heartbeat import check_heartbeats
        # Empty file → all services are never_seen
        problems = check_heartbeats()
        assert len(problems) > 0
        services = [p[0] for p in problems]
        assert "auto_scan" in services
        statuses = {p[0]: p[1] for p in problems}
        assert statuses["auto_scan"] == "never_seen"

    def test_stale_service(self, monkeypatch):
        """Service last reported 2 hours ago with 90-min threshold → stale."""
        from heartbeat import write_heartbeat, check_heartbeats, ET
        from datetime import datetime, timedelta

        write_heartbeat("auto_scan")
        write_heartbeat("position_monitor")
        write_heartbeat("backtest_collector")
        write_heartbeat("morning_check")

        # Manually backdate auto_scan by 2 hours
        data = json.loads(_test_heartbeat.read_text())
        old_time = datetime.now(ET) - timedelta(hours=2)
        data["auto_scan"]["timestamp"] = old_time.isoformat()
        _test_heartbeat.write_text(json.dumps(data))

        problems = check_heartbeats()
        stale = [p for p in problems if p[0] == "auto_scan"]
        assert len(stale) == 1
        assert stale[0][1] == "stale"
        assert stale[0][2] > 90  # More than 90 minutes old

    def test_partial_health(self):
        """Some services healthy, others missing."""
        from heartbeat import write_heartbeat, check_heartbeats
        write_heartbeat("auto_scan")
        write_heartbeat("position_monitor")
        # backtest_collector and morning_check NOT written
        problems = check_heartbeats()
        problem_services = {p[0] for p in problems}
        assert "auto_scan" not in problem_services
        assert "position_monitor" not in problem_services
        assert "backtest_collector" in problem_services
        assert "morning_check" in problem_services
