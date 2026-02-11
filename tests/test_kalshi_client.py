#!/usr/bin/env python3
"""
Tests for Kalshi API client.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from kalshi_client import KalshiClient, KalshiAPIError, KalshiRateLimitError


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Create a KalshiClient instance for testing."""
    return KalshiClient(
        api_key_id="test-api-key",
        private_key_path="",
        demo_mode=True,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestInitialization:
    """Tests for client initialization."""

    def test_demo_mode_url(self):
        """Demo mode should use demo URL."""
        client = KalshiClient(demo_mode=True)
        assert "demo" in client.base_url.lower()

    def test_live_mode_url(self):
        """Live mode should use production URL."""
        client = KalshiClient(demo_mode=False)
        assert "elections" in client.base_url.lower()

    def test_initial_counters(self, client):
        """Request counters should start at zero."""
        assert client._request_count == 0
        assert client._error_count == 0


# =============================================================================
# ERROR CLASS TESTS
# =============================================================================

class TestErrors:
    """Tests for error classes."""

    def test_api_error(self):
        """KalshiAPIError should contain status and message."""
        error = KalshiAPIError(400, "Bad request")
        assert error.status == 400
        assert "400" in str(error)
        assert "Bad request" in str(error)

    def test_rate_limit_error(self):
        """KalshiRateLimitError should contain retry_after."""
        error = KalshiRateLimitError(retry_after=30)
        assert error.status == 429
        assert error.retry_after == 30
        assert "30" in str(error)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
