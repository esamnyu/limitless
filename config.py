"""
Project Atlas - Configuration Module
Loads environment variables and defines trading parameters.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file
load_dotenv()


@dataclass
class Config:
    """Central configuration for the Paper Sniper."""

    # Limitless API
    limitless_api_url: str = "https://api.limitless.exchange"

    # Market to monitor (slug from URL, e.g., "btc-above-90757-jan-11-2000-utc")
    limitless_market_slug: str = os.getenv("LIMITLESS_MARKET_SLUG", "")

    # Binance Settings
    binance_symbol: str = "BTC/USDT:USDT"  # Perpetual Futures

    # Strategy Parameters
    velocity_threshold_usd: float = 50.0  # Trigger when price moves $50 in 1 second
    poll_interval_seconds: float = 5.0    # Poll Limitless every 5 seconds (avoid rate limits)
    price_history_window: int = 10        # Keep last 10 price samples for velocity calc

    # Safety: Minimum time to expiry (avoid adaptive fees)
    min_time_to_expiry_minutes: int = 15

    # Phase 1: Paper Trading (no real execution)
    paper_trading: bool = True

    # Logging
    log_to_csv: bool = True
    csv_filename: str = "paper_sniper_log.csv"

    def validate(self) -> bool:
        """Validate required configuration is present."""
        errors = []

        if not self.limitless_market_slug:
            errors.append("LIMITLESS_MARKET_SLUG is required in .env")
            errors.append("  Get it from the market URL: limitless.exchange/markets/<slug>")

        if errors:
            for error in errors:
                print(f"[CONFIG ERROR] {error}")
            return False

        return True

    def __str__(self) -> str:
        return f"""
=== Project Atlas Configuration ===
Limitless API: {self.limitless_api_url}
Market Slug: {self.limitless_market_slug}
Binance Symbol: {self.binance_symbol}
Velocity Threshold: ${self.velocity_threshold_usd}
Poll Interval: {self.poll_interval_seconds}s
Min Time to Expiry: {self.min_time_to_expiry_minutes} min
Paper Trading: {self.paper_trading}
===================================
"""


# Global config instance
config = Config()
