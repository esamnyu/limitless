#!/usr/bin/env python3
"""
MIDNIGHT SCANNER v1.0 - Automated Weather Trading Scanner

Automatically scans for trading opportunities during the optimal 11pm-12am window.
Sends Discord alerts when high-edge opportunities are detected.

Usage:
  python midnight_scanner.py              # Run scanner daemon (continuous)
  python midnight_scanner.py --once       # Run single scan and exit
  python midnight_scanner.py --test-notify # Test Discord webhook

Schedule (ET):
  11:00 PM - First scan (overnight setup)
  11:30 PM - Second scan (updated NWS data)
  11:55 PM - Critical scan (rounding arbitrage window)
  12:05 AM - Post-midnight verification
"""

import argparse
import asyncio
import logging
import os
import re
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from config import (
    StationConfig,
    get_station_config,
    DEFAULT_CITY,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)
from kalshi_client import KalshiClient
from nws_client import NWSClient, MOSClient
from notifier import DiscordNotifier
from strategies import (
    HourlyForecast,
    TradeTicket,
    check_midnight_high,
    get_peak_forecast,
    generate_trade_ticket,
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)

# =============================================================================
# SCANNER CONFIGURATION
# =============================================================================

# Scan schedule (24-hour format, ET timezone)
SCAN_TIMES = ["23:00", "23:30", "23:55", "00:05"]

# Minimum edge to trigger an alert (40%)
ALERT_EDGE_THRESHOLD = 0.40

# Only alert for BUY recommendations
ALERT_RECOMMENDATIONS = ["BUY", "FADE_NWS"]


# =============================================================================
# SCANNER CLASS
# =============================================================================

class MidnightScanner:
    """Automated scanner for midnight weather trading opportunities."""

    def __init__(self, city_code: str = DEFAULT_CITY):
        self.city_code = city_code.upper()
        self.station_config = get_station_config(self.city_code)
        self.tz = ZoneInfo(self.station_config.timezone)

        # Clients
        self.nws: Optional[NWSClient] = None
        self.mos: Optional[MOSClient] = None
        self.kalshi: Optional[KalshiClient] = None
        self.notifier: Optional[DiscordNotifier] = None

        # State
        self.running = False
        self.last_scan_time: Optional[datetime] = None

    async def start(self):
        """Initialize all clients."""
        logger.info(f"Starting Midnight Scanner for {self.city_code}")

        # Validate credentials
        api_key = os.getenv("KALSHI_API_KEY_ID")
        private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

        if not api_key or not private_key_path:
            logger.error("Missing KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH")
            raise SystemExit(1)

        # Initialize NWS client
        self.nws = NWSClient(self.station_config)
        await self.nws.start()

        # Initialize MOS client
        self.mos = MOSClient(self.station_config)
        await self.mos.start()

        # Initialize Kalshi client
        self.kalshi = KalshiClient(
            api_key_id=api_key,
            private_key_path=private_key_path,
            demo_mode=False,
        )
        await self.kalshi.start()

        # Initialize Discord notifier
        self.notifier = DiscordNotifier()
        await self.notifier.start()

        self.running = True
        logger.info("Midnight Scanner initialized")

    async def stop(self):
        """Shutdown all clients."""
        self.running = False

        if self.nws:
            await self.nws.stop()
        if self.mos:
            await self.mos.stop()
        if self.kalshi:
            await self.kalshi.stop()
        if self.notifier:
            await self.notifier.stop()

        logger.info("Midnight Scanner stopped")

    async def get_kalshi_markets(self) -> list[dict]:
        """Fetch open markets for the configured city."""
        try:
            markets = await self.kalshi.get_markets(
                series_ticker=self.station_config.series_ticker,
                status="open",
                limit=100
            )
            logger.debug(f"Fetched {len(markets)} markets")
            return markets
        except Exception as e:
            logger.error(f"Market fetch failed: {e}")
            return []

    def find_target_market(self, markets: list[dict], target_temp: float) -> Optional[dict]:
        """Find the market bracket containing the target temperature."""
        now = datetime.now(self.tz)
        tomorrow = now + timedelta(days=1)
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                  'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
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

    async def run_scan(self) -> Optional[TradeTicket]:
        """Execute a single scan and return the trade ticket."""
        scan_start = datetime.now(self.tz)
        scan_time_str = scan_start.strftime("%I:%M %p ET")

        logger.info(f"Running scan at {scan_time_str}")
        print(f"\n{'='*60}")
        print(f"  MIDNIGHT SCANNER - {scan_time_str}")
        print(f"  City: {self.station_config.city_name}")
        print(f"{'='*60}")

        # 1. Fetch NWS hourly forecast
        print("\n[1/4] Fetching NWS hourly forecast...")
        forecasts = await self.nws.get_hourly_forecast()
        if not forecasts:
            logger.error("No forecast data available")
            print("[ERR] No forecast data available.")
            return None

        # 2. Fetch MOS model data
        print("[2/4] Fetching MOS model data...")
        mav_forecast = await self.mos.get_mav()
        met_forecast = await self.mos.get_met()

        mav_high = mav_forecast.max_temp_f if mav_forecast else None
        met_high = met_forecast.max_temp_f if met_forecast else None

        if mav_high:
            print(f"  MAV (GFS MOS): {mav_high:.0f}F")
        if met_high:
            print(f"  MET (NAM MOS): {met_high:.0f}F")

        # 3. Analyze weather patterns
        print("[3/4] Analyzing weather patterns...")

        peak_forecast = get_peak_forecast(forecasts, self.tz)
        if not peak_forecast:
            logger.error("Could not determine peak forecast")
            print("[ERR] Could not determine peak forecast.")
            return None

        is_midnight, midnight_temp, afternoon_temp = check_midnight_high(forecasts, self.tz)

        print(f"  NWS Forecast High: {peak_forecast.temp_f:.0f}F")
        print(f"  Peak Hour Wind:    {peak_forecast.wind_gust_mph:.0f} mph gusts")
        print(f"  Midnight High:     {'YES' if is_midnight else 'No'}")

        # 4. Fetch Kalshi markets and generate ticket
        print("[4/4] Fetching Kalshi markets...")
        markets = await self.get_kalshi_markets()

        # Calculate physics high for market lookup
        from strategies import calculate_wind_penalty, calculate_wet_bulb_penalty
        wind_penalty = calculate_wind_penalty(peak_forecast.wind_gust_mph)
        wet_bulb_penalty = calculate_wet_bulb_penalty(
            peak_forecast.temp_f, peak_forecast.dewpoint_f, peak_forecast.precip_prob
        )
        physics_high = peak_forecast.temp_f - wind_penalty - wet_bulb_penalty

        if is_midnight and midnight_temp:
            physics_high = midnight_temp

        target_market = self.find_target_market(markets, physics_high)

        # Generate trade ticket
        ticket = generate_trade_ticket(
            peak_forecast=peak_forecast,
            is_midnight=is_midnight,
            midnight_temp=midnight_temp,
            afternoon_temp=afternoon_temp,
            mav_high=mav_high,
            met_high=met_high,
            market=target_market,
        )

        # Print summary
        self._print_ticket_summary(ticket)

        # Record scan time
        self.last_scan_time = scan_start

        return ticket

    def _print_ticket_summary(self, ticket: TradeTicket):
        """Print a compact ticket summary."""
        print("\n" + "-"*60)
        print(f"TARGET:      {ticket.target_bracket_low}F to {ticket.target_bracket_high}F")
        print(f"TICKER:      {ticket.target_ticker}")
        print(f"PRICE:       {ticket.current_bid_cents}c / {ticket.current_ask_cents}c")
        print(f"ENTRY:       {ticket.entry_price_cents}c")
        print(f"EDGE:        {'+' if ticket.estimated_edge > 0 else ''}{ticket.estimated_edge:.0%}")
        print(f"CONFIDENCE:  {ticket.confidence}/10")
        print("-"*60)
        print(f">>> RECOMMENDATION: {ticket.recommendation} <<<")
        print("="*60)

    async def check_and_alert(self, ticket: TradeTicket):
        """Check if ticket meets alert criteria and send notification."""
        # Check edge threshold
        if ticket.estimated_edge < ALERT_EDGE_THRESHOLD:
            logger.info(
                f"Edge {ticket.estimated_edge:.0%} below threshold "
                f"{ALERT_EDGE_THRESHOLD:.0%} - no alert"
            )
            print(f"\n[INFO] Edge {ticket.estimated_edge:.0%} < {ALERT_EDGE_THRESHOLD:.0%} threshold - no alert sent")
            return

        # Check recommendation
        if ticket.recommendation not in ALERT_RECOMMENDATIONS:
            logger.info(f"Recommendation {ticket.recommendation} not alertable")
            print(f"\n[INFO] Recommendation {ticket.recommendation} - no alert sent")
            return

        # Send Discord alert
        scan_time = datetime.now(self.tz).strftime("%I:%M %p ET")
        success = await self.notifier.send_trade_alert(ticket, scan_time)

        if success:
            print(f"\n[ALERT] Discord notification sent for {ticket.target_ticker}")
        else:
            print(f"\n[WARN] Failed to send Discord notification")

    def _get_next_scan_time(self) -> datetime:
        """Calculate the next scheduled scan time."""
        now = datetime.now(self.tz)

        for time_str in SCAN_TIMES:
            hour, minute = map(int, time_str.split(":"))
            scan_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Handle midnight crossing
            if hour < 12 and now.hour >= 12:
                scan_time += timedelta(days=1)

            if scan_time > now:
                return scan_time

        # All scans for today are done - schedule first scan tomorrow
        hour, minute = map(int, SCAN_TIMES[0].split(":"))
        return (now + timedelta(days=1)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )

    async def run_daemon(self):
        """Run the scanner as a daemon with scheduled scans."""
        logger.info("Starting scanner daemon")
        print("\n" + "="*60)
        print("  MIDNIGHT SCANNER DAEMON")
        print(f"  Schedule: {', '.join(SCAN_TIMES)} ET")
        print(f"  Alert Threshold: {ALERT_EDGE_THRESHOLD:.0%} edge")
        print("="*60)

        while self.running:
            next_scan = self._get_next_scan_time()
            wait_seconds = (next_scan - datetime.now(self.tz)).total_seconds()

            if wait_seconds > 0:
                print(f"\n[WAITING] Next scan at {next_scan.strftime('%I:%M %p ET')} "
                      f"({wait_seconds/60:.1f} minutes)")
                logger.info(f"Sleeping until {next_scan}")

                # Sleep in chunks to allow for graceful shutdown
                while wait_seconds > 0 and self.running:
                    sleep_time = min(wait_seconds, 60)
                    await asyncio.sleep(sleep_time)
                    wait_seconds -= sleep_time

            if not self.running:
                break

            # Run the scan
            try:
                ticket = await self.run_scan()
                if ticket:
                    await self.check_and_alert(ticket)
            except Exception as e:
                logger.exception(f"Scan error: {e}")
                print(f"\n[ERROR] Scan failed: {e}")

            # Small delay to avoid double-scanning
            await asyncio.sleep(5)

        logger.info("Scanner daemon stopped")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Midnight Scanner - Automated Weather Trading Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Schedule (ET):
  11:00 PM - First scan (overnight setup)
  11:30 PM - Second scan (updated NWS data)
  11:55 PM - Critical scan (rounding arbitrage window)
  12:05 AM - Post-midnight verification

Examples:
  python midnight_scanner.py              # Run daemon (continuous)
  python midnight_scanner.py --once       # Single scan
  python midnight_scanner.py --test-notify # Test Discord webhook
        """
    )
    parser.add_argument(
        "--city",
        type=str,
        default=DEFAULT_CITY,
        help=f"City code (default: {DEFAULT_CITY})"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan and exit"
    )
    parser.add_argument(
        "--test-notify",
        action="store_true",
        help="Send a test Discord notification"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    scanner = MidnightScanner(city_code=args.city)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        scanner.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await scanner.start()

    try:
        if args.test_notify:
            # Test Discord webhook
            print("\n[TEST] Sending test Discord notification...")
            success = await scanner.notifier.send_test_message()
            if success:
                print("[TEST] Discord webhook test PASSED")
            else:
                print("[TEST] Discord webhook test FAILED")
                print("       Check DISCORD_WEBHOOK_URL in .env file")

        elif args.once:
            # Single scan mode
            ticket = await scanner.run_scan()
            if ticket:
                await scanner.check_and_alert(ticket)

        else:
            # Daemon mode
            await scanner.run_daemon()

    finally:
        await scanner.stop()


if __name__ == "__main__":
    asyncio.run(main())
