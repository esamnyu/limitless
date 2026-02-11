#!/usr/bin/env python3
"""
WEATHER SNIPER - Discord Notification Module

Sends trade alerts via Discord webhooks with rate limiting.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import aiohttp
from dotenv import load_dotenv

from strategies import TradeTicket

load_dotenv()
logger = logging.getLogger(__name__)

# Rate limiting: Track last alert time per ticker
_last_alert_times: dict[str, datetime] = {}
ALERT_COOLDOWN_MINUTES = 30  # Don't spam the same ticker


class DiscordNotifier:
    """Discord webhook notifier for trade alerts."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Initialize the HTTP session."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        if self.webhook_url:
            logger.info("Discord notifier initialized")
        else:
            logger.warning("DISCORD_WEBHOOK_URL not set - notifications disabled")

    async def stop(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()

    def _should_send_alert(self, ticker: str) -> bool:
        """Check if we should send an alert (rate limiting)."""
        if ticker in _last_alert_times:
            elapsed = datetime.now() - _last_alert_times[ticker]
            if elapsed < timedelta(minutes=ALERT_COOLDOWN_MINUTES):
                logger.debug(f"Rate limited: {ticker} (last alert {elapsed.seconds}s ago)")
                return False
        return True

    def _record_alert(self, ticker: str):
        """Record that we sent an alert for a ticker."""
        _last_alert_times[ticker] = datetime.now()

    def _format_trade_ticket(self, ticket: TradeTicket, scan_time: str) -> dict:
        """Format a trade ticket as a Discord embed."""
        # Color based on recommendation
        color_map = {
            "BUY": 0x00FF00,       # Green
            "FADE_NWS": 0xFFAA00,  # Orange
            "PASS": 0x808080,      # Gray
        }
        color = color_map.get(ticket.recommendation, 0x808080)

        # Build description
        description_lines = [
            f"**NWS Forecast High:** {ticket.nws_forecast_high:.0f}F",
            f"**Physics High:** {ticket.physics_high:.1f}F",
        ]

        if ticket.wind_penalty > 0:
            description_lines.append(f"  - Wind Penalty: -{ticket.wind_penalty:.1f}F")
        if ticket.wet_bulb_penalty > 0:
            description_lines.append(f"  - Wet Bulb Penalty: -{ticket.wet_bulb_penalty:.1f}F")

        description_lines.extend([
            "",
            f"**Target Bracket:** {ticket.target_bracket_low}F to {ticket.target_bracket_high}F",
            f"**Current Price:** {ticket.current_bid_cents}c / {ticket.current_ask_cents}c",
            f"**Entry Price:** {ticket.entry_price_cents}c",
            f"**Implied Odds:** {ticket.implied_odds:.0%}",
            f"**Estimated Edge:** {'+' if ticket.estimated_edge > 0 else ''}{ticket.estimated_edge:.0%}",
            f"**Confidence:** {ticket.confidence}/10",
        ])

        # Add strategy flags
        flags = []
        if ticket.is_midnight_risk:
            flags.append("Midnight High")
        if ticket.is_wet_bulb_risk:
            flags.append("Wet Bulb")
        if ticket.is_mos_fade:
            flags.append("MOS Fade")
        if flags:
            description_lines.append(f"\n**Active Strategies:** {', '.join(flags)}")

        embed = {
            "title": f"NYC SNIPER ALERT - {ticket.recommendation}",
            "description": "\n".join(description_lines),
            "color": color,
            "fields": [
                {
                    "name": "Ticker",
                    "value": f"`{ticket.target_ticker}`",
                    "inline": True,
                },
                {
                    "name": "Scan Time",
                    "value": scan_time,
                    "inline": True,
                },
            ],
            "footer": {
                "text": f"Rationale: {ticket.rationale}",
            },
        }

        return embed

    async def send_trade_alert(
        self,
        ticket: TradeTicket,
        scan_time: Optional[str] = None
    ) -> bool:
        """
        Send a trade alert to Discord.

        Returns True if alert was sent, False if skipped or failed.
        """
        if not self.webhook_url:
            logger.warning("Cannot send alert: DISCORD_WEBHOOK_URL not configured")
            return False

        if not self.session:
            logger.error("Notifier session not started")
            return False

        # Rate limiting
        if not self._should_send_alert(ticket.target_ticker):
            logger.info(f"Skipping alert for {ticket.target_ticker} (rate limited)")
            return False

        # Format the message
        scan_time = scan_time or datetime.now().strftime("%I:%M %p ET")
        embed = self._format_trade_ticket(ticket, scan_time)

        payload = {
            "username": "NYC Sniper",
            "embeds": [embed],
        }

        try:
            async with self.session.post(self.webhook_url, json=payload) as resp:
                if resp.status == 204:
                    logger.info(f"Discord alert sent for {ticket.target_ticker}")
                    self._record_alert(ticket.target_ticker)
                    return True
                else:
                    body = await resp.text()
                    logger.error(f"Discord webhook failed: {resp.status} - {body}")
                    return False

        except asyncio.TimeoutError:
            logger.error("Discord webhook timed out")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"Discord webhook error: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error sending Discord alert: {e}")
            return False

    async def send_test_message(self) -> bool:
        """Send a test message to verify webhook configuration."""
        if not self.webhook_url:
            logger.warning("Cannot send test: DISCORD_WEBHOOK_URL not configured")
            return False

        if not self.session:
            logger.error("Notifier session not started")
            return False

        payload = {
            "username": "NYC Sniper",
            "embeds": [{
                "title": "NYC Sniper - Test Alert",
                "description": "Webhook is configured correctly!",
                "color": 0x00FF00,
                "footer": {"text": f"Sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"},
            }],
        }

        try:
            async with self.session.post(self.webhook_url, json=payload) as resp:
                if resp.status == 204:
                    logger.info("Discord test message sent successfully")
                    return True
                else:
                    body = await resp.text()
                    logger.error(f"Discord test failed: {resp.status} - {body}")
                    return False

        except Exception as e:
            logger.exception(f"Error sending Discord test: {e}")
            return False


async def test_discord():
    """Quick test for Discord webhook."""
    notifier = DiscordNotifier()
    await notifier.start()
    try:
        success = await notifier.send_test_message()
        if success:
            print("Discord webhook test PASSED")
        else:
            print("Discord webhook test FAILED")
    finally:
        await notifier.stop()


if __name__ == "__main__":
    asyncio.run(test_discord())
