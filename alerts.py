#!/usr/bin/env python3
"""
Atlas Alerts - Discord webhook notifications.

Sends alerts for:
- New signals detected
- Win rate milestones
- Bot started/stopped
- Errors and circuit breakers
"""

import json
import os
from datetime import datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK", "")
SIGNALS_FILE = Path("atlas_signals.jsonl")


class AlertManager:
    """Manages Discord webhook alerts."""

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url or WEBHOOK_URL
        self.last_signal_count = 0
        self.alerted_milestones = set()

    async def send(self, title: str, message: str, color: int = 0x00FF00):
        """Send a Discord embed message."""
        if not self.webhook_url:
            return False

        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Atlas Bot"}
        }

        payload = {"embeds": [embed]}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    return resp.status in (200, 204)
        except Exception as e:
            print(f"[ALERT ERROR] {e}")
            return False

    async def signal_alert(self, action: str, asset: str, edge: float, profit_est: float, velocity: float):
        """Alert for new signal detected."""
        emoji = "🟢" if "YES" in action else "🔴"

        message = (
            f"**Asset:** {asset}\n"
            f"**Action:** {action}\n"
            f"**Edge:** {edge*100:.1f}%\n"
            f"**Est. Profit:** ${profit_est:.2f}\n"
            f"**Velocity:** ${velocity:+.1f}/s"
        )

        await self.send(f"{emoji} New Signal: {action}", message, color=0x00FF00 if "YES" in action else 0xFF6600)

    async def bot_started(self, markets: int, mode: str):
        """Alert when bot starts."""
        message = (
            f"**Mode:** {mode}\n"
            f"**Markets:** {markets}\n"
            f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await self.send("🚀 Atlas Bot Started", message, color=0x00BFFF)

    async def bot_stopped(self, reason: str = "Manual stop"):
        """Alert when bot stops."""
        await self.send("🛑 Atlas Bot Stopped", f"**Reason:** {reason}", color=0xFF0000)

    async def circuit_breaker(self, drawdown: float, balance: float):
        """Alert when circuit breaker triggers."""
        message = (
            f"**Drawdown:** {drawdown*100:.1f}%\n"
            f"**Current Balance:** ${balance:.2f}\n"
            f"**Action:** Trading halted"
        )
        await self.send("⚠️ CIRCUIT BREAKER TRIGGERED", message, color=0xFF0000)

    async def kill_switch(self):
        """Alert when kill switch activated."""
        await self.send("🔴 KILL SWITCH ACTIVATED", "Bot shutting down immediately.", color=0xFF0000)

    async def milestone_alert(self, signals: int, wins: int, losses: int):
        """Alert for win rate milestones."""
        if signals < 10:
            return

        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        # Check for milestones
        milestones = [
            (20, "20 signals reached!"),
            (50, "50 signals - getting statistically significant!"),
            (100, "100 signals - ready for analysis!"),
        ]

        for count, msg in milestones:
            if signals >= count and count not in self.alerted_milestones:
                self.alerted_milestones.add(count)
                message = (
                    f"**Total Signals:** {signals}\n"
                    f"**Wins:** {wins}\n"
                    f"**Losses:** {losses}\n"
                    f"**Win Rate:** {win_rate:.1f}%"
                )

                color = 0x00FF00 if win_rate >= 70 else 0xFFFF00 if win_rate >= 50 else 0xFF0000
                await self.send(f"📊 Milestone: {msg}", message, color=color)

    async def error_alert(self, error: str):
        """Alert for errors."""
        await self.send("❌ Error", f"```{error[:500]}```", color=0xFF0000)


# Singleton instance
alerts = AlertManager()


async def test_webhook():
    """Test the Discord webhook."""
    print("Testing Discord webhook...")
    success = await alerts.send(
        "🧪 Test Alert",
        "Atlas alerts are working!\n\nYou'll receive notifications for:\n"
        "• New signals\n"
        "• Win rate milestones\n"
        "• Bot status changes\n"
        "• Errors",
        color=0x00FF00
    )
    if success:
        print("✓ Webhook test successful! Check Discord.")
    else:
        print("✗ Webhook test failed. Check your URL.")
    return success


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_webhook())
