#!/usr/bin/env python3
"""
Weather Bot Alerts - Discord webhook notifications for Kalshi weather trading.

Sends alerts for:
- Daily forecast summaries
- Trading opportunities found
- Paper/live trades executed
- Settlement results (win/loss)
- Model consensus updates
- Win rate milestones
"""

import os
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK", "")


class AlertManager:
    """Manages Discord webhook alerts for weather trading."""

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url or WEBHOOK_URL
        self.alerted_milestones = set()

        # Track stats
        self.trades_today = 0
        self.wins = 0
        self.losses = 0

    async def send(self, title: str, message: str, color: int = 0x00FF00):
        """Send a Discord embed message."""
        if not self.webhook_url:
            return False

        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "Kalshi Weather Bot"}
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

    # -------------------------------------------------------------------------
    # BOT STATUS
    # -------------------------------------------------------------------------

    async def bot_started(self, mode: str, balance: float = 0):
        """Alert when bot starts."""
        message = (
            f"**Mode:** {mode}\n"
            f"**Market:** KXHIGHNY (NYC High Temp)\n"
            f"**Kalshi Balance:** ${balance:.2f}\n"
            f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await self.send("🌤️ Weather Bot Started", message, color=0x00BFFF)

    async def bot_stopped(self, runtime_min: float, trades: int, paper_pnl: float):
        """Alert when bot stops."""
        message = (
            f"**Runtime:** {runtime_min:.1f} minutes\n"
            f"**Trades:** {trades}\n"
            f"**Paper P&L:** ${paper_pnl:.2f}"
        )
        await self.send("🛑 Weather Bot Stopped", message, color=0xFF0000)

    # -------------------------------------------------------------------------
    # FORECASTS & OPPORTUNITIES
    # -------------------------------------------------------------------------

    async def daily_forecast(
        self,
        date: str,
        gfs_temp: float,
        ecmwf_temp: float,
        bracket: str,
        confidence: float,
    ):
        """Send daily forecast summary."""
        avg = (gfs_temp + ecmwf_temp) / 2
        divergence = abs(gfs_temp - ecmwf_temp)

        agree_emoji = "✅" if divergence <= 2 else "⚠️"

        message = (
            f"**Date:** {date}\n"
            f"**GFS:** {gfs_temp:.1f}°F\n"
            f"**ECMWF:** {ecmwf_temp:.1f}°F\n"
            f"**Average:** {avg:.1f}°F\n"
            f"**Bracket:** {bracket}\n"
            f"**Divergence:** {divergence:.1f}°F {agree_emoji}\n"
            f"**Confidence:** {confidence:.0%}"
        )

        color = 0x00FF00 if divergence <= 2 else 0xFFFF00
        await self.send(f"🌡️ Forecast: {date}", message, color=color)

    async def opportunity_found(
        self,
        date: str,
        gfs_temp: float,
        ecmwf_temp: float,
        model_bracket: str,
        market_price: float,
        market_favorite: str,
        edge: float,
        potential_return: float,
    ):
        """Alert when trading opportunity is found."""
        avg = (gfs_temp + ecmwf_temp) / 2

        message = (
            f"**Date:** {date}\n"
            f"\n**Models:**\n"
            f"├ GFS: {gfs_temp:.1f}°F\n"
            f"├ ECMWF: {ecmwf_temp:.1f}°F\n"
            f"└ Avg: {avg:.1f}°F → **{model_bracket}**\n"
            f"\n**Market:**\n"
            f"├ Our bracket: {model_bracket} @ **{market_price:.0%}**\n"
            f"└ Favorite: {market_favorite}\n"
            f"\n**Edge:** {edge:.0%}\n"
            f"**Potential:** {potential_return:.1f}x return"
        )

        # Color based on edge strength
        if edge >= 0.50:
            color = 0x00FF00  # Green - strong
        elif edge >= 0.30:
            color = 0x90EE90  # Light green - good
        else:
            color = 0xFFFF00  # Yellow - moderate

        await self.send(f"🎯 OPPORTUNITY: {date}", message, color=color)

    # -------------------------------------------------------------------------
    # TRADES
    # -------------------------------------------------------------------------

    async def paper_trade(
        self,
        date: str,
        bracket: str,
        contracts: int,
        entry_price: float,
        max_profit: float,
        max_loss: float,
    ):
        """Alert for paper trade recorded."""
        message = (
            f"**Date:** {date}\n"
            f"**Bracket:** {bracket}\n"
            f"**Contracts:** {contracts}\n"
            f"**Entry:** {entry_price:.0%}\n"
            f"\n**If Win:** +${max_profit:.2f}\n"
            f"**If Lose:** -${max_loss:.2f}"
        )
        await self.send("📝 Paper Trade Recorded", message, color=0x87CEEB)

    async def live_trade(
        self,
        date: str,
        ticker: str,
        contracts: int,
        price: float,
        order_id: str,
    ):
        """Alert for live trade executed."""
        cost = contracts * price
        potential = contracts * (1 - price)

        message = (
            f"**Date:** {date}\n"
            f"**Ticker:** `{ticker}`\n"
            f"**Contracts:** {contracts} YES\n"
            f"**Price:** {price:.0%}\n"
            f"**Cost:** ${cost:.2f}\n"
            f"**Potential Profit:** ${potential:.2f}\n"
            f"**Order ID:** `{order_id[:8]}...`"
        )
        await self.send("💰 LIVE TRADE EXECUTED", message, color=0x00FF00)

    # -------------------------------------------------------------------------
    # SETTLEMENTS
    # -------------------------------------------------------------------------

    async def settlement_result(
        self,
        date: str,
        predicted_bracket: str,
        actual_temp: float,
        actual_bracket: str,
        won: bool,
        pnl: float,
    ):
        """Alert for trade settlement result."""
        emoji = "✅" if won else "❌"
        result = "WIN" if won else "LOSS"

        message = (
            f"**Date:** {date}\n"
            f"**Predicted:** {predicted_bracket}\n"
            f"**Actual Temp:** {actual_temp:.1f}°F\n"
            f"**Actual Bracket:** {actual_bracket}\n"
            f"\n**Result:** {emoji} **{result}**\n"
            f"**P&L:** ${pnl:+.2f}"
        )

        color = 0x00FF00 if won else 0xFF0000
        await self.send(f"{emoji} Settlement: {date}", message, color=color)

        # Update stats
        if won:
            self.wins += 1
        else:
            self.losses += 1

    async def daily_summary(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        paper_pnl: float,
        upcoming_trades: int,
    ):
        """Send daily performance summary."""
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        message = (
            f"**Total Trades:** {total_trades}\n"
            f"**Wins:** {wins} ✅\n"
            f"**Losses:** {losses} ❌\n"
            f"**Win Rate:** {win_rate:.1f}%\n"
            f"**Paper P&L:** ${paper_pnl:+.2f}\n"
            f"\n**Pending:** {upcoming_trades} trades awaiting settlement"
        )

        color = 0x00FF00 if win_rate >= 70 else 0xFFFF00 if win_rate >= 50 else 0xFF0000
        await self.send("📊 Daily Summary", message, color=color)

    # -------------------------------------------------------------------------
    # MILESTONES & ALERTS
    # -------------------------------------------------------------------------

    async def milestone(self, trades: int, wins: int, losses: int):
        """Alert for win rate milestones."""
        if trades < 5:
            return

        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        milestones = [
            (5, "5 trades - Early results!"),
            (10, "10 trades - Getting data!"),
            (20, "20 trades - Statistically meaningful!"),
            (50, "50 trades - Strong sample size!"),
        ]

        for count, msg in milestones:
            if trades >= count and count not in self.alerted_milestones:
                self.alerted_milestones.add(count)

                emoji = "🏆" if win_rate >= 75 else "📈" if win_rate >= 60 else "📊"

                message = (
                    f"**Total Trades:** {trades}\n"
                    f"**Wins:** {wins}\n"
                    f"**Losses:** {losses}\n"
                    f"**Win Rate:** {win_rate:.1f}%\n"
                    f"\n**Target:** 75-85%"
                )

                color = 0x00FF00 if win_rate >= 75 else 0xFFFF00 if win_rate >= 60 else 0xFF6600
                await self.send(f"{emoji} Milestone: {msg}", message, color=color)

    async def model_divergence_warning(self, date: str, gfs: float, ecmwf: float):
        """Alert when models disagree significantly."""
        divergence = abs(gfs - ecmwf)

        message = (
            f"**Date:** {date}\n"
            f"**GFS:** {gfs:.1f}°F\n"
            f"**ECMWF:** {ecmwf:.1f}°F\n"
            f"**Divergence:** {divergence:.1f}°F\n"
            f"\n⚠️ Skipping trade - models disagree"
        )
        await self.send("⚠️ Model Divergence", message, color=0xFFFF00)

    async def low_balance_warning(self, balance: float):
        """Alert when Kalshi balance is low."""
        message = (
            f"**Current Balance:** ${balance:.2f}\n"
            f"**Minimum Required:** $10.00\n"
            f"\n💡 Fund your Kalshi account to enable live trading"
        )
        await self.send("💰 Low Balance Warning", message, color=0xFF6600)

    async def error(self, error: str):
        """Alert for errors."""
        await self.send("❌ Error", f"```{error[:500]}```", color=0xFF0000)


# Singleton instance
alerts = AlertManager()


async def test_webhook():
    """Test the Discord webhook."""
    print("Testing Discord webhook...")

    success = await alerts.send(
        "🌤️ Weather Bot Test",
        "Weather alerts are working!\n\n"
        "You'll receive notifications for:\n"
        "• Daily forecasts (GFS + ECMWF)\n"
        "• Trading opportunities\n"
        "• Paper/live trades\n"
        "• Settlement results\n"
        "• Win rate milestones",
        color=0x00BFFF
    )

    if success:
        print("✓ Webhook test successful! Check Discord.")
    else:
        print("✗ Webhook test failed. Check DISCORD_WEBHOOK in .env")

    return success


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_webhook())
