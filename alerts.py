#!/usr/bin/env python3
"""Discord webhook alerts for Kalshi weather trading."""

import os
from datetime import datetime, timezone
import aiohttp
from dotenv import load_dotenv

load_dotenv()


class AlertManager:
    """Manages Discord webhook alerts for weather trading."""

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK", "")
        self.alerted_milestones = set()
        self.wins = self.losses = 0
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a persistent aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=5, ttl_dns_cache=300),
                timeout=aiohttp.ClientTimeout(total=10),
            )
        return self._session

    async def close(self):
        """Close the persistent session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def send(self, title: str, message: str, color: int = 0x00FF00):
        """Send a Discord embed message."""
        if not self.webhook_url:
            return False
        try:
            session = await self._get_session()
            async with session.post(self.webhook_url, json={"embeds": [{
                "title": title, "description": message, "color": color,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "Kalshi Weather Bot"}
            }]}) as resp:
                return resp.status in (200, 204)
        except:
            return False

    async def bot_started(self, mode: str, balance: float = 0):
        await self.send("Weather Bot Started",
            f"**Mode:** {mode}\n**Balance:** ${balance:.2f}\n**Time:** {datetime.now():%Y-%m-%d %H:%M}", 0x00BFFF)

    async def bot_stopped(self, runtime_min: float, trades: int, paper_pnl: float):
        await self.send("Weather Bot Stopped",
            f"**Runtime:** {runtime_min:.1f}min\n**Trades:** {trades}\n**P&L:** ${paper_pnl:.2f}", 0xFF0000)

    async def opportunity_found(self, date: str, gfs_temp: float, ecmwf_temp: float,
                                 model_bracket: str, market_price: float, market_favorite: str,
                                 edge: float, potential_return: float):
        avg = (gfs_temp + ecmwf_temp) / 2
        color = 0x00FF00 if edge >= 0.50 else 0x90EE90 if edge >= 0.30 else 0xFFFF00
        await self.send(f"OPPORTUNITY: {date}",
            f"**GFS:** {gfs_temp:.1f}°F | **ECMWF:** {ecmwf_temp:.1f}°F | **Avg:** {avg:.1f}°F\n"
            f"**Bracket:** {model_bracket} @ {market_price:.0%} | **Edge:** {edge:.0%}", color)

    async def paper_trade(self, date: str, bracket: str, contracts: int,
                          entry_price: float, max_profit: float, max_loss: float):
        await self.send("Paper Trade",
            f"**{date}** | {bracket} | {contracts} contracts @ {entry_price:.0%}\n"
            f"Win: +${max_profit:.2f} | Loss: -${max_loss:.2f}", 0x87CEEB)

    async def live_trade(self, date: str, ticker: str, contracts: int, price: float, order_id: str):
        await self.send("LIVE TRADE",
            f"**{ticker}**\n{contracts} YES @ {price:.0%}\nOrder: {order_id[:8]}...", 0x00FF00)

    async def settlement_result(self, date: str, predicted_bracket: str, actual_temp: float,
                                 actual_bracket: str, won: bool, pnl: float):
        self.wins += won
        self.losses += not won
        await self.send(f"{'WIN' if won else 'LOSS'}: {date}",
            f"Predicted: {predicted_bracket} | Actual: {actual_bracket} ({actual_temp:.1f}°F)\n**P&L:** ${pnl:+.2f}",
            0x00FF00 if won else 0xFF0000)

    async def milestone(self, trades: int, wins: int, losses: int):
        if trades < 5:
            return
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        for count, msg in [(5, "5 trades"), (10, "10 trades"), (20, "20 trades"), (50, "50 trades")]:
            if trades >= count and count not in self.alerted_milestones:
                self.alerted_milestones.add(count)
                color = 0x00FF00 if win_rate >= 75 else 0xFFFF00 if win_rate >= 60 else 0xFF6600
                await self.send(f"Milestone: {msg}",
                    f"**Wins:** {wins} | **Losses:** {losses} | **Win Rate:** {win_rate:.1f}%", color)

    async def error(self, error: str):
        await self.send("Error", f"```{error[:500]}```", 0xFF0000)


alerts = AlertManager()

if __name__ == "__main__":
    import asyncio
    asyncio.run(alerts.send("Test", "Webhook test!", 0x00BFFF))
