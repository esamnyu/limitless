#!/usr/bin/env python3
"""
Atlas Live Dashboard - Real-time terminal monitoring with price charts.

Usage:
    python3 dashboard.py           # Monitor all assets
    python3 dashboard.py --asset BTC  # Monitor specific asset
"""

import argparse
import asyncio
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import ccxt.pro as ccxtpro
import plotext as plt
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SIGNALS_FILE = Path("atlas_signals.jsonl")
VELOCITY_THRESHOLD = 25.0


class Dashboard:
    """Real-time terminal dashboard for Atlas bot."""

    def __init__(self, asset: str = "BTC"):
        self.asset = asset
        self.console = Console()

        # Price history (5 minutes at ~1 sample/sec)
        self.prices: deque[tuple[float, float]] = deque(maxlen=300)
        self.velocities: deque[tuple[float, float]] = deque(maxlen=300)

        # Current state
        self.current_price = 0.0
        self.current_velocity = 0.0
        self.signal_times: list[float] = []  # Timestamps of signals

        # Stats
        self.start_time = time.time()
        self.last_signal_count = 0

    def load_signals(self) -> list[dict]:
        """Load signals from file."""
        if not SIGNALS_FILE.exists():
            return []

        signals = []
        try:
            with open(SIGNALS_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sig = json.loads(line)
                        # Filter by asset
                        market = sig.get("market", "").lower()
                        if self.asset.lower() in market:
                            signals.append(sig)
        except Exception:
            pass

        return signals

    def get_signal_times(self) -> list[float]:
        """Get timestamps of signals for chart markers."""
        signals = self.load_signals()
        times = []
        for sig in signals:
            try:
                ts = sig.get("ts", "")
                dt = datetime.fromisoformat(ts)
                times.append(dt.timestamp())
            except:
                pass
        return times

    def make_chart(self) -> str:
        """Generate ASCII price chart with signal markers."""
        if len(self.prices) < 2:
            return "Waiting for price data..."

        # Extract data
        times = [t for t, _ in self.prices]
        prices = [p for _, p in self.prices]

        # Get signal times
        signal_times = self.get_signal_times()

        # Clear and configure plot
        plt.clf()
        plt.plotsize(60, 15)
        plt.theme("dark")

        # Plot price line
        plt.plot(times, prices, label=f"{self.asset} Price", color="cyan")

        # Mark signals on chart
        for st in signal_times:
            if times[0] <= st <= times[-1]:
                # Find closest price point
                idx = min(range(len(times)), key=lambda i: abs(times[i] - st))
                plt.scatter([times[idx]], [prices[idx]], marker="x", color="red")

        # Configure axes
        plt.xlabel("Time")
        plt.ylabel("Price ($)")

        # Get string output
        return plt.build()

    def make_velocity_bar(self) -> Text:
        """Create velocity indicator bar."""
        vel = abs(self.current_velocity)
        threshold = VELOCITY_THRESHOLD
        pct = min(vel / threshold, 2.0)  # Cap at 200%

        bar_width = 30
        filled = int(pct * bar_width / 2)

        # Color based on intensity
        if vel < threshold * 0.5:
            color = "green"
        elif vel < threshold:
            color = "yellow"
        else:
            color = "red bold"

        bar = "█" * filled + "░" * (bar_width - filled)
        direction = "+" if self.current_velocity > 0 else "-" if self.current_velocity < 0 else " "

        text = Text()
        text.append(f"Velocity: ${direction}{vel:.1f}/s  [", style="white")
        text.append(bar, style=color)
        text.append(f"]  threshold: ${threshold:.0f}/s", style="white")

        if vel >= threshold:
            text.append("  TRIGGER!", style="red bold blink")

        return text

    def make_signals_table(self) -> Table:
        """Create recent signals table."""
        signals = self.load_signals()

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Time", width=10)
        table.add_column("Action", width=10)
        table.add_column("Edge", width=8, justify="right")
        table.add_column("Profit", width=8, justify="right")

        for sig in signals[-5:]:  # Last 5 signals
            ts = sig.get("ts", "")[:19]
            time_str = ts[11:19] if len(ts) > 11 else ts
            action = sig.get("action", "")
            edge = sig.get("edge", 0) * 100
            profit = sig.get("profit_est", 0)

            action_style = "green" if "YES" in action else "red"
            table.add_row(
                time_str,
                Text(action, style=action_style),
                f"{edge:.0f}%",
                f"${profit:.2f}"
            )

        return table

    def make_status_bar(self) -> Text:
        """Create status bar."""
        uptime = time.time() - self.start_time
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)

        signals = self.load_signals()
        signal_count = len(signals)

        text = Text()
        text.append(f"Signals: {signal_count}", style="cyan")
        text.append(" | ", style="dim")
        text.append(f"Uptime: {hours}h {minutes}m", style="white")
        text.append(" | ", style="dim")
        text.append("Press Ctrl+C to exit", style="dim")

        return text

    def make_layout(self) -> Layout:
        """Create the full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="chart", size=18),
            Layout(name="velocity", size=3),
            Layout(name="signals", size=10),
            Layout(name="footer", size=3),
        )

        # Header
        price_text = Text()
        price_text.append("ATLAS LIVE MONITOR", style="bold cyan")
        price_text.append(f"                    {self.asset}: ", style="white")
        price_text.append(f"${self.current_price:,.2f}", style="bold green")

        layout["header"].update(Panel(price_text, style="cyan"))

        # Chart
        chart_str = self.make_chart()
        layout["chart"].update(Panel(chart_str, title=f"{self.asset} Price (5min)", border_style="cyan"))

        # Velocity
        layout["velocity"].update(Panel(self.make_velocity_bar(), border_style="cyan"))

        # Signals
        layout["signals"].update(Panel(
            self.make_signals_table(),
            title="Recent Signals",
            border_style="cyan"
        ))

        # Footer
        layout["footer"].update(Panel(self.make_status_bar(), style="dim"))

        return layout

    async def run(self):
        """Main run loop."""
        exchange = ccxtpro.kraken({"enableRateLimit": True})

        symbol_map = {
            "BTC": "BTC/USD",
            "ETH": "ETH/USD",
            "SOL": "SOL/USD",
        }
        symbol = symbol_map.get(self.asset, "BTC/USD")

        self.console.print(f"[cyan]Starting dashboard for {self.asset}...[/cyan]")

        try:
            with Live(self.make_layout(), refresh_per_second=2, console=self.console) as live:
                while True:
                    try:
                        ticker = await exchange.watch_ticker(symbol)
                        now = time.time()
                        price = ticker.get("last", 0)

                        if price > 0:
                            self.current_price = price
                            self.prices.append((now, price))

                            # Calculate velocity
                            if len(self.prices) >= 2:
                                old_t, old_p = self.prices[-min(10, len(self.prices))]
                                dt = now - old_t
                                if dt > 0.1:
                                    self.current_velocity = (price - old_p) / dt
                                    self.velocities.append((now, self.current_velocity))

                            # Update display
                            live.update(self.make_layout())

                    except Exception as e:
                        await asyncio.sleep(1)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Dashboard stopped.[/yellow]")
        finally:
            await exchange.close()


async def main():
    parser = argparse.ArgumentParser(description="Atlas Live Dashboard")
    parser.add_argument("--asset", default="BTC", choices=["BTC", "ETH", "SOL"],
                        help="Asset to monitor")
    args = parser.parse_args()

    dashboard = Dashboard(asset=args.asset)
    await dashboard.run()


if __name__ == "__main__":
    asyncio.run(main())
