#!/usr/bin/env python3
"""
Atlas Analytics - Track paper trading performance and win rate.

Usage:
    python3 analytics.py           # Show current stats
    python3 analytics.py --live    # Continuous monitoring
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import aiohttp

SIGNALS_FILE = Path("atlas_signals.jsonl")
API_URL = "https://api.limitless.exchange"


def load_signals() -> list[dict]:
    """Load all signals from JSONL file."""
    if not SIGNALS_FILE.exists():
        return []

    signals = []
    with open(SIGNALS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    signals.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return signals


async def check_market_resolution(session: aiohttp.ClientSession, slug: str) -> dict:
    """Check if a market has resolved and what the outcome was."""
    try:
        async with session.get(f"{API_URL}/markets/{slug}") as resp:
            if resp.status != 200:
                return {"status": "unknown", "error": f"HTTP {resp.status}"}

            data = await resp.json()

            # Check resolution status
            resolved = data.get("resolved", False)
            expired = data.get("expired", False)

            if resolved:
                # Get winning outcome
                winning_index = data.get("winningIndex")
                if winning_index == 0:
                    return {"status": "resolved", "winner": "NO"}
                elif winning_index == 1:
                    return {"status": "resolved", "winner": "YES"}
                else:
                    return {"status": "resolved", "winner": "unknown"}
            elif expired:
                return {"status": "expired_unresolved"}
            else:
                # Still active - check time to expiry
                exp_ts = data.get("expirationTimestamp", 0) / 1000
                ttl_min = (exp_ts - time.time()) / 60
                return {"status": "active", "ttl_min": ttl_min}

    except Exception as e:
        return {"status": "error", "error": str(e)}


async def analyze_signals(signals: list[dict]) -> dict:
    """Analyze all signals and calculate win rate."""
    if not signals:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "win_rate": 0.0,
            "by_asset": {},
            "by_hour": {},
        }

    results = {
        "total": len(signals),
        "wins": 0,
        "losses": 0,
        "pending": 0,
        "unknown": 0,
        "details": [],
        "by_asset": {},
        "by_hour": {},
        "total_edge": 0.0,
        "total_profit_est": 0.0,
    }

    async with aiohttp.ClientSession() as session:
        for sig in signals:
            slug = sig.get("market", "")
            action = sig.get("action", "")
            edge = sig.get("edge", 0)
            profit_est = sig.get("profit_est", 0)
            ts = sig.get("ts", "")

            # Parse timestamp for hour analysis
            try:
                dt = datetime.fromisoformat(ts)
                hour = dt.hour
                results["by_hour"][hour] = results["by_hour"].get(hour, 0) + 1
            except:
                hour = None

            # Determine asset from slug
            asset = "BTC"
            if "eth" in slug.lower():
                asset = "ETH"
            elif "sol" in slug.lower():
                asset = "SOL"

            if asset not in results["by_asset"]:
                results["by_asset"][asset] = {"total": 0, "wins": 0, "losses": 0, "pending": 0}
            results["by_asset"][asset]["total"] += 1

            results["total_edge"] += edge
            results["total_profit_est"] += profit_est

            # Check resolution
            resolution = await check_market_resolution(session, slug)

            detail = {
                "ts": ts,
                "asset": asset,
                "action": action,
                "edge": edge,
                "profit_est": profit_est,
                "resolution": resolution,
                "outcome": "pending",
            }

            if resolution["status"] == "resolved":
                winner = resolution.get("winner", "")

                # Determine if we won
                if action == "BUY_YES" and winner == "YES":
                    detail["outcome"] = "WIN"
                    results["wins"] += 1
                    results["by_asset"][asset]["wins"] += 1
                elif action == "BUY_NO" and winner == "NO":
                    detail["outcome"] = "WIN"
                    results["wins"] += 1
                    results["by_asset"][asset]["wins"] += 1
                elif winner in ["YES", "NO"]:
                    detail["outcome"] = "LOSS"
                    results["losses"] += 1
                    results["by_asset"][asset]["losses"] += 1
                else:
                    detail["outcome"] = "unknown"
                    results["unknown"] += 1
            elif resolution["status"] == "active":
                detail["outcome"] = "pending"
                detail["ttl_min"] = resolution.get("ttl_min", 0)
                results["pending"] += 1
                results["by_asset"][asset]["pending"] += 1
            else:
                detail["outcome"] = "unknown"
                results["unknown"] += 1

            results["details"].append(detail)

    # Calculate win rate (excluding pending)
    decided = results["wins"] + results["losses"]
    results["win_rate"] = (results["wins"] / decided * 100) if decided > 0 else 0.0
    results["avg_edge"] = results["total_edge"] / len(signals) if signals else 0

    return results


def print_report(results: dict):
    """Print formatted analytics report."""
    print("\n" + "=" * 60)
    print("               ATLAS PAPER TRADING ANALYTICS")
    print("=" * 60)

    print(f"\n  Total Signals:    {results['total']}")
    print(f"  Resolved:         {results['wins'] + results['losses']}")
    print(f"  Pending:          {results['pending']}")

    if results['wins'] + results['losses'] > 0:
        print(f"\n  Wins:             {results['wins']}")
        print(f"  Losses:           {results['losses']}")
        print(f"  WIN RATE:         {results['win_rate']:.1f}%")
    else:
        print(f"\n  No resolved trades yet")

    print(f"\n  Avg Edge:         {results.get('avg_edge', 0)*100:.1f}%")
    print(f"  Est. Total Profit: ${results.get('total_profit_est', 0):.2f}")

    # By asset
    if results["by_asset"]:
        print("\n  By Asset:")
        for asset, stats in results["by_asset"].items():
            decided = stats["wins"] + stats["losses"]
            wr = (stats["wins"] / decided * 100) if decided > 0 else 0
            print(f"    {asset}: {stats['total']} signals, {stats['wins']}W/{stats['losses']}L ({wr:.0f}%), {stats['pending']} pending")

    # By hour
    if results["by_hour"]:
        print("\n  By Hour (UTC):")
        for hour in sorted(results["by_hour"].keys()):
            count = results["by_hour"][hour]
            bar = "█" * count
            print(f"    {hour:02d}:00  {bar} ({count})")

    # Recent signals
    if results.get("details"):
        print("\n  Recent Signals:")
        for detail in results["details"][-5:]:
            ts_short = detail["ts"][11:19] if len(detail["ts"]) > 19 else detail["ts"]
            outcome = detail["outcome"]
            if outcome == "WIN":
                icon = "✓"
            elif outcome == "LOSS":
                icon = "✗"
            else:
                icon = "○"
                if "ttl_min" in detail:
                    outcome = f"pending ({detail['ttl_min']:.0f}m)"

            print(f"    {icon} {ts_short} {detail['asset']} {detail['action']} "
                  f"edge:{detail['edge']*100:.0f}% → {outcome}")

    print("\n" + "=" * 60)

    # Recommendation
    total = results['total']
    win_rate = results['win_rate']

    print("\n  RECOMMENDATION:")
    if total < 20:
        print(f"    Need more data ({total}/20 minimum signals)")
    elif total < 50:
        print(f"    Collecting data... ({total}/50 for confidence)")
    elif total < 100:
        if win_rate >= 70:
            print(f"    Looking good! ({total}/100 signals, {win_rate:.0f}% WR)")
            print(f"    Consider micro-live after 100 signals")
        elif win_rate >= 50:
            print(f"    Marginal ({win_rate:.0f}% WR) - tune parameters")
        else:
            print(f"    Poor ({win_rate:.0f}% WR) - strategy needs work")
    else:
        if win_rate >= 70:
            print(f"    READY for micro-live testing ($10-20 positions)")
        elif win_rate >= 60:
            print(f"    Borderline ({win_rate:.0f}% WR) - proceed with caution")
        else:
            print(f"    NOT READY ({win_rate:.0f}% WR) - fix strategy first")

    print()


async def live_monitor(interval: int = 60):
    """Continuously monitor and display stats."""
    print("Live monitoring... (Ctrl+C to stop)\n")

    while True:
        signals = load_signals()
        results = await analyze_signals(signals)

        # Clear screen and print
        print("\033[2J\033[H", end="")  # Clear screen
        print_report(results)
        print(f"  [Auto-refresh in {interval}s]")

        await asyncio.sleep(interval)


async def main():
    parser = argparse.ArgumentParser(description="Atlas Paper Trading Analytics")
    parser.add_argument("--live", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval (seconds)")
    args = parser.parse_args()

    if args.live:
        await live_monitor(args.interval)
    else:
        signals = load_signals()
        results = await analyze_signals(signals)
        print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
