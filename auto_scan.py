#!/usr/bin/env python3
"""
AUTO SCAN — Automated Edge Scanner with Discord Alerts

Runs edge_scanner_v2 at scheduled intervals, captures output,
and sends Discord webhook alerts when:
  1. A tradeable opportunity (90+ confidence) is found
  2. An existing position needs attention (morning check)
  3. Market conditions change significantly between scans

Designed to run via cron at optimal windows:
  - 6:00 AM ET  → Morning pre-settlement check
  - 10:00 AM ET → Market open (stale pricing = edge)
  - 3:00 PM ET  → Post-HRRR convergence (OPTIMAL window)
  - 10:00 PM ET → Overnight positioning (next-day setup)

Usage:
  python3 auto_scan.py                # Full scan + Discord alert
  python3 auto_scan.py --city NYC     # Single city
  python3 auto_scan.py --quiet        # Only alert if tradeable
  python3 auto_scan.py --dry-run      # Print what would be sent, don't send

Cron setup (add to crontab -e):
  0 6 * * *   cd /Users/miqadmin/Documents/limitless && /usr/bin/python3 auto_scan.py >> /tmp/auto_scan.log 2>&1
  0 10 * * *  cd /Users/miqadmin/Documents/limitless && /usr/bin/python3 auto_scan.py >> /tmp/auto_scan.log 2>&1
  0 15 * * *  cd /Users/miqadmin/Documents/limitless && /usr/bin/python3 auto_scan.py >> /tmp/auto_scan.log 2>&1
  0 22 * * *  cd /Users/miqadmin/Documents/limitless && /usr/bin/python3 auto_scan.py >> /tmp/auto_scan.log 2>&1
"""

import argparse
import asyncio
import io
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# Import the v2 scanner
from edge_scanner_v2 import (
    CITIES,
    MIN_CONFIDENCE_TO_TRADE,
    EnsembleV2,
    NWSData,
    Opportunity,
    analyze_opportunities_v2,
    fetch_ensemble_v2,
    fetch_kalshi_brackets,
    fetch_nws,
    get_entry_timing,
    kde_probability,
    compute_confidence_score,
    shorten_bracket_title,
)
from notifications import send_discord_embeds

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCAN_LOG_DIR = os.path.join(PROJECT_ROOT, "scan_logs")


def format_discord_alert(
    all_opps: list[Opportunity],
    city_summaries: list[dict],
    balance: float,
    scan_time: str,
    failed_cities: list[dict] = None,
) -> list[dict]:
    """Format scan results into Discord embed messages."""
    embeds = []
    failed_cities = failed_cities or []

    tradeable = [o for o in all_opps if o.confidence_score >= MIN_CONFIDENCE_TO_TRADE]

    # Header embed
    color = 0x00FF00 if tradeable else 0xFFAA00 if all_opps else 0xFF0000
    status = f"🎯 {len(tradeable)} TRADEABLE" if tradeable else f"👀 {len(all_opps)} opportunities (observe)"

    header_desc = (
        f"**{status}**\n"
        f"Balance: ${balance:.2f} | Cities: {len(city_summaries)}\n"
        f"Gate: {MIN_CONFIDENCE_TO_TRADE}+ confidence required"
    )

    if failed_cities:
        failed_names = [f["city"] for f in failed_cities]
        header_desc += f"\n⚠️ **{len(failed_cities)} city scan(s) failed:** {', '.join(failed_names)}"

    header = {
        "title": f"⚡ EDGE SCANNER v2.0 — {scan_time}",
        "description": header_desc,
        "color": color,
    }
    embeds.append(header)

    # Per-city summary
    for cs in city_summaries:
        city_text = (
            f"Ensemble: {cs['mean']:.1f}°F ±{cs['std']:.1f}° ({cs['members']} members)\n"
            f"NWS: {cs['nws_high']:.0f}°F | Physics: {cs['physics']:.1f}°F\n"
            f"Confidence: {cs['conf_label']} ({cs['conf_score']:.0f}/100)\n"
        )
        if cs.get("opps"):
            city_text += f"Opportunities: {len(cs['opps'])}\n"
            for opp in cs["opps"][:3]:  # Top 3 per city
                price = opp.yes_bid if opp.side == "yes" else (100 - opp.yes_ask)
                short = shorten_bracket_title(opp.bracket_title)
                icon = "🎯" if opp.confidence_score >= MIN_CONFIDENCE_TO_TRADE else "👀"
                city_text += (
                    f"{icon} {opp.side.upper()} {short} @ {price}¢ "
                    f"(KDE:{opp.kde_prob*100:.0f}% edge:{opp.edge_after_fees*100:+.0f}¢ "
                    f"conf:{opp.confidence_score:.0f})\n"
                )
        else:
            city_text += "No opportunities above threshold.\n"

        embeds.append({
            "title": f"📍 {cs['name']}",
            "description": city_text,
            "color": 0x00FF00 if any(o.confidence_score >= MIN_CONFIDENCE_TO_TRADE for o in cs.get("opps", [])) else 0x808080,
        })

    # Tradeable alert (if any)
    if tradeable:
        alert_text = "**🚨 ACTION REQUIRED — TRADEABLE SETUPS:**\n\n"
        for opp in tradeable:
            # Use bid+1 for maker pricing (0% fee)
            entry_price = opp.yes_bid + 1 if opp.side == "yes" else (100 - opp.yes_ask + 1)
            entry_price = min(entry_price, 50)  # Enforce max entry
            display_price = opp.yes_bid if opp.side == "yes" else (100 - opp.yes_ask)
            short = shorten_bracket_title(opp.bracket_title)
            cost = entry_price / 100 * opp.suggested_contracts
            alert_text += (
                f"**{opp.city} — {opp.side.upper()} {short} @ {display_price}¢**\n"
                f"KDE: {opp.kde_prob*100:.1f}% | Edge: {opp.edge_after_fees*100:+.1f}¢ | "
                f"Kelly: {opp.suggested_contracts} contracts\n"
                f"Confidence: {opp.confidence} ({opp.confidence_score:.0f}/100)\n"
                f"Rationale: {opp.rationale}\n"
                f"**Execute:** `python3 execute_trade.py {opp.ticker} {opp.side} {entry_price} {opp.suggested_contracts}`\n"
                f"Cost: ${cost:.2f} | Payout: ${opp.suggested_contracts:.2f}\n\n"
            )
        alert_text += "⚠️ Copy the command above and run in terminal to execute."
        embeds.append({
            "title": "🎯 TRADEABLE OPPORTUNITIES",
            "description": alert_text[:4096],
            "color": 0x00FF00,
        })

    return embeds


async def run_scan(city_filter: str = None, quiet: bool = False, dry_run: bool = False):
    """Run the full scan and send alerts."""
    now = datetime.now(ET)
    scan_time = now.strftime("%I:%M %p ET, %a %b %d")
    print(f"\n{'='*60}")
    print(f"  AUTO SCAN — {scan_time}")
    print(f"{'='*60}")

    # Get balance
    balance = 0.0
    try:
        from kalshi_client import fetch_balance_quick
        balance = await fetch_balance_quick()
    except Exception as e:
        print(f"  [WARN] Balance: {e}")

    cities_to_scan = {city_filter.upper(): CITIES[city_filter.upper()]} if city_filter else CITIES

    tomorrow = (datetime.now(ET) + timedelta(days=1)).date()
    target_date_str = tomorrow.isoformat()

    all_opps = []
    city_summaries = []
    failed_cities = []

    async with aiohttp.ClientSession() as session:
        for city_key in cities_to_scan:
            print(f"  Scanning {city_key}...", end=" ")

            try:
                ens_task = fetch_ensemble_v2(session, city_key, target_date_str)
                nws_task = fetch_nws(session, city_key, tomorrow)
                mkt_task = fetch_kalshi_brackets(session, city_key)

                results = await asyncio.gather(ens_task, nws_task, mkt_task, return_exceptions=True)

                # Check for exceptions in any of the three fetches
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    error_msgs = [f"{type(e).__name__}: {e}" for e in errors]
                    raise RuntimeError(f"Fetch errors: {'; '.join(error_msgs)}")

                ensemble, nws_data, brackets = results
                opps = analyze_opportunities_v2(city_key, ensemble, nws_data, brackets, balance)
                all_opps.extend(opps)

                conf_label, conf_score, _ = compute_confidence_score(ensemble, nws_data)

                # Save ensemble snapshot for backtest pipeline
                try:
                    from backtest_collector import save_ensemble_snapshot
                    snapshot_data = {
                        "mean": ensemble.mean,
                        "std": ensemble.std,
                        "total_count": ensemble.total_count,
                        "per_model_means": {mg.name: mg.mean for mg in ensemble.models},
                        "nws_forecast_high": nws_data.forecast_high,
                        "physics_high": nws_data.physics_high,
                        "conf_score": conf_score,
                    }
                    save_ensemble_snapshot(city_key, datetime.now(ET) + timedelta(days=1), snapshot_data)
                except Exception:
                    pass  # Non-critical — don't break scan if snapshot fails

                city_summaries.append({
                    "name": CITIES[city_key]["name"],
                    "key": city_key,
                    "mean": ensemble.mean,
                    "std": ensemble.std,
                    "members": ensemble.total_count,
                    "nws_high": nws_data.forecast_high,
                    "physics": nws_data.physics_high,
                    "conf_label": conf_label,
                    "conf_score": conf_score,
                    "opps": opps,
                })

                tradeable_count = sum(1 for o in opps if o.confidence_score >= MIN_CONFIDENCE_TO_TRADE)
                print(f"✓ {ensemble.total_count} members, {len(opps)} opps ({tradeable_count} tradeable)")

            except Exception as e:
                failed_cities.append({"city": city_key, "error": str(e)})
                print(f"✗ FAILED — {e}")
                continue

    if failed_cities:
        failed_names = [f["city"] for f in failed_cities]
        scanned_count = len(cities_to_scan) - len(failed_cities)
        print(f"\n  ⚠ {len(failed_cities)} city scan(s) failed: {', '.join(failed_names)}")
        print(f"  {scanned_count}/{len(cities_to_scan)} cities scanned successfully")

    # Summary
    tradeable = [o for o in all_opps if o.confidence_score >= MIN_CONFIDENCE_TO_TRADE]
    print(f"\n  Total: {len(all_opps)} opportunities, {len(tradeable)} tradeable")
    print(f"  Balance: ${balance:.2f}")

    # Save scan log
    os.makedirs(SCAN_LOG_DIR, exist_ok=True)
    log_file = os.path.join(SCAN_LOG_DIR, f"scan_{now.strftime('%Y%m%d_%H%M')}.txt")

    # Capture full v2 output to log
    buf = io.StringIO()
    with redirect_stdout(buf):
        from edge_scanner_v2 import print_summary_v2
        print_summary_v2(all_opps, balance)
    summary_output = buf.getvalue()

    with open(log_file, "w") as f:
        f.write(f"AUTO SCAN — {scan_time}\n")
        f.write(f"Balance: ${balance:.2f}\n")
        f.write(f"Opportunities: {len(all_opps)} total, {len(tradeable)} tradeable\n\n")
        for cs in city_summaries:
            f.write(f"{cs['name']}: {cs['mean']:.1f}°F ±{cs['std']:.1f}° | "
                    f"NWS: {cs['nws_high']:.0f}°F | Conf: {cs['conf_label']} ({cs['conf_score']:.0f})\n")
            for opp in cs["opps"]:
                price = opp.yes_bid if opp.side == "yes" else (100 - opp.yes_ask)
                short = shorten_bracket_title(opp.bracket_title)
                gate = "★" if opp.confidence_score >= MIN_CONFIDENCE_TO_TRADE else " "
                f.write(f"  {gate} {opp.side.upper()} {short} @ {price}¢ | "
                        f"KDE:{opp.kde_prob*100:.0f}% edge:{opp.edge_after_fees*100:+.0f}¢ "
                        f"conf:{opp.confidence_score:.0f}\n")
        f.write(f"\n{summary_output}")

    print(f"  Log saved: {log_file}")

    # Send Discord alert
    if quiet and not tradeable:
        print("  [QUIET] No tradeable opportunities — skipping Discord alert")
    else:
        embeds = format_discord_alert(all_opps, city_summaries, balance, scan_time, failed_cities)
        await send_discord_embeds(embeds, dry_run=dry_run, context="auto_scan")

    # Return results for programmatic use
    return {
        "scan_time": scan_time,
        "balance": balance,
        "total_opps": len(all_opps),
        "tradeable": len(tradeable),
        "opps": all_opps,
        "log_file": log_file,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Scan — Automated Edge Scanner with Discord Alerts")
    parser.add_argument("--city", type=str, default=None, help="City code (NYC, CHI, DEN, MIA, LAX)")
    parser.add_argument("--quiet", action="store_true", help="Only send Discord alert if tradeable setup found")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be sent, don't actually send")
    args = parser.parse_args()
    asyncio.run(run_scan(args.city, args.quiet, args.dry_run))
