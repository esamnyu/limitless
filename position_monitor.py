#!/usr/bin/env python3
"""
POSITION MONITOR — Automated take-profit and exit management.

Since Kalshi has NO native take-profit or stop-loss orders, this script
polls positions and places sell orders when exit conditions are met.

EXIT RULES:
  1. FREEROLL:    When price doubles (2x entry), sell half to recover cost basis
  2. EFFICIENCY:  When price hits 90¢, sell everything (90¢ now > $1 tomorrow)
  3. THESIS BREAK: When confidence drops below 40, alert to sell everything

Run via cron every 5 minutes when positions are open:
  */5 * * * * cd /Users/miqadmin/Documents/limitless && python3 position_monitor.py >> /tmp/position_monitor.log 2>&1

Or run manually:
  python3 position_monitor.py             # Check all positions
  python3 position_monitor.py --once      # Single check, no loop
  python3 position_monitor.py --status    # Show positions only
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from kalshi_client import KalshiClient
from position_store import load_positions, save_positions, position_transaction
from notifications import send_discord_alert

ET = ZoneInfo("America/New_York")

# Exit thresholds (single source of truth: config.py)
from config import (
    FREEROLL_MULTIPLIER,
    CAPITAL_EFFICIENCY_THRESHOLD_CENTS as EFFICIENCY_EXIT_CENTS,
    TRAILING_OFFSET_CENTS,
    SETTLEMENT_HOLD_THRESHOLD_CENTS as SETTLEMENT_HOLD_THRESHOLD,
    SETTLEMENT_HOUR_ET,
    SETTLEMENT_WINDOW_HOURS,
)

MIN_TRAILING_FLOOR = 0.0  # Never trail below entry price

# How long to wait for a limit sell order to fill before re-evaluating
PENDING_SELL_EXPIRY_MINUTES = 30


async def _check_pending_sells(positions: list, client: KalshiClient, now: datetime) -> list[str]:
    """Check positions with pending_sell status — confirm fill or cancel stale orders."""
    actions = []
    for pos in positions:
        if pos.get("status") != "pending_sell":
            continue

        ticker = pos["ticker"]
        sell_order_id = pos.get("sell_order_id", "")
        sell_placed_at = pos.get("sell_placed_at", "")

        # Check if the sell order has filled by querying Kalshi positions
        api_positions = await client.get_positions()
        api_qty = 0
        for ap in api_positions:
            if ap.get("ticker") == ticker:
                api_qty = abs(ap.get("position", 0))
                break

        expected_remaining = pos.get("_pending_remaining_qty", 0)

        if api_qty <= expected_remaining:
            # Sell filled — mark closed or update quantity
            if expected_remaining == 0:
                pos["status"] = "closed"
                pos["notes"].append(f"{now.isoformat()}: Sell order filled — position closed")
                actions.append(f"CONFIRMED FILL: {ticker} sell order filled, position closed")
            else:
                pos["status"] = "open"
                pos["contracts"] = expected_remaining
                pos.pop("sell_order_id", None)
                pos.pop("sell_placed_at", None)
                pos.pop("_pending_remaining_qty", None)
                pos["notes"].append(f"{now.isoformat()}: Partial sell filled — {expected_remaining} contracts remain")
                actions.append(f"PARTIAL FILL: {ticker} — {expected_remaining} contracts remain open")
        else:
            # Sell hasn't filled — check if stale
            if sell_placed_at:
                placed = datetime.fromisoformat(sell_placed_at)
                elapsed_min = (now - placed).total_seconds() / 60
                if elapsed_min > PENDING_SELL_EXPIRY_MINUTES:
                    # Cancel stale order and revert to open
                    if sell_order_id:
                        await client.cancel_order(sell_order_id)
                    pos["status"] = "open"
                    pos["contracts"] = pos.get("_pre_sell_qty", pos["contracts"])
                    pos.pop("sell_order_id", None)
                    pos.pop("sell_placed_at", None)
                    pos.pop("_pending_remaining_qty", None)
                    pos.pop("_pre_sell_qty", None)
                    pos["notes"].append(
                        f"{now.isoformat()}: Sell order unfilled after {elapsed_min:.0f}min — cancelled, reverted to open"
                    )
                    actions.append(f"STALE ORDER: {ticker} sell cancelled after {elapsed_min:.0f}min — reverted to open")
                    await send_discord_alert(
                        "⏳ SELL ORDER EXPIRED",
                        f"**{ticker}**: Limit sell unfilled after {elapsed_min:.0f} minutes.\n"
                        f"Order cancelled — position reverted to OPEN.\n"
                        f"Will re-evaluate on next monitor cycle.",
                        color=0xFFAA00,
                    )
                else:
                    print(f"    {ticker}: pending_sell — order placed {elapsed_min:.0f}min ago, waiting...")  # noqa: T201
    return actions


async def check_and_manage_positions():
    """Check all open positions against exit rules."""
    positions = load_positions()
    open_positions = [p for p in positions if p["status"] == "open"]
    pending_sells = [p for p in positions if p.get("status") == "pending_sell"]

    if not open_positions and not pending_sells:
        print(f"  [{datetime.now(ET).strftime('%H:%M')}] No open positions to monitor.")
        return

    api_key = os.getenv("KALSHI_API_KEY_ID")
    pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not api_key or not pk_path:
        print("  Missing credentials")
        return

    client = KalshiClient(api_key_id=api_key, private_key_path=pk_path, demo_mode=False)
    await client.start()

    try:
        # Get current positions from Kalshi (actual fills)
        api_positions = await client.get_positions()
        api_pos_map = {}
        for ap in api_positions:
            ticker = ap.get("ticker", "")
            qty = ap.get("position", 0)  # Net position (positive = yes, negative = no)
            if qty != 0:
                api_pos_map[ticker] = ap

        balance = await client.get_balance()
        now = datetime.now(ET)

        print(f"\n  POSITION MONITOR — {now.strftime('%I:%M %p ET')}")
        print(f"  Balance: ${balance:.2f} | Open: {len(open_positions)} | Pending sells: {len(pending_sells)}")
        print(f"  {'─'*50}")

        actions_taken = []

        # First: check pending sell orders for fill confirmation
        if pending_sells:
            pending_actions = await _check_pending_sells(positions, client, now)
            actions_taken.extend(pending_actions)
            # Refresh open positions after pending sell resolution
            open_positions = [p for p in positions if p["status"] == "open"]

        for pos in open_positions:
            ticker = pos["ticker"]
            side = pos["side"]
            entry_price = pos["avg_price"]
            contracts = pos["contracts"]

            # Get current market price
            orderbook = await client.get_orderbook(ticker)
            current_bid = 0

            # YES side: sell at YES bid. NO side: sell at NO bid (= 100 - YES ask).
            if side == "yes":
                if orderbook.get("yes"):
                    bids = [l for l in orderbook["yes"] if l[1] > 0]
                    if bids:
                        current_bid = max(b[0] for b in bids)
                sell_price = current_bid
            else:
                # For NO positions, our sell price is the NO bid.
                # Kalshi orderbook "no" key has NO bids directly.
                if orderbook.get("no"):
                    no_bids = [l for l in orderbook["no"] if l[1] > 0]
                    if no_bids:
                        current_bid = max(b[0] for b in no_bids)
                sell_price = current_bid

            # Calculate P&L
            if sell_price > 0:
                pnl_per_contract = (sell_price - entry_price) / 100
                total_pnl = pnl_per_contract * contracts
                roi = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            else:
                pnl_per_contract = 0
                total_pnl = 0
                roi = 0

            pnl_color = "+" if total_pnl >= 0 else ""
            print(f"\n  {side.upper()} {contracts}x {ticker}")
            print(f"    Entry: {entry_price}c | Now: {sell_price}c | P&L: {pnl_color}${total_pnl:.2f} ({roi:+.0f}%)")

            # Check if position actually exists on Kalshi
            api_pos = api_pos_map.get(ticker)
            if not api_pos:
                print(f"    [!] Not found in Kalshi positions — may have settled")
                # Auto-close if market likely settled
                pos["notes"].append(f"{now.isoformat()}: Not found in API — likely settled")
                pos["status"] = "settled"
                actions_taken.append(f"SETTLED: {ticker} no longer in API positions")
                await send_discord_alert(
                    "📋 POSITION SETTLED",
                    f"**{side.upper()} {contracts}x {ticker}**\n"
                    f"Entry: {entry_price}c | No longer in API — market settled.\n"
                    f"Check Kalshi portfolio for final settlement.",
                    color=0x3498DB,
                )
                continue

            # ── Settlement proximity check ──
            hours_to_settlement = (SETTLEMENT_HOUR_ET - now.hour) % 24
            if hours_to_settlement > 12:
                hours_to_settlement -= 24  # e.g. 10PM = -9 hours
            near_settlement = 0 < hours_to_settlement <= SETTLEMENT_WINDOW_HOURS

            # ═══════════════════════════════════════════════════
            # EXIT RULE 1: EFFICIENCY EXIT (90¢)
            # OVERRIDE: If near settlement AND price > 80¢, HOLD for $1
            # ═══════════════════════════════════════════════════
            if sell_price >= EFFICIENCY_EXIT_CENTS:
                if near_settlement and sell_price >= SETTLEMENT_HOLD_THRESHOLD:
                    print(f"    >>> Price {sell_price}c >= {EFFICIENCY_EXIT_CENTS}c BUT {hours_to_settlement:.0f}h to settlement")
                    print(f"    >>> HOLDING for $1.00 settlement (expected +${(100 - sell_price) / 100 * contracts:.2f} more)")
                    pos["notes"].append(f"{now.isoformat()}: Held at {sell_price}c — near settlement")
                else:
                    print(f"    >>> EFFICIENCY EXIT — Price {sell_price}c >= {EFFICIENCY_EXIT_CENTS}c")
                    print(f"    >>> Placing LIMIT SELL {contracts}x @ {sell_price}c")

                    result = await client.place_order(
                        ticker=ticker,
                        side=side,
                        action="sell",
                        count=contracts,
                        price=sell_price,
                        order_type="limit",
                    )

                    if result:
                        order_id = result.get("order", {}).get("order_id", "")
                        pos["status"] = "pending_sell"
                        pos["sell_order_id"] = order_id
                        pos["sell_placed_at"] = now.isoformat()
                        pos["_pending_remaining_qty"] = 0  # Expect full close
                        pos["_pre_sell_qty"] = contracts
                        pos["pnl_realized"] += total_pnl
                        pos["notes"].append(f"{now.isoformat()}: EFFICIENCY EXIT sell placed at {sell_price}c (order: {order_id})")
                        actions_taken.append(f"EFFICIENCY EXIT: Sell {contracts}x {ticker} @ {sell_price}c placed (pending fill)")

                        await send_discord_alert(
                            "💰 EFFICIENCY EXIT — SELL PLACED",
                            f"**Sell {contracts}x {ticker} @ {sell_price}c** (limit order)\n"
                            f"Entry: {entry_price}c | Expected P&L: ${total_pnl:.2f} ({roi:+.0f}% ROI)\n"
                            f"Status: PENDING FILL — will confirm on next cycle.",
                            color=0x00FF00,
                        )
                continue

            # ═══════════════════════════════════════════════════
            # EXIT RULE 2: FREEROLL (sell half at 2x entry)
            # ═══════════════════════════════════════════════════
            freeroll_price = entry_price * FREEROLL_MULTIPLIER
            if not pos.get("freerolled") and sell_price >= freeroll_price and contracts > 1:
                sell_qty = contracts // 2
                print(f"    >>> FREEROLL — Price {sell_price}c >= {freeroll_price:.0f}c (2x entry)")
                print(f"    >>> Selling {sell_qty} of {contracts} contracts")

                result = await client.place_order(
                    ticker=ticker,
                    side=side,
                    action="sell",
                    count=sell_qty,
                    price=sell_price,
                    order_type="limit",
                )

                if result:
                    order_id = result.get("order", {}).get("order_id", "")
                    realized = (sell_price - entry_price) / 100 * sell_qty
                    remaining = contracts - sell_qty
                    pos["status"] = "pending_sell"
                    pos["sell_order_id"] = order_id
                    pos["sell_placed_at"] = now.isoformat()
                    pos["_pending_remaining_qty"] = remaining
                    pos["_pre_sell_qty"] = contracts
                    pos["freerolled"] = True
                    pos["pnl_realized"] += realized
                    pos["peak_price"] = sell_price
                    pos["trailing_floor"] = max(entry_price, sell_price - TRAILING_OFFSET_CENTS)
                    pos["notes"].append(f"{now.isoformat()}: FREEROLL sell {sell_qty}x @ {sell_price}c placed (order: {order_id})")
                    actions_taken.append(f"FREEROLL: Sell {sell_qty}x {ticker} @ {sell_price}c placed (pending fill)")

                    await send_discord_alert(
                        "🎰 FREEROLL — SELL PLACED",
                        f"**Sell {sell_qty} of {contracts} {ticker} @ {sell_price}c** (limit order)\n"
                        f"Entry: {entry_price}c | Expected: +${realized:.2f}\n"
                        f"Status: PENDING FILL — will confirm on next cycle.",
                        color=0x00FF00,
                    )
                continue

            # ═══════════════════════════════════════════════════
            # EXIT RULE 3: TRAILING PROFIT LOCK (after freeroll)
            # Ratchets up as price rises, sells if price drops from peak
            # ═══════════════════════════════════════════════════
            if pos.get("freerolled") and contracts > 0:
                peak = pos.get("peak_price", sell_price)
                floor = pos.get("trailing_floor", entry_price)

                # Update peak and floor if price made new high
                if sell_price > peak:
                    pos["peak_price"] = sell_price
                    new_floor = max(entry_price, sell_price - TRAILING_OFFSET_CENTS)
                    if new_floor > floor:
                        pos["trailing_floor"] = new_floor
                        floor = new_floor
                        print(f"    New peak {sell_price}c — trailing floor raised to {floor}c")

                # Liquidity check: require meaningful bid depth before triggering stop
                # Prevents false triggers on stale 1¢ bids with no real volume
                bid_volume = 0
                book_key = "yes" if side == "yes" else "no"
                if orderbook.get(book_key):
                    for lvl in orderbook[book_key]:
                        if lvl[0] == current_bid and lvl[1] > 0:
                            bid_volume = lvl[1]
                            break

                # Thin book = no meaningful liquidity at the bid price
                thin_book = (bid_volume < 3 and sell_price <= 5)

                # Check if trailing stop triggered
                if sell_price <= floor and sell_price > 0 and not thin_book:
                    print(f"    >>> TRAILING STOP — Price {sell_price}c <= floor {floor}c (peak was {peak}c)")
                    print(f"    >>> Selling remaining {contracts}x")

                    result = await client.place_order(
                        ticker=ticker,
                        side=side,
                        action="sell",
                        count=contracts,
                        price=sell_price,
                        order_type="limit",
                    )

                    if result:
                        order_id = result.get("order", {}).get("order_id", "")
                        realized = (sell_price - entry_price) / 100 * contracts
                        pos["pnl_realized"] += realized
                        pos["status"] = "pending_sell"
                        pos["sell_order_id"] = order_id
                        pos["sell_placed_at"] = now.isoformat()
                        pos["_pending_remaining_qty"] = 0
                        pos["_pre_sell_qty"] = contracts
                        pos["notes"].append(f"{now.isoformat()}: TRAILING STOP sell placed at {sell_price}c (order: {order_id})")
                        actions_taken.append(f"TRAILING STOP: Sell {contracts}x {ticker} @ {sell_price}c placed (pending fill)")

                        await send_discord_alert(
                            "📉 TRAILING STOP — SELL PLACED",
                            f"**Sell {contracts}x {ticker} @ {sell_price}c** (limit order)\n"
                            f"Entry: {entry_price}c | Peak: {peak}c | Floor: {floor}c\n"
                            f"Expected: +${realized:.2f}\n"
                            f"Status: PENDING FILL — will confirm on next cycle.",
                            color=0xFF6600,
                        )
                    continue
                elif thin_book and sell_price <= floor:
                    print(f"    Trailing stop SKIPPED — thin book (bid_vol={bid_volume}, price={sell_price}c). Waiting for liquidity.")
                else:
                    print(f"    Trailing: peak={peak}c floor={floor}c current={sell_price}c — holding")

            # ═══════════════════════════════════════════════════
            # EXIT RULE 4: LOSS WARNING (alert only, no auto-sell)
            # ═══════════════════════════════════════════════════
            if sell_price < entry_price and roi < -30:
                print(f"    [!] Position down {roi:.0f}% — monitor closely")
                last_alert = pos.get("_last_loss_alert", "")
                roi_bucket = str(int(roi / 10) * 10)  # Alert per 10% bucket
                if roi_bucket != last_alert:
                    pos["_last_loss_alert"] = roi_bucket
                    await send_discord_alert(
                        "⚠️ POSITION WARNING",
                        f"**{side.upper()} {contracts}x {ticker}**\n"
                        f"Entry: {entry_price}c | Now: {sell_price}c | P&L: ${total_pnl:.2f} ({roi:+.0f}%)\n"
                        f"Consider re-scanning to check if thesis still holds.\n"
                        f"`python3 edge_scanner_v2.py` to re-evaluate",
                        color=0xFF0000,
                    )
            else:
                if not pos.get("freerolled"):
                    print(f"    Holding — no exit trigger met (freeroll at {freeroll_price:.0f}c)")

        # Save updated positions
        save_positions(positions)

        if actions_taken:
            print(f"\n  ACTIONS TAKEN:")
            for a in actions_taken:
                print(f"    {a}")
        else:
            print(f"\n  No exit triggers met. Holding all positions.")

    finally:
        await client.stop()


async def show_status():
    """Display all positions with current status."""
    positions = load_positions()

    print(f"\n{'='*60}")
    print(f"  POSITIONS — {datetime.now(ET).strftime('%I:%M %p ET, %a %b %d')}")
    print(f"{'='*60}")

    open_pos = [p for p in positions if p["status"] == "open"]
    pending_pos = [p for p in positions if p.get("status") == "pending_sell"]
    closed_pos = [p for p in positions if p["status"] == "closed"]

    if not positions:
        print("  No tracked positions.")
        return

    if pending_pos:
        print(f"\n  PENDING SELL ({len(pending_pos)}):")
        for p in pending_pos:
            placed = p.get("sell_placed_at", "unknown")
            print(f"    {p['side'].upper()} {p['contracts']}x {p['ticker']} @ {p['avg_price']}c")
            print(f"      Sell placed: {placed} | Order: {p.get('sell_order_id', 'N/A')}")

    if open_pos:
        print(f"\n  OPEN ({len(open_pos)}):")
        for p in open_pos:
            fr = " [FREEROLLED]" if p.get("freerolled") else ""
            print(f"    {p['side'].upper()} {p['contracts']}x {p['ticker']} @ {p['avg_price']}c{fr}")
            print(f"      Opened: {p['entry_time']}")
            print(f"      Exit rules: freeroll@{p['exit_rules']['freeroll_at']}c, efficiency@{p['exit_rules']['efficiency_exit']}c")
            if p.get("pnl_realized", 0) > 0:
                print(f"      Realized P&L: ${p['pnl_realized']:.2f}")

    if closed_pos:
        print(f"\n  CLOSED ({len(closed_pos)}):")
        total_pnl = 0
        for p in closed_pos:
            pnl = p.get("pnl_realized", 0)
            total_pnl += pnl
            print(f"    {p['side'].upper()} {p.get('original_contracts', p['contracts'])}x {p['ticker']} → ${pnl:+.2f}")
        print(f"\n    Total Realized P&L: ${total_pnl:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Position Monitor — Auto take-profit and exit management")
    parser.add_argument("--status", action="store_true", help="Show all positions")
    parser.add_argument("--once", action="store_true", help="Single check (for cron)")
    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status())
    else:
        asyncio.run(check_and_manage_positions())
