#!/usr/bin/env python3
"""
EXECUTE TRADE — One-command trade execution from Discord alerts.

Designed to be copy-pasted from Discord alert into terminal.
Validates everything before placing the order, then confirms.

Usage (from Discord alert):
  python3 execute_trade.py KXHIGHLAX-26FEB11-B62.5 yes 20 10
  #                        ^^^^^^^^^^^^^^^^^^^^^^^^ ^^^ ^^ ^^
  #                        ticker                   side ¢  qty

  python3 execute_trade.py KXHIGHLAX-26FEB11-B62.5 yes 20 10 --confirm
  #  Add --confirm to skip the interactive y/n prompt (still prints summary)

Arguments:
  ticker   — Kalshi market ticker (e.g. KXHIGHNY-26FEB11-B36.5)
  side     — "yes" or "no"
  price    — Limit price in cents (e.g. 20 = 20¢)
  quantity — Number of contracts

Safety:
  - Always uses LIMIT orders (maker, 0% fee)
  - Validates price against MAX_ENTRY_PRICE (50¢)
  - Validates position size against MAX_POSITION_PCT (10% of NLV)
  - Shows full order summary before executing
  - Requires explicit confirmation (unless --confirm flag)
  - Sends Discord confirmation after fill
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
from position_store import register_position
from notifications import send_discord_alert

ET = ZoneInfo("America/New_York")

# Risk limits (single source of truth: config.py)
from config import MAX_ENTRY_PRICE_CENTS as MAX_ENTRY_PRICE, MAX_POSITION_PCT


async def send_discord_confirmation(ticker: str, side: str, price: int, qty: int, cost: float, status: str):
    """Send trade confirmation to Discord via shared notifications module."""
    color = 0x00FF00 if status == "FILLED" else 0xFFAA00 if status == "RESTING" else 0xFF0000
    now = datetime.now(ET).strftime("%I:%M %p ET")

    emoji = '✅' if status == 'FILLED' else '⏳' if status == 'RESTING' else '❌'
    await send_discord_alert(
        title=f"{emoji} TRADE {status}",
        description=(
            f"**{side.upper()} {ticker}**\n"
            f"Price: {price}¢ | Qty: {qty} | Cost: ${cost:.2f}\n"
            f"Max Payout: ${qty:.2f}\n"
            f"Time: {now}"
        ),
        color=color,
        context="execute_trade",
    )


async def execute(ticker: str, side: str, price: int, quantity: int, confirm: bool = False):
    """Validate and execute a trade."""
    now = datetime.now(ET)
    print(f"\n{'='*56}")
    print(f"  TRADE EXECUTION — {now.strftime('%I:%M %p ET, %a %b %d')}")
    print(f"{'='*56}")

    # ── Validate inputs ──
    side = side.lower()
    if side not in ("yes", "no"):
        print(f"  ✗ Invalid side: {side}. Must be 'yes' or 'no'.")
        return False

    if price < 1 or price > 99:
        print(f"  ✗ Invalid price: {price}¢. Must be 1-99.")
        return False

    if side == "yes" and price > MAX_ENTRY_PRICE:
        print(f"  ✗ Price {price}¢ exceeds MAX_ENTRY_PRICE ({MAX_ENTRY_PRICE}¢).")
        print(f"    Risk/reward too poor above {MAX_ENTRY_PRICE}¢ on YES side.")
        return False

    if quantity < 1:
        print(f"  ✗ Invalid quantity: {quantity}. Must be ≥ 1.")
        return False

    # ── Connect to Kalshi ──
    api_key = os.getenv("KALSHI_API_KEY_ID")
    pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not api_key or not pk_path:
        print("  ✗ Missing KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH in .env")
        return False

    client = KalshiClient(api_key_id=api_key, private_key_path=pk_path, demo_mode=False)
    await client.start()

    try:
        # ── Fetch balance ──
        balance = await client.get_balance()
        print(f"\n  Account Balance: ${balance:.2f}")

        # ── Validate position size ──
        cost_per_contract = price / 100
        total_cost = cost_per_contract * quantity
        max_allowed = balance * MAX_POSITION_PCT

        if total_cost > max_allowed:
            print(f"  ✗ Total cost ${total_cost:.2f} exceeds {MAX_POSITION_PCT*100:.0f}% of NLV (${max_allowed:.2f})")
            print(f"    Max contracts at {price}¢: {int(max_allowed / cost_per_contract)}")
            return False

        if total_cost > balance:
            print(f"  ✗ Insufficient balance. Need ${total_cost:.2f}, have ${balance:.2f}")
            return False

        # ── Fetch current market data ──
        print(f"\n  Fetching orderbook for {ticker}...")
        orderbook = await client.get_orderbook(ticker)

        ob_yes_bid = 0
        ob_yes_ask = 0
        if orderbook.get("yes"):
            yes_bids = [l for l in orderbook["yes"] if l[1] > 0]
            if yes_bids:
                ob_yes_bid = max(b[0] for b in yes_bids)
        if orderbook.get("no"):
            no_bids = [l for l in orderbook["no"] if l[1] > 0]
            if no_bids:
                ob_yes_ask = 100 - max(b[0] for b in no_bids)

        # ── Order summary ──
        max_payout = quantity
        profit_if_win = max_payout - total_cost
        roi = (profit_if_win / total_cost * 100) if total_cost > 0 else 0

        print(f"\n  ORDER SUMMARY")
        print(f"  {'─'*42}")
        print(f"  Ticker:       {ticker}")
        print(f"  Side:         {side.upper()}")
        print(f"  Price:        {price}¢ (LIMIT, maker 0% fee)")
        print(f"  Quantity:     {quantity} contracts")
        print(f"  Cost:         ${total_cost:.2f}")
        print(f"  Max Payout:   ${max_payout:.2f}")
        print(f"  Profit:       ${profit_if_win:.2f} ({roi:.0f}% ROI)")
        print(f"  {'─'*42}")
        print(f"  Book:         Bid={ob_yes_bid}¢  Ask={ob_yes_ask}¢")
        print(f"  Balance after: ${balance - total_cost:.2f}")
        print(f"  {'─'*42}")

        # Check if our price crosses the spread
        if side == "yes" and ob_yes_ask > 0 and price >= ob_yes_ask:
            print(f"\n  ⚠ Your bid ({price}¢) is AT/ABOVE the ask ({ob_yes_ask}¢).")
            print(f"    You'll pay taker fees. Consider bidding {max(1, ob_yes_ask - 1)}¢.")

        # ── Confirm ──
        if not confirm:
            response = input(f"\n  Execute this trade? (y/n): ").strip().lower()
            if response != "y":
                print("  Cancelled.")
                return False

        # ── Execute ──
        print(f"\n  Placing order...")
        result = await client.place_order(
            ticker=ticker,
            side=side,
            action="buy",
            count=quantity,
            price=price,
            order_type="limit",
        )

        if result:
            order = result.get("order", result)
            status = order.get("status", "unknown").upper()
            order_id = order.get("order_id", "N/A")

            if status in ("RESTING", "PENDING"):
                print(f"  ⏳ Order RESTING (limit order in book)")
                print(f"     Order ID: {order_id}")
                print(f"     Waiting for fill at {price}¢...")
            elif status == "EXECUTED":
                print(f"  ✅ Order FILLED!")
                print(f"     Order ID: {order_id}")
            else:
                print(f"  Order status: {status}")
                print(f"     Order ID: {order_id}")

            # Register position for the position monitor to track
            register_position(ticker, side, price, quantity, order_id, status)

            await send_discord_confirmation(ticker, side, price, quantity, total_cost, status)
            return True
        else:
            print(f"  ✗ Order placement failed. Check logs.")
            await send_discord_confirmation(ticker, side, price, quantity, total_cost, "FAILED")
            return False

    finally:
        await client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute Trade — One-command Kalshi order placement",
        epilog="Example: python3 execute_trade.py KXHIGHLAX-26FEB11-B62.5 yes 20 10",
    )
    parser.add_argument("ticker", help="Kalshi market ticker")
    parser.add_argument("side", choices=["yes", "no"], help="Trade side")
    parser.add_argument("price", type=int, help="Limit price in cents (1-99)")
    parser.add_argument("quantity", type=int, help="Number of contracts")
    parser.add_argument("--confirm", action="store_true", help="Skip interactive confirmation")
    args = parser.parse_args()

    asyncio.run(execute(args.ticker, args.side, args.price, args.quantity, args.confirm))
