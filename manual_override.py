#!/usr/bin/env python3
"""
ONE-CLICK SNIPER TRADE EXECUTOR
Place entry, wait for fill, place exit, close app.

No FOMO. No panic. No over-trading.

Usage:
    python3 sniper_trade.py --ticker KXHIGHNY-26JAN17-B33.5 --entry 30 --exit 70 --size 100
"""

import asyncio
import argparse
import os
from kalshi_client import KalshiClient
from alerts import send_alert


async def get_market_context(client: KalshiClient, ticker: str) -> dict:
    """Get current market state for informed decision."""
    market = await client.get_market(ticker)
    orderbook = await client.get_orderbook(ticker)

    market_detail = market.get("market", {})

    return {
        "ticker": ticker,
        "title": market_detail.get("title", ""),
        "yes_bid": market_detail.get("yes_bid", 0),
        "yes_ask": market_detail.get("yes_ask", 100),
        "volume": market_detail.get("volume", 0),
        "expiry": market_detail.get("expiration_time", ""),
        "orderbook": orderbook
    }


async def snipe_opportunity(
    ticker: str,
    entry_price_cents: int,
    exit_price_cents: int,
    position_size: int,
    max_wait_minutes: int = 60,
    demo_mode: bool = False
):
    """
    Execute sniper trade with discipline.

    Strategy:
    1. Place LIMIT buy order at entry price
    2. Wait for fill (max wait time)
    3. Once filled, place LIMIT sell order at exit price
    4. Done - close app and wait for settlement

    Args:
        ticker: Kalshi market ticker
        entry_price_cents: Limit buy price (e.g., 30 = 30¢)
        exit_price_cents: Limit sell price (e.g., 70 = 70¢)
        position_size: Number of contracts
        max_wait_minutes: Max time to wait for entry fill
        demo_mode: Use demo API (default: live)
    """

    client = KalshiClient(
        api_key_id=os.getenv("KALSHI_API_KEY_ID"),
        private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH"),
        demo_mode=demo_mode
    )
    await client.start()

    print("\n" + "=" * 80)
    print("🎯 SNIPER TRADE EXECUTION")
    print("=" * 80)

    # Get market context
    context = await get_market_context(client, ticker)

    print(f"\n📊 MARKET CONTEXT")
    print(f"   Ticker:  {ticker}")
    print(f"   Title:   {context['title']}")
    print(f"   Bid:     {context['yes_bid']}¢")
    print(f"   Ask:     {context['yes_ask']}¢")
    print(f"   Spread:  {context['yes_ask'] - context['yes_bid']}¢")
    print(f"   Volume:  {context['volume']:,}")

    # Validate entry price
    if entry_price_cents > context['yes_ask']:
        print(f"\n⚠️ WARNING: Your entry {entry_price_cents}¢ > current ask {context['yes_ask']}¢")
        print(f"   You're paying MORE than the ask price!")
        print(f"   Consider using {context['yes_ask']}¢ or lower.")

        response = input(f"\n   Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Trade cancelled by user")
            await client.stop()
            return

    # Calculate expected profit
    profit_per_contract = exit_price_cents - entry_price_cents
    max_cost = entry_price_cents * position_size / 100
    expected_profit = profit_per_contract * position_size / 100

    print(f"\n💰 TRADE PARAMETERS")
    print(f"   Entry Price:  {entry_price_cents}¢")
    print(f"   Exit Price:   {exit_price_cents}¢")
    print(f"   Position:     {position_size} contracts")
    print(f"   Max Cost:     ${max_cost:.2f}")
    print(f"   Expected Profit: ${expected_profit:.2f} ({100 * profit_per_contract / entry_price_cents:.0f}% ROI)")

    # Confirm trade
    print(f"\n⚠️ CONFIRM TRADE")
    print(f"   This is a LIVE trade using REAL money.")
    response = input(f"   Type 'EXECUTE' to continue: ")

    if response != "EXECUTE":
        print("❌ Trade cancelled by user")
        await client.stop()
        return

    # ========================================================================
    # STEP 1: Place Entry Order (LIMIT, not market)
    # ========================================================================

    print(f"\n" + "-" * 80)
    print(f"📤 STEP 1: PLACING ENTRY ORDER")
    print("-" * 80)
    print(f"   Type:     LIMIT BUY")
    print(f"   Price:    {entry_price_cents}¢")
    print(f"   Size:     {position_size} contracts")

    entry_order = await client.place_order(
        ticker=ticker,
        side="yes",
        action="buy",
        count=position_size,
        price=entry_price_cents,
        order_type="limit"
    )

    order_id = entry_order.get('order', {}).get('order_id')
    if not order_id:
        print(f"❌ ENTRY ORDER FAILED")
        print(f"   Response: {entry_order}")
        await send_alert(f"🚨 Sniper trade FAILED: {ticker} - Entry order rejected")
        await client.stop()
        return

    print(f"✅ Entry order placed: {order_id}")
    await send_alert(f"📤 Sniper entry: {ticker} @ {entry_price_cents}¢ × {position_size}")

    # ========================================================================
    # STEP 2: Wait for Fill
    # ========================================================================

    print(f"\n" + "-" * 80)
    print(f"⏳ STEP 2: WAITING FOR FILL (max {max_wait_minutes} minutes)")
    print("-" * 80)

    filled = False
    for elapsed_minutes in range(max_wait_minutes):
        await asyncio.sleep(60)  # Check every minute

        # Check order status
        orders = await client.get_orders(ticker=ticker)
        entry_status = next(
            (o for o in orders if o.get('order_id') == order_id),
            {}
        ).get('status')

        if entry_status == 'filled':
            print(f"✅ FILLED at {entry_price_cents}¢ after {elapsed_minutes + 1} minutes!")
            await send_alert(f"✅ Sniper FILLED: {ticker} @ {entry_price_cents}¢")
            filled = True
            break

        # Progress update every 5 minutes
        if (elapsed_minutes + 1) % 5 == 0:
            remaining = max_wait_minutes - (elapsed_minutes + 1)
            print(f"   ⏰ Still waiting... ({elapsed_minutes + 1}/{max_wait_minutes} min, {remaining} min remaining)")

    if not filled:
        print(f"\n⏰ TIMEOUT: Entry order not filled after {max_wait_minutes} minutes")
        print(f"   Reason: Price may have moved away from your {entry_price_cents}¢ limit")
        print(f"   Cancelling order...")

        cancel_result = await client.cancel_order(order_id)
        print(f"   Cancelled: {cancel_result}")

        await send_alert(f"⏰ Sniper timeout: {ticker} - Entry not filled after {max_wait_minutes}min")
        await client.stop()
        return

    # ========================================================================
    # STEP 3: Place Exit Order (Take Profit)
    # ========================================================================

    print(f"\n" + "-" * 80)
    print(f"📤 STEP 3: PLACING EXIT ORDER (TAKE PROFIT)")
    print("-" * 80)
    print(f"   Type:     LIMIT SELL")
    print(f"   Price:    {exit_price_cents}¢")
    print(f"   Size:     {position_size} contracts")

    exit_order = await client.place_order(
        ticker=ticker,
        side="yes",
        action="sell",
        count=position_size,
        price=exit_price_cents,
        order_type="limit"
    )

    exit_order_id = exit_order.get('order', {}).get('order_id')
    if not exit_order_id:
        print(f"❌ EXIT ORDER FAILED")
        print(f"   Response: {exit_order}")
        print(f"⚠️ WARNING: You have an OPEN POSITION without exit order!")
        print(f"   Go to Kalshi and manually close the position.")
        await send_alert(f"🚨 Sniper exit FAILED: {ticker} - Manual close required!")
        await client.stop()
        return

    print(f"✅ Exit order placed: {exit_order_id}")
    await send_alert(f"📤 Sniper exit: {ticker} @ {exit_price_cents}¢ × {position_size}")

    # ========================================================================
    # DONE
    # ========================================================================

    print(f"\n" + "=" * 80)
    print(f"🎉 SNIPER TRADE COMPLETE")
    print("=" * 80)

    print(f"\n📊 TRADE SUMMARY")
    print(f"   Ticker:       {ticker}")
    print(f"   Entry:        {entry_price_cents}¢ (FILLED)")
    print(f"   Exit Target:  {exit_price_cents}¢ (PENDING)")
    print(f"   Position:     {position_size} contracts")
    print(f"   Cost:         ${entry_price_cents * position_size / 100:.2f}")
    print(f"   If Exit Fills: ${expected_profit:.2f} profit ({100 * profit_per_contract / entry_price_cents:.0f}% ROI)")

    print(f"\n⏭️ NEXT STEPS")
    print(f"   1. ✅ Entry filled at {entry_price_cents}¢")
    print(f"   2. ⏳ Exit order active at {exit_price_cents}¢ (will fill when market reaches target)")
    print(f"   3. 💤 CLOSE THIS APP AND GO OUTSIDE")
    print(f"   4. 📱 You'll get Discord alert when exit fills")
    print(f"   5. ⏰ Check back after settlement (6-12 hours)")

    print(f"\n🚨 IMPORTANT: DO NOT TOUCH ANYTHING")
    print(f"   - Don't cancel exit order")
    print(f"   - Don't place more orders")
    print(f"   - Don't watch the screen")
    print(f"   - Let the system work")

    print(f"\n" + "=" * 80)

    await send_alert(
        f"✅ Sniper trade complete: {ticker}\n"
        f"Entry: {entry_price_cents}¢ (filled)\n"
        f"Exit: {exit_price_cents}¢ (pending)\n"
        f"Expected profit: ${expected_profit:.2f}"
    )

    await client.stop()


async def main():
    parser = argparse.ArgumentParser(description="Sniper trade executor")
    parser.add_argument("--ticker", required=True, help="Kalshi market ticker")
    parser.add_argument("--entry", type=int, required=True, help="Entry price in cents (e.g., 30)")
    parser.add_argument("--exit", type=int, required=True, help="Exit price in cents (e.g., 70)")
    parser.add_argument("--size", type=int, default=100, help="Position size (contracts)")
    parser.add_argument("--wait", type=int, default=60, help="Max minutes to wait for fill")
    parser.add_argument("--demo", action="store_true", help="Use demo mode")

    args = parser.parse_args()

    await snipe_opportunity(
        ticker=args.ticker,
        entry_price_cents=args.entry,
        exit_price_cents=args.exit,
        position_size=args.size,
        max_wait_minutes=args.wait,
        demo_mode=args.demo
    )


if __name__ == "__main__":
    asyncio.run(main())
