#!/usr/bin/env python3
"""40-Cent Bridge - Force the seller's hand."""

import asyncio
import os
from dotenv import load_dotenv
from kalshi_client import KalshiClient

load_dotenv()


async def main():
    # Cancel this order
    OLD_ORDER_ID = "ce0b6a74-9c9b-4c0f-9c7c-c42e948c5b5b"

    # New aggressive bid
    TICKER = "KXHIGHNY-26JAN22-B42.5"
    SIDE = "yes"
    ACTION = "buy"
    NEW_PRICE_CENTS = 40
    NEW_CONTRACTS = 250

    total_cost = NEW_CONTRACTS * NEW_PRICE_CENTS / 100

    print("=" * 60)
    print("  40-CENT BRIDGE - FORCE THE SELLER")
    print("=" * 60)
    print(f"  Cancel: {OLD_ORDER_ID[:12]}...")
    print(f"  New:    {NEW_CONTRACTS}x @ {NEW_PRICE_CENTS}¢")
    print(f"  Cost:   ${total_cost:.2f}")
    print("=" * 60)

    api_key = os.getenv('KALSHI_API_KEY_ID')
    private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')

    client = KalshiClient(
        api_key_id=api_key,
        private_key_path=private_key_path,
        demo_mode=False
    )

    try:
        await client.start()

        # Step 1: Cancel old order
        print(f"\n[1/4] Canceling 35¢ order...")
        cancel_result = await client.cancel_order(OLD_ORDER_ID)
        canceled_remaining = cancel_result.get("reduced_by", 0)
        print(f"      Canceled {canceled_remaining} contracts")

        # Step 2: Check balance
        await asyncio.sleep(0.5)  # Brief pause for balance to settle
        balance = await client.get_balance()
        print(f"\n[2/4] Balance: ${balance:.2f}")

        # Step 3: Get orderbook
        ob = await client.get_orderbook(TICKER)
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])

        best_bid = yes_bids[0][0] if yes_bids else 0
        best_ask = 100 - no_bids[0][0] if no_bids else 100

        print(f"\n[3/4] Orderbook: Bid {best_bid}¢ / Ask {best_ask}¢")
        print(f"      Spread: {best_ask - best_bid}¢")
        print(f"      Our 40¢ will be {40 - best_bid}¢ above best bid")
        print(f"      Gap to ask: {best_ask - 40}¢")

        # Step 4: Place aggressive 40¢ bid
        print(f"\n[4/4] Placing {NEW_CONTRACTS}x @ {NEW_PRICE_CENTS}¢...")

        result = await client.place_order(
            ticker=TICKER,
            side=SIDE,
            action=ACTION,
            count=NEW_CONTRACTS,
            price=NEW_PRICE_CENTS,
            order_type="limit"
        )

        if result:
            order = result.get("order", {})
            order_id = order.get("order_id", "N/A")
            status = order.get("status", "N/A")
            remaining = order.get("remaining_count", 0)
            filled = NEW_CONTRACTS - remaining

            print()
            print("=" * 60)
            print(f"  ORDER RESULT")
            print("=" * 60)
            print(f"  Order ID:  {order_id}")
            print(f"  Status:    {status}")
            print(f"  Filled:    {filled} contracts")
            print(f"  Remaining: {remaining} contracts")
            print("=" * 60)

            if filled > 0:
                print(f"\n  ✓ GOT {filled} CONTRACTS @ 40¢!")
            if remaining > 0:
                print(f"\n  ⏳ {remaining} still resting - watching for fills...")
        else:
            print("\n[ERROR] Order failed")

        # Final status
        print("\n" + "=" * 60)
        print("  POSITION SUMMARY")
        print("=" * 60)

        positions = await client.get_positions()
        for pos in positions:
            ticker = pos.get("ticker", "")
            if "B42.5" in ticker and "26JAN22" in ticker:
                contracts = pos.get("position", 0)
                exposure = pos.get("market_exposure", 0) / 100
                avg_price = (exposure / contracts * 100) if contracts > 0 else 0
                print(f"  Ticker:    {ticker}")
                print(f"  Position:  {contracts} contracts")
                print(f"  Exposure:  ${exposure:.2f}")
                print(f"  Avg Price: {avg_price:.1f}¢")

        # Resting orders
        orders = await client.get_orders(status="resting")
        jan22_orders = [o for o in orders if "26JAN22" in o.get("ticker", "") and "B42.5" in o.get("ticker", "")]
        if jan22_orders:
            print(f"\n  RESTING ORDERS:")
            for o in jan22_orders:
                rem = o.get("remaining_count", 0)
                price = o.get("yes_price", 0)
                oid = o.get("order_id", "")[:12]
                print(f"    {rem}x @ {price}¢ (ID: {oid}...)")

        new_balance = await client.get_balance()
        print(f"\n  Balance:   ${new_balance:.2f}")
        print("=" * 60)

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
