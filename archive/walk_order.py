#!/usr/bin/env python3
"""Cancel and repost order at midpoint - Walk the Order strategy."""

import asyncio
import os
from dotenv import load_dotenv
from kalshi_client import KalshiClient

load_dotenv()


async def main():
    # Cancel this order
    OLD_ORDER_ID = "fc587d57-f76b-45e9-942d-8aca556b07f9"

    # New order parameters
    TICKER = "KXHIGHNY-26JAN22-B42.5"
    SIDE = "yes"
    ACTION = "buy"
    NEW_PRICE_CENTS = 35  # Midpoint
    NEW_CONTRACTS = 303   # $106.07 / 0.35 = 303

    total_cost = NEW_CONTRACTS * NEW_PRICE_CENTS / 100

    print("=" * 60)
    print("  WALK THE ORDER - MIDPOINT STRATEGY")
    print("=" * 60)
    print(f"  Canceling: {OLD_ORDER_ID[:8]}...")
    print(f"  New Order: {NEW_CONTRACTS}x @ {NEW_PRICE_CENTS}¢")
    print(f"  Cost: ${total_cost:.2f}")
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
        print(f"\n[1/3] Canceling order {OLD_ORDER_ID[:12]}...")
        cancel_result = await client.cancel_order(OLD_ORDER_ID)
        print(f"      Cancel result: {cancel_result}")

        # Step 2: Check balance
        balance = await client.get_balance()
        print(f"\n[2/3] Balance: ${balance:.2f}")

        # Step 3: Get current orderbook
        ob = await client.get_orderbook(TICKER)
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])

        best_bid = yes_bids[0][0] if yes_bids else 0
        best_ask = 100 - no_bids[0][0] if no_bids else 100

        print(f"      Orderbook: Bid {best_bid}¢ / Ask {best_ask}¢")

        # Step 4: Place new order at midpoint
        print(f"\n[3/3] Placing {NEW_CONTRACTS}x @ {NEW_PRICE_CENTS}¢...")

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
            remaining = order.get("remaining_count", "N/A")
            filled = NEW_CONTRACTS - remaining if isinstance(remaining, int) else 0

            print()
            print("=" * 60)
            print(f"  NEW ORDER PLACED")
            print("=" * 60)
            print(f"  Order ID:  {order_id}")
            print(f"  Status:    {status}")
            print(f"  Filled:    {filled}")
            print(f"  Remaining: {remaining}")
            print("=" * 60)

            if status == "resting":
                print("\n  ⚠️  Order RESTING - not filled yet")
                print("     Monitor for fills before 10:51 PM MOS release")
            elif remaining == 0:
                print("\n  ✓  ORDER FULLY FILLED!")
        else:
            print("\n[ERROR] Order failed")

        # Final status
        print("\n[FINAL STATUS]")
        positions = await client.get_positions()
        for pos in positions:
            ticker = pos.get("ticker", "")
            if "26JAN22" in ticker and "B42.5" in ticker:
                contracts = pos.get("position", 0)
                exposure = pos.get("market_exposure", 0) / 100
                print(f"  Position: {contracts} contracts")
                print(f"  Exposure: ${exposure:.2f}")

        orders = await client.get_orders(status="resting")
        for o in orders:
            if "B42.5" in o.get("ticker", ""):
                remaining = o.get("remaining_count", 0)
                price = o.get("yes_price", 0)
                oid = o.get("order_id", "")[:12]
                print(f"  Resting: {remaining} @ {price}¢ (ID: {oid}...)")

        new_balance = await client.get_balance()
        print(f"  Balance: ${new_balance:.2f}")

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
