#!/usr/bin/env python3
"""Retry order with buffer for balance timing."""

import asyncio
import os
from dotenv import load_dotenv
from kalshi_client import KalshiClient

load_dotenv()


async def main():
    TICKER = "KXHIGHNY-26JAN22-B42.5"
    SIDE = "yes"
    ACTION = "buy"
    PRICE_CENTS = 35

    api_key = os.getenv('KALSHI_API_KEY_ID')
    private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')

    client = KalshiClient(
        api_key_id=api_key,
        private_key_path=private_key_path,
        demo_mode=False
    )

    try:
        await client.start()

        # Check balance and orderbook
        balance = await client.get_balance()
        print(f"Balance: ${balance:.2f}")

        ob = await client.get_orderbook(TICKER)
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])

        best_bid = yes_bids[0][0] if yes_bids else 0
        best_ask = 100 - no_bids[0][0] if no_bids else 100

        print(f"Orderbook: Bid {best_bid}¢ / Ask {best_ask}¢")
        print(f"Spread: {best_ask - best_bid}¢")

        # Calculate max contracts with 1% buffer
        max_contracts = int((balance * 0.99) / (PRICE_CENTS / 100))
        print(f"\nMax contracts @ {PRICE_CENTS}¢: {max_contracts}")

        # Check for resting orders
        orders = await client.get_orders(status="resting")
        for o in orders:
            print(f"Resting order: {o.get('ticker')} - {o.get('remaining_count')} @ {o.get('yes_price')}¢")

        # Place order
        print(f"\nPlacing {max_contracts}x @ {PRICE_CENTS}¢...")

        result = await client.place_order(
            ticker=TICKER,
            side=SIDE,
            action=ACTION,
            count=max_contracts,
            price=PRICE_CENTS,
            order_type="limit"
        )

        if result:
            order = result.get("order", {})
            order_id = order.get("order_id", "N/A")
            status = order.get("status", "N/A")
            remaining = order.get("remaining_count", "N/A")

            print()
            print("=" * 60)
            print(f"  Order ID:  {order_id}")
            print(f"  Status:    {status}")
            print(f"  Remaining: {remaining}")
            print("=" * 60)
        else:
            print("[ERROR] Order failed - trying smaller size...")

            # Try with 90% of balance
            smaller = int((balance * 0.90) / (PRICE_CENTS / 100))
            print(f"Retrying with {smaller} contracts...")

            result2 = await client.place_order(
                ticker=TICKER,
                side=SIDE,
                action=ACTION,
                count=smaller,
                price=PRICE_CENTS,
                order_type="limit"
            )

            if result2:
                order = result2.get("order", {})
                print(f"  Order ID:  {order.get('order_id', 'N/A')}")
                print(f"  Status:    {order.get('status', 'N/A')}")
                print(f"  Remaining: {order.get('remaining_count', 'N/A')}")

        # Final check
        new_balance = await client.get_balance()
        print(f"\nFinal Balance: ${new_balance:.2f}")

        positions = await client.get_positions()
        for pos in positions:
            if "B42.5" in pos.get("ticker", ""):
                print(f"Position: {pos.get('position')} contracts")

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
