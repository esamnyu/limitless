#!/usr/bin/env python3
"""40-Cent Bridge - Retry with correct size."""

import asyncio
import os
from dotenv import load_dotenv
from kalshi_client import KalshiClient

load_dotenv()


async def main():
    TICKER = "KXHIGHNY-26JAN22-B42.5"
    SIDE = "yes"
    ACTION = "buy"
    PRICE_CENTS = 40

    api_key = os.getenv('KALSHI_API_KEY_ID')
    private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')

    client = KalshiClient(
        api_key_id=api_key,
        private_key_path=private_key_path,
        demo_mode=False
    )

    try:
        await client.start()

        balance = await client.get_balance()
        print(f"Balance: ${balance:.2f}")

        # Calculate max with 2% buffer
        max_contracts = int((balance * 0.98) / (PRICE_CENTS / 100))
        print(f"Max contracts @ {PRICE_CENTS}¢: {max_contracts}")

        ob = await client.get_orderbook(TICKER)
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])
        best_bid = yes_bids[0][0] if yes_bids else 0
        best_ask = 100 - no_bids[0][0] if no_bids else 100
        print(f"Orderbook: Bid {best_bid}¢ / Ask {best_ask}¢ (Spread: {best_ask - best_bid}¢)")

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
            remaining = order.get("remaining_count", 0)
            filled = max_contracts - remaining

            print()
            print("=" * 60)
            print(f"  Order ID:  {order_id}")
            print(f"  Status:    {status}")
            print(f"  Filled:    {filled}")
            print(f"  Remaining: {remaining}")
            print("=" * 60)
        else:
            print("[ERROR] Order still failing")

        # Position check
        positions = await client.get_positions()
        for pos in positions:
            if "B42.5" in pos.get("ticker", ""):
                contracts = pos.get("position", 0)
                exposure = pos.get("market_exposure", 0) / 100
                print(f"\nPosition: {contracts} contracts (${exposure:.2f})")

        orders = await client.get_orders(status="resting")
        for o in orders:
            if "B42.5" in o.get("ticker", ""):
                rem = o.get("remaining_count", 0)
                price = o.get("yes_price", 0)
                print(f"Resting: {rem}x @ {price}¢")

        new_bal = await client.get_balance()
        print(f"Balance: ${new_bal:.2f}")

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
