#!/usr/bin/env python3
"""Execute trade - user has already confirmed."""

import asyncio
import os
from dotenv import load_dotenv
from kalshi_client import KalshiClient

load_dotenv()


async def main():
    # Trade parameters - USER AUTHORIZED
    TICKER = "KXHIGHNY-26JAN22-B42.5"
    SIDE = "yes"
    ACTION = "buy"
    PRICE_CENTS = 29
    CONTRACTS = 365

    total_cost = CONTRACTS * PRICE_CENTS / 100

    print("=" * 60)
    print("  EXECUTING TRADE - FRONT-RUN MOS PLAY")
    print("=" * 60)
    print(f"  {CONTRACTS}x {TICKER} @ {PRICE_CENTS}¢")
    print(f"  Total Cost: ${total_cost:.2f}")
    print("=" * 60)

    # Initialize client
    api_key = os.getenv('KALSHI_API_KEY_ID')
    private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')

    client = KalshiClient(
        api_key_id=api_key,
        private_key_path=private_key_path,
        demo_mode=False  # LIVE MODE
    )

    try:
        await client.start()

        # Check balance
        balance = await client.get_balance()
        print(f"\n[CHECK] Balance: ${balance:.2f}")

        if total_cost > balance:
            print(f"[ERROR] Insufficient funds. Need ${total_cost:.2f}")
            return

        # Get current orderbook
        ob = await client.get_orderbook(TICKER)
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])

        print(f"[ORDERBOOK] Yes Bids: {yes_bids[:3]}")
        print(f"[ORDERBOOK] No Bids: {no_bids[:3]}")

        # Place the order
        print(f"\n[PLACING] {CONTRACTS}x {TICKER} @ {PRICE_CENTS}¢...")

        result = await client.place_order(
            ticker=TICKER,
            side=SIDE,
            action=ACTION,
            count=CONTRACTS,
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
            print(f"  ORDER RESULT")
            print("=" * 60)
            print(f"  Order ID:  {order_id}")
            print(f"  Status:    {status}")
            print(f"  Remaining: {remaining}")
            print("=" * 60)
        else:
            print("\n[ERROR] Order failed - empty response")

        # Check positions
        print("\n[POSITIONS AFTER TRADE]")
        positions = await client.get_positions()
        for pos in positions:
            ticker = pos.get("ticker", "")
            if "26JAN22" in ticker:
                contracts = pos.get("position", 0)
                exposure = pos.get("market_exposure", 0) / 100
                print(f"  {ticker}: {contracts} contracts, ${exposure:.2f} exposure")

        # Check resting orders
        print("\n[RESTING ORDERS]")
        orders = await client.get_orders(status="resting")
        for o in orders:
            ticker = o.get("ticker", "")
            if "26JAN22" in ticker:
                remaining = o.get("remaining_count", 0)
                price = o.get("yes_price", 0)
                print(f"  {ticker}: {remaining} @ {price}¢ (resting)")

        new_balance = await client.get_balance()
        print(f"\n[BALANCE] ${new_balance:.2f}")

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
