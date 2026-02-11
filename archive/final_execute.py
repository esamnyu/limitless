#!/usr/bin/env python3
"""Final Execution - Cancel legacy orders, deploy to Jan 22."""

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
    TARGET_CONTRACTS = 248

    api_key = os.getenv('KALSHI_API_KEY_ID')
    private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')

    client = KalshiClient(
        api_key_id=api_key,
        private_key_path=private_key_path,
        demo_mode=False
    )

    try:
        await client.start()

        print("=" * 60)
        print("  FINAL EXECUTION - CLEAR & DEPLOY")
        print("=" * 60)

        # Step 1: Get all resting orders
        print("\n[1/4] Finding legacy sell orders...")
        orders = await client.get_orders(status="resting")

        canceled_count = 0
        for o in orders:
            ticker = o.get("ticker", "")
            order_id = o.get("order_id", "")

            # Cancel Jan 21 orders (T33 and B33.5)
            if "26JAN21" in ticker:
                remaining = o.get("remaining_count", 0)
                price = o.get("yes_price", 0)
                print(f"      Canceling: {ticker} ({remaining}x @ {price}¢)")

                result = await client.cancel_order(order_id)
                if result:
                    canceled_count += 1
                    print(f"      ✓ Canceled")

        print(f"\n      Canceled {canceled_count} legacy orders")

        # Step 2: Wait for balance to update
        await asyncio.sleep(1)

        # Step 3: Check new balance
        balance = await client.get_balance()
        print(f"\n[2/4] New Balance: ${balance:.2f}")

        # Calculate max contracts with small buffer
        max_contracts = min(TARGET_CONTRACTS, int((balance * 0.99) / (PRICE_CENTS / 100)))
        total_cost = max_contracts * PRICE_CENTS / 100

        print(f"      Deploying: {max_contracts}x @ {PRICE_CENTS}¢ = ${total_cost:.2f}")

        # Step 4: Get orderbook
        print(f"\n[3/4] Checking orderbook...")
        ob = await client.get_orderbook(TICKER)
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])
        best_bid = yes_bids[0][0] if yes_bids else 0
        best_ask = 100 - no_bids[0][0] if no_bids else 100
        print(f"      Bid {best_bid}¢ / Ask {best_ask}¢ (Spread: {best_ask - best_bid}¢)")

        # Step 5: Execute the trade
        print(f"\n[4/4] EXECUTING: {max_contracts}x {TICKER} @ {PRICE_CENTS}¢")

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
            print("  ORDER EXECUTED")
            print("=" * 60)
            print(f"  Order ID:  {order_id}")
            print(f"  Status:    {status}")
            print(f"  Filled:    {filled} contracts")
            print(f"  Remaining: {remaining} contracts (resting)")

            if filled > 0:
                print(f"\n  ✓ IMMEDIATE FILL: {filled} @ 40¢")
            if remaining > 0:
                print(f"  ⏳ RESTING: {remaining} @ 40¢ (waiting for sellers)")

            print("=" * 60)
        else:
            print("\n[ERROR] Order failed!")

        # Final position summary
        print("\n" + "=" * 60)
        print("  FINAL POSITION SUMMARY")
        print("=" * 60)

        positions = await client.get_positions()
        for pos in positions:
            ticker = pos.get("ticker", "")
            contracts = pos.get("position", 0)
            exposure = pos.get("market_exposure", 0) / 100

            if contracts != 0:
                avg = (exposure / contracts * 100) if contracts > 0 else 0
                print(f"\n  {ticker}")
                print(f"    Contracts: {contracts}")
                print(f"    Exposure:  ${exposure:.2f}")
                print(f"    Avg Price: {avg:.1f}¢")

        # Resting orders
        orders = await client.get_orders(status="resting")
        jan22_orders = [o for o in orders if "26JAN22" in o.get("ticker", "")]
        if jan22_orders:
            print(f"\n  RESTING ORDERS (Jan 22):")
            for o in jan22_orders:
                rem = o.get("remaining_count", 0)
                price = o.get("yes_price", 0)
                print(f"    {rem}x @ {price}¢")

        final_balance = await client.get_balance()
        print(f"\n  Cash Balance: ${final_balance:.2f}")
        print("=" * 60)

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
