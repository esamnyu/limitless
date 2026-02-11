#!/usr/bin/env python3
"""Full account status check."""

import asyncio
import os
from dotenv import load_dotenv
from kalshi_client import KalshiClient

load_dotenv()


async def main():
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
        print("  FULL ACCOUNT STATUS")
        print("=" * 60)

        balance = await client.get_balance()
        print(f"\nCash Balance: ${balance:.2f}")

        # All positions
        print("\n--- ALL POSITIONS ---")
        positions = await client.get_positions()
        total_exposure = 0
        for pos in positions:
            ticker = pos.get("ticker", "")
            contracts = pos.get("position", 0)
            exposure = pos.get("market_exposure", 0) / 100
            if contracts != 0:
                total_exposure += exposure
                print(f"{ticker}")
                print(f"  Contracts: {contracts}")
                print(f"  Exposure:  ${exposure:.2f}")
                print()

        print(f"Total Position Exposure: ${total_exposure:.2f}")

        # All resting orders
        print("\n--- ALL RESTING ORDERS ---")
        orders = await client.get_orders(status="resting")
        total_order_value = 0
        for o in orders:
            ticker = o.get("ticker", "")
            remaining = o.get("remaining_count", 0)
            yes_price = o.get("yes_price", 0)
            order_value = remaining * yes_price / 100
            total_order_value += order_value

            print(f"{ticker}")
            print(f"  Remaining: {remaining} @ {yes_price}¢")
            print(f"  Capital Locked: ${order_value:.2f}")
            print()

        print(f"Total Capital in Resting Orders: ${total_order_value:.2f}")

        # Calculate true buying power
        buying_power = balance - total_order_value
        print(f"\n--- BUYING POWER ---")
        print(f"Cash:           ${balance:.2f}")
        print(f"Locked Orders:  ${total_order_value:.2f}")
        print(f"Available:      ${buying_power:.2f}")

        # What can we buy at 40¢?
        max_40 = int(buying_power / 0.40)
        print(f"\nMax contracts @ 40¢: {max_40}")

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
