#!/usr/bin/env python3
"""Get detailed information about specific positions."""

import asyncio
import os
from dotenv import load_dotenv
from kalshi_client import KalshiClient


async def main():
    load_dotenv()
    api_key = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

    client = KalshiClient(
        api_key_id=api_key,
        private_key_path=private_key_path,
        demo_mode=False
    )

    try:
        await client.start()

        # Check both positions
        tickers = ["KXHIGHNY-26JAN17-B40.5", "KXHIGHNY-26JAN16-B33.5"]

        for ticker in tickers:
            print(f"\n{'='*80}")
            print(f"Market: {ticker}")
            print('='*80)

            market = await client.get_market(ticker)
            print(f"\nTitle: {market.get('title', 'N/A')}")
            print(f"Status: {market.get('status', 'N/A')}")
            print(f"Close Time: {market.get('close_time', 'N/A')}")
            print(f"Expiration: {market.get('expiration_time', 'N/A')}")
            print(f"Result: {market.get('result', 'Not settled')}")
            print(f"Volume: {market.get('volume', 0)} contracts")
            print(f"\nYes Bid: {market.get('yes_bid', 'N/A')}¢")
            print(f"Yes Ask: {market.get('yes_ask', 'N/A')}¢")
            print(f"Last Price: {market.get('last_price', 'N/A')}¢")

        # Get order history
        print(f"\n{'='*80}")
        print("Recent Orders")
        print('='*80)
        orders = await client.get_orders()
        if orders:
            for order in orders[:10]:
                print(f"\n{order.get('ticker', 'N/A')}")
                print(f"  Side: {order.get('side', 'N/A')} | Action: {order.get('action', 'N/A')}")
                print(f"  Count: {order.get('count', 0)} @ {order.get('yes_price' if order.get('side') == 'yes' else 'no_price', 'N/A')}¢")
                print(f"  Status: {order.get('status', 'N/A')}")
                print(f"  Created: {order.get('created_time', 'N/A')}")
        else:
            print("No recent orders found")

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
