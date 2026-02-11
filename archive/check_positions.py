#!/usr/bin/env python3
"""Quick script to check current Kalshi positions."""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from kalshi_client import KalshiClient


async def main():
    # Load credentials
    load_dotenv()
    api_key = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

    if not api_key or not private_key_path:
        print("❌ Missing credentials in .env file")
        print("   Need: KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        return

    # Check if we're in demo or live mode
    # If not specified, default to LIVE mode (since demo mode needs explicit opt-in)
    demo_mode = os.getenv("KALSHI_DEMO_MODE", "false").lower() == "true"
    mode_label = "DEMO" if demo_mode else "LIVE"

    print(f"🔍 Fetching positions from Kalshi ({mode_label} mode)...\n")

    # Initialize client
    client = KalshiClient(
        api_key_id=api_key,
        private_key_path=private_key_path,
        demo_mode=demo_mode
    )

    try:
        await client.start()

        # Get balance
        balance = await client.get_balance()
        print(f"💰 Account Balance: ${balance:,.2f}\n")

        # Get positions
        positions = await client.get_positions()

        if not positions:
            print("📭 No open positions")
            return

        print(f"📊 Current Positions ({len(positions)} total):\n")
        print("-" * 80)

        total_realized_pnl = 0
        total_unrealized_pnl = 0
        total_fees = 0

        for i, pos in enumerate(positions, 1):
            ticker = pos.get("ticker", "N/A")
            position = pos.get("position", 0)
            realized_pnl = pos.get("realized_pnl", 0) / 100  # cents to dollars
            market_exposure = pos.get("market_exposure", 0) / 100
            fees_paid = pos.get("fees_paid", 0) / 100

            # Determine if position is open or closed
            if position != 0:
                # Active position - get current market price
                market_data = await client.get_market(ticker)
                yes_bid = market_data.get("yes_bid", 0)
                yes_ask = market_data.get("yes_ask", 0)

                side = "LONG (YES)" if position > 0 else "SHORT (YES)"

                print(f"{i}. {ticker}")
                print(f"   Side: {side} | Contracts: {abs(position)}")
                print(f"   Cost Basis: ${market_exposure:.2f}")

                # Only calculate unrealized P&L if we have valid market prices
                if yes_bid > 0 or yes_ask > 0:
                    print(f"   Current Bid/Ask: {yes_bid}¢ / {yes_ask}¢")
                    if position > 0:
                        current_price = yes_bid if yes_bid else yes_ask
                        market_value = position * current_price / 100
                    else:
                        current_price = yes_ask if yes_ask else yes_bid
                        market_value = abs(position) * current_price / 100
                    unrealized_pnl = market_value - market_exposure
                    print(f"   Market Value: ${market_value:.2f}")
                    print(f"   Unrealized P&L: ${unrealized_pnl:+.2f}")
                    total_unrealized_pnl += unrealized_pnl
                else:
                    print(f"   Market Value: (market closed)")

                print(f"   Fees Paid: ${fees_paid:.2f}")
            else:
                # Closed position - show realized P&L
                print(f"{i}. {ticker}")
                print(f"   Status: CLOSED")
                print(f"   Realized P&L: ${realized_pnl:+.2f}")
                print(f"   Fees Paid: ${fees_paid:.2f}")

            print()
            total_realized_pnl += realized_pnl
            total_fees += fees_paid

        print("-" * 80)
        print(f"Realized P&L:   ${total_realized_pnl:+.2f}")
        print(f"Unrealized P&L: ${total_unrealized_pnl:+.2f}")
        print(f"Total Fees:     ${total_fees:.2f}")
        total_pnl = total_realized_pnl + total_unrealized_pnl - total_fees
        print(f"Net P&L:        ${total_pnl:+.2f}")
        print()
        print(f"Available Cash: ${balance:,.2f}")

    except Exception as e:
        print(f"❌ Error fetching positions: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
