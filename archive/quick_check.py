#!/usr/bin/env python3
"""Quick check of positions and markets."""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from kalshi_client import KalshiClient
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

load_dotenv()

async def main():
    api_key = os.getenv('KALSHI_API_KEY_ID')
    private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')

    client = KalshiClient(api_key_id=api_key, private_key_path=private_key_path, demo_mode=False)
    await client.start()

    # Get balance
    balance = await client.get_balance()
    print(f'=== ACCOUNT ===')
    print(f'Balance: ${balance:.2f}')

    # Get positions
    positions = await client.get_positions()
    print(f'\n=== POSITIONS ({len(positions)} total) ===')
    active_tickers = []
    for pos in positions:
        if pos.get('position', 0) != 0:
            ticker = pos.get('ticker', 'N/A')
            active_tickers.append(ticker)
            position = pos.get('position', 0)
            market_exposure = pos.get('market_exposure', 0) / 100
            fees = pos.get('fees_paid', 0) / 100
            print(f'Ticker: {ticker}')
            print(f'  Position: {position} contracts')
            print(f'  Cost Basis: ${market_exposure:.2f}')
            print(f'  Fees: ${fees:.2f}')
            # Get orderbook for this ticker
            ob = await client.get_orderbook(ticker)
            yes_bids = ob.get('yes', [])
            no_bids = ob.get('no', [])
            if yes_bids:
                print(f'  Yes Bids: {yes_bids[:3]}')
            if no_bids:
                print(f'  No Bids: {no_bids[:3]}')
            print()

    if not active_tickers:
        print('No active positions.')

    # Get tomorrow's weather markets
    print('\n=== TOMORROW WEATHER MARKETS (KXHIGHNY) ===')
    markets = await client.get_markets(series_ticker='KXHIGHNY', status='open', limit=50)

    tz = ZoneInfo('America/New_York')
    now = datetime.now(tz)
    tomorrow = now + timedelta(days=1)
    months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    tomorrow_str = f'{tomorrow.year % 100:02d}{months[tomorrow.month-1]}{tomorrow.day:02d}'

    print(f'Target date: {tomorrow_str} (Jan 22)')
    print()

    tomorrow_markets = []
    for m in markets:
        ticker = m.get('ticker', '')
        if tomorrow_str in ticker:
            tomorrow_markets.append(m)
            subtitle = m.get('subtitle', '')
            yes_bid = m.get('yes_bid', 0)
            yes_ask = m.get('yes_ask', 0)
            print(f'{ticker}')
            print(f'  {subtitle}')
            print(f'  Bid: {yes_bid}c | Ask: {yes_ask}c | Spread: {yes_ask - yes_bid}c')
            print()

    if not tomorrow_markets:
        print('No markets found for tomorrow.')

    await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
