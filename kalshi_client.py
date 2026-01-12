#!/usr/bin/env python3
"""
Kalshi API Client - For BTC hourly price prediction markets.

Implements RSA-PSS authentication and key trading endpoints.
Fully legal for US/NYC users.

Usage:
    from kalshi_client import KalshiClient
    client = KalshiClient(api_key_id, private_key_path)
    markets = await client.get_btc_markets()
"""

import base64
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    """Async client for Kalshi trading API."""

    # Production API
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    # Demo API (for testing)
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"

    def __init__(
        self,
        api_key_id: str = "",
        private_key_path: str = "",
        demo_mode: bool = True,
    ):
        """
        Initialize Kalshi client.

        Args:
            api_key_id: Your Kalshi API key ID
            private_key_path: Path to your RSA private key PEM file
            demo_mode: Use demo API (default True for safety)
        """
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        self.demo_mode = demo_mode
        self.base_url = self.DEMO_URL if demo_mode else self.BASE_URL

        self.private_key = None
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    async def start(self):
        """Initialize the client session and load keys."""
        self.session = aiohttp.ClientSession()

        if self.private_key_path and Path(self.private_key_path).exists():
            with open(self.private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                )
            print(f"[KALSHI] Loaded private key from {self.private_key_path}")

        mode = "DEMO" if self.demo_mode else "PRODUCTION"
        print(f"[KALSHI] Client initialized ({mode} mode)")

    async def stop(self):
        """Close the client session."""
        if self.session:
            await self.session.close()

    def _sign_request(self, method: str, path: str) -> dict:
        """
        Generate RSA-PSS signature for authenticated requests.

        Returns headers dict with authentication.
        """
        timestamp_ms = int(time.time() * 1000)
        timestamp_str = str(timestamp_ms)

        # Message to sign: timestamp + method + path (without query params)
        path_without_query = path.split("?")[0]
        message = f"{timestamp_str}{method}{path_without_query}"

        # Sign with RSA-PSS
        signature = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        signature_b64 = base64.b64encode(signature).decode("utf-8")

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        data: dict = None,
        authenticated: bool = False,
    ) -> dict:
        """Make an API request with optional authentication."""
        # Rate limiting
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

        url = f"{self.base_url}{path}"

        headers = {"Content-Type": "application/json"}
        if authenticated and self.private_key:
            headers = self._sign_request(method, path)

        try:
            if method == "GET":
                async with self.session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        text = await resp.text()
                        print(f"[KALSHI] {method} {path} failed: {resp.status} - {text[:200]}")
                        return {}
            elif method == "POST":
                async with self.session.post(url, headers=headers, json=data) as resp:
                    if resp.status in (200, 201):
                        return await resp.json()
                    else:
                        text = await resp.text()
                        print(f"[KALSHI] {method} {path} failed: {resp.status} - {text[:200]}")
                        return {}
            elif method == "DELETE":
                async with self.session.delete(url, headers=headers) as resp:
                    if resp.status in (200, 204):
                        return await resp.json() if resp.status == 200 else {}
                    else:
                        text = await resp.text()
                        print(f"[KALSHI] {method} {path} failed: {resp.status}")
                        return {}
        except Exception as e:
            print(f"[KALSHI ERROR] {method} {path}: {e}")
            return {}

    # -------------------------------------------------------------------------
    # PUBLIC ENDPOINTS (no auth required)
    # -------------------------------------------------------------------------

    async def get_exchange_status(self) -> dict:
        """Get exchange operational status."""
        return await self._request("GET", "/exchange/status")

    async def get_markets(
        self,
        series_ticker: str = None,
        status: str = "open",
        limit: int = 100,
    ) -> list:
        """
        Get list of markets.

        Args:
            series_ticker: Filter by series (e.g., 'KXBTC' for Bitcoin)
            status: Market status filter ('open', 'closed', etc.)
            limit: Max results to return
        """
        params = [f"limit={limit}"]
        if series_ticker:
            params.append(f"series_ticker={series_ticker}")
        if status:
            params.append(f"status={status}")

        query = "&".join(params)
        result = await self._request("GET", f"/markets?{query}")
        return result.get("markets", [])

    async def get_btc_markets(self) -> list:
        """Get all open BTC price prediction markets."""
        # Kalshi BTC markets use series ticker 'KXBTC' or similar
        markets = await self.get_markets(limit=200)

        btc_markets = []
        for m in markets:
            title = m.get("title", "").upper()
            ticker = m.get("ticker", "").upper()

            # Filter for BTC price markets
            if "BTC" in title or "BITCOIN" in title or "KXBTC" in ticker:
                btc_markets.append(m)

        return btc_markets

    async def get_market(self, ticker: str) -> dict:
        """Get details for a specific market."""
        return await self._request("GET", f"/markets/{ticker}")

    async def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """
        Get orderbook for a market.

        Returns:
            {
                "orderbook": {
                    "yes": [[price, quantity], ...],
                    "no": [[price, quantity], ...]
                }
            }
        """
        result = await self._request("GET", f"/markets/{ticker}/orderbook?depth={depth}")
        return result.get("orderbook", {"yes": [], "no": []})

    async def get_series(self, series_ticker: str) -> dict:
        """Get series information."""
        return await self._request("GET", f"/series/{series_ticker}")

    # -------------------------------------------------------------------------
    # AUTHENTICATED ENDPOINTS
    # -------------------------------------------------------------------------

    async def get_balance(self) -> float:
        """Get account balance in dollars."""
        result = await self._request("GET", "/portfolio/balance", authenticated=True)
        # Balance is in cents
        cents = result.get("balance", 0)
        return cents / 100.0

    async def get_positions(self) -> list:
        """Get current positions."""
        result = await self._request("GET", "/portfolio/positions", authenticated=True)
        return result.get("market_positions", [])

    async def get_orders(self, ticker: str = None) -> list:
        """Get open orders."""
        path = "/portfolio/orders"
        if ticker:
            path += f"?ticker={ticker}"
        result = await self._request("GET", path, authenticated=True)
        return result.get("orders", [])

    async def place_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        action: str,  # "buy" or "sell"
        count: int,  # Number of contracts
        price: int,  # Price in cents (1-99)
        order_type: str = "limit",
    ) -> dict:
        """
        Place an order.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            price: Price in cents (1-99)
            order_type: "limit" or "market"

        Returns:
            Order confirmation dict
        """
        data = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
        }

        if order_type == "limit":
            data["yes_price"] = price if side == "yes" else None
            data["no_price"] = price if side == "no" else None

        return await self._request("POST", "/portfolio/orders", data=data, authenticated=True)

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        return await self._request("DELETE", f"/portfolio/orders/{order_id}", authenticated=True)

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def parse_market(self, market: dict) -> dict:
        """
        Parse market data into standardized format.

        Returns:
            {
                "ticker": str,
                "title": str,
                "strike": float,
                "expiry_ts": float,
                "yes_price": float,
                "no_price": float,
                "volume": int,
            }
        """
        title = market.get("title", "")

        # Get strike from floor_strike field (proper API field)
        strike = float(market.get("floor_strike", 0))

        # Fallback: extract from title if floor_strike not available
        if strike == 0:
            import re
            strike_match = re.search(r"\$?([\d,]+)", title)
            strike = float(strike_match.group(1).replace(",", "")) if strike_match else 0

        # Parse expiry from close_time or expiration_time
        expiry_str = market.get("expiration_time", market.get("close_time", ""))
        try:
            expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            expiry_ts = expiry_dt.timestamp()
        except:
            expiry_ts = 0

        # Get prices - try both cents and dollars fields
        yes_price = market.get("yes_bid", 0)
        no_price = market.get("no_bid", 0)

        # If values are > 1, they're in cents (0-100 scale)
        if yes_price > 1:
            yes_price = yes_price / 100.0
        if no_price > 1:
            no_price = no_price / 100.0

        # Use ask prices if bids are 0
        if yes_price == 0:
            yes_price = market.get("yes_ask", 50)
            if yes_price > 1:
                yes_price = yes_price / 100.0
        if no_price == 0:
            no_price = market.get("no_ask", 50)
            if no_price > 1:
                no_price = no_price / 100.0

        # Last resort: use last_price
        if yes_price == 0 and no_price == 0:
            last = market.get("last_price", 50)
            if last > 1:
                last = last / 100.0
            yes_price = last
            no_price = 1 - last

        return {
            "ticker": market.get("ticker", ""),
            "title": title,
            "strike": strike,
            "expiry_ts": expiry_ts,
            "expiry_str": expiry_str,
            "yes_price": yes_price,
            "no_price": no_price,
            "volume": market.get("volume", 0),
            "status": market.get("status", ""),
        }


# For testing
import asyncio

async def test_client():
    """Test the Kalshi client with public endpoints."""
    client = KalshiClient(demo_mode=True)
    await client.start()

    print("\n[TEST] Exchange Status:")
    status = await client.get_exchange_status()
    print(f"  {status}")

    print("\n[TEST] BTC Markets:")
    markets = await client.get_btc_markets()
    print(f"  Found {len(markets)} BTC markets")

    for m in markets[:5]:
        parsed = client.parse_market(m)
        print(f"  - {parsed['ticker']}: {parsed['title'][:40]}...")
        print(f"    Strike: ${parsed['strike']:,.0f} | YES: {parsed['yes_price']:.0%} | NO: {parsed['no_price']:.0%}")

    await client.stop()


if __name__ == "__main__":
    asyncio.run(test_client())
