#!/usr/bin/env python3
"""Kalshi API Client - RSA-PSS authenticated trading for prediction markets."""

import asyncio
import base64
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    """Async client for Kalshi trading API."""
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"

    def __init__(self, api_key_id: str = "", private_key_path: str = "", demo_mode: bool = True):
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        self.demo_mode = demo_mode
        self.base_url = self.DEMO_URL if demo_mode else self.BASE_URL
        self.private_key = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0

    async def start(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10, ttl_dns_cache=300, keepalive_timeout=120),
            timeout=aiohttp.ClientTimeout(total=10, connect=2),
        )
        if self.private_key_path and Path(self.private_key_path).exists():
            self.private_key = serialization.load_pem_private_key(
                Path(self.private_key_path).read_bytes(), password=None)

    async def stop(self):
        if self.session:
            await self.session.close()

    def _sign(self, method: str, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method}/trade-api/v2{path.split('?')[0]}"
        sig = base64.b64encode(self.private_key.sign(
            msg.encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())).decode()
        return {"Content-Type": "application/json", "KALSHI-ACCESS-KEY": self.api_key_id,
                "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts}

    async def _req(self, method: str, path: str, data: dict = None, auth: bool = False) -> dict:
        await asyncio.sleep(max(0, 0.1 - (time.time() - self.last_request_time)))
        self.last_request_time = time.time()
        headers = self._sign(method, path) if auth and self.private_key else {"Content-Type": "application/json"}
        try:
            async with getattr(self.session, method.lower())(
                f"{self.base_url}{path}", headers=headers, json=data
            ) as resp:
                return await resp.json() if resp.status in (200, 201) else {}
        except:
            return {}

    async def get_exchange_status(self) -> dict:
        return await self._req("GET", "/exchange/status")

    async def get_markets(self, series_ticker: str = None, status: str = "open", limit: int = 100) -> list:
        params = [f"limit={limit}"]
        if series_ticker:
            params.append(f"series_ticker={series_ticker}")
        if status:
            params.append(f"status={status}")
        return (await self._req("GET", f"/markets?{'&'.join(params)}")).get("markets", [])

    async def get_market(self, ticker: str) -> dict:
        return await self._req("GET", f"/markets/{ticker}")

    async def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        return (await self._req("GET", f"/markets/{ticker}/orderbook?depth={depth}")).get("orderbook", {})

    async def get_balance(self) -> float:
        return (await self._req("GET", "/portfolio/balance", auth=True)).get("balance", 0) / 100.0

    async def get_positions(self) -> list:
        return (await self._req("GET", "/portfolio/positions", auth=True)).get("market_positions", [])

    async def get_orders(self, ticker: str = None) -> list:
        path = f"/portfolio/orders?ticker={ticker}" if ticker else "/portfolio/orders"
        return (await self._req("GET", path, auth=True)).get("orders", [])

    async def place_order(self, ticker: str, side: str, action: str, count: int, price: int, order_type: str = "limit") -> dict:
        data = {"ticker": ticker, "side": side, "action": action, "count": count, "type": order_type}
        if order_type == "limit":
            data["yes_price" if side == "yes" else "no_price"] = price
        return await self._req("POST", "/portfolio/orders", data, auth=True)

    async def cancel_order(self, order_id: str) -> dict:
        return await self._req("DELETE", f"/portfolio/orders/{order_id}", auth=True)

    def parse_market(self, market: dict) -> dict:
        strike_type = market.get("strike_type", "")
        strike = float(market.get("floor_strike" if strike_type in ("greater", "between") else "cap_strike", 0) or 0)
        if strike == 0:
            m = re.search(r"[BT](\d+(?:\.\d+)?)", market.get("ticker", ""))
            strike = float(m.group(1)) if m else 0

        yes_price = (market.get("yes_bid") or market.get("yes_ask") or market.get("last_price") or 50)
        no_price = (market.get("no_bid") or market.get("no_ask") or 50)
        if yes_price > 1:
            yes_price /= 100
        if no_price > 1:
            no_price /= 100

        try:
            expiry_ts = datetime.fromisoformat(market.get("expiration_time", "").replace("Z", "+00:00")).timestamp()
        except:
            expiry_ts = 0

        return {"ticker": market.get("ticker", ""), "title": market.get("title", ""),
                "strike": strike, "strike_type": strike_type, "expiry_ts": expiry_ts,
                "yes_price": yes_price, "no_price": no_price, "volume": market.get("volume", 0)}


if __name__ == "__main__":
    async def test():
        c = KalshiClient(demo_mode=True)
        await c.start()
        print(await c.get_exchange_status())
        await c.stop()
    asyncio.run(test())
