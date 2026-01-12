#!/usr/bin/env python3
"""
==============================================================================
                          PROJECT ATLAS v1.0
               High-Frequency Latency Arbitrage Bot
==============================================================================

Production-ready bot combining all improvements:
- Multi-market monitoring (all BTC markets simultaneously)
- Real-time CEX price streaming (Kraken WebSocket)
- Smart gas optimization (aggressive fees for fast inclusion)
- Velocity-triggered execution
- Position tracking with P/L
- Safety limits and auto-stop

Usage:
    # Monitor mode (no trades):
    python3 -u atlas.py

    # Live trading (requires PRIVATE_KEY in .env):
    python3 -u atlas.py --live

Requirements:
    - .env file with:
        PRIVATE_KEY=0x...  (for live trading)
        RPC_URL=https://...  (optional, defaults to public Base RPC)
"""

import argparse
import asyncio
import json
import os
import re
import time
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pathlib import Path

import aiohttp
import ccxt.pro as ccxtpro
import socketio
from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_typed_data
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider

from gas_optimizer import GasOptimizer
from alerts import AlertManager

# USDC on Base mainnet
USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
USDC_ABI = [{"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"}]

# Kill switch file path
KILL_SWITCH_PATH = Path("/tmp/atlas_stop")

load_dotenv()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Market:
    """A prediction market being tracked."""
    slug: str
    title: str
    strike: float
    expiry_ts: float
    asset: str = "BTC"  # BTC, ETH, or SOL
    yes_price: float = 0.0
    no_price: float = 0.0
    updated_at: float = 0.0
    # For order execution
    yes_token_id: str = ""
    no_token_id: str = ""
    venue_address: str = ""
    collateral_decimals: int = 6  # USDC

    @property
    def ttl_min(self) -> float:
        return (self.expiry_ts - time.time()) / 60

    @property
    def tradeable(self) -> bool:
        return self.ttl_min > 15 and self.yes_price > 0


@dataclass
class Signal:
    """An arbitrage signal."""
    market: Market
    action: str  # BUY_YES or BUY_NO
    cex_price: float
    velocity: float
    expected_prob: float
    market_prob: float
    edge: float
    profit_est: float
    ts: float = field(default_factory=time.time)


@dataclass
class Trade:
    """An executed trade."""
    signal: Signal
    order_id: str
    status: str  # pending, filled, failed
    entry_price: float
    size_usd: float
    gas_cost: float
    ts: float = field(default_factory=time.time)
    pnl: float = 0.0


# =============================================================================
# MAIN BOT
# =============================================================================

class Atlas:
    """
    Project Atlas - Production Arbitrage Bot.
    """

    VERSION = "1.0.0"

    def __init__(self, live_mode: bool = False):
        # Mode
        self.live_mode = live_mode
        self.monitor_only = not live_mode

        # Config
        self.api_url = "https://api.limitless.exchange"
        self.ws_url = "wss://ws.limitless.exchange"
        self.rpc_url = os.getenv("RPC_URL", "https://mainnet.base.org")
        self.private_key = os.getenv("PRIVATE_KEY", "")

        # WebSocket client
        self.sio = socketio.AsyncClient(logger=False, engineio_logger=False)
        self.ws_connected = False

        # Strategy params (optimized based on 0x8dxd bot research)
        self.velocity_threshold = 25.0  # $/sec trigger (lowered from $50)
        self.min_edge = 0.15  # 15% minimum edge (raised from 8%)
        self.position_size = 10.0  # $10 trades
        self.max_losses = 3  # stop after 3 consecutive losses
        self.momentum_confirm_sec = 2.0  # seconds to confirm momentum

        # Supported assets
        self.assets = ["BTC", "ETH", "SOL"]

        # State (per asset)
        self.markets: dict[str, Market] = {}
        self.cex_prices: dict[str, float] = {a: 0.0 for a in self.assets}
        self.cex_velocities: dict[str, float] = {a: 0.0 for a in self.assets}
        self.price_histories: dict[str, list[tuple[float, float]]] = {a: [] for a in self.assets}

        # Momentum confirmation state (per asset)
        self.momentum_start_times: dict[str, float] = {a: 0.0 for a in self.assets}
        self.momentum_directions: dict[str, int] = {a: 0 for a in self.assets}  # 1=bullish, -1=bearish, 0=none
        self.last_trigger_times: dict[str, float] = {a: 0.0 for a in self.assets}

        # Trading state
        self.trades: list[Trade] = []
        self.consecutive_losses = 0
        self.halted = False
        self.halt_reason = ""

        # Circuit breaker
        self.starting_balance = 0.0
        self.max_drawdown_pct = 0.20  # Halt if down 20%

        # Nonce management (for rapid txs)
        self.current_nonce: Optional[int] = None

        # Components
        self.gas: Optional[GasOptimizer] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.web3: Optional[AsyncWeb3] = None

        # Authentication
        self.wallet_address = ""
        self.authenticated = False
        self.usdc_balance = 0.0

        # Discord alerts
        self.alerts = AlertManager()

        # Stats
        self.signals_detected = 0
        self.trades_executed = 0
        self.start_time = 0.0
        self.running = False

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    async def start(self):
        """Initialize and start the bot."""
        self._print_banner()

        # Check kill switch at startup
        if self._check_kill_switch():
            print("[HALT] Kill switch active (/tmp/atlas_stop exists)")
            print("  Remove the file to start: rm /tmp/atlas_stop")
            return

        # Validate
        if self.live_mode and not self.private_key:
            print("[ERROR] Live mode requires PRIVATE_KEY in .env")
            print("  Run without --live for monitor mode")
            return

        # Setup wallet
        if self.private_key:
            try:
                account = Account.from_key(self.private_key)
                self.wallet_address = account.address
                print(f"[INIT] Wallet: {self.wallet_address[:10]}...{self.wallet_address[-6:]}")
            except Exception as e:
                print(f"[ERROR] Invalid private key: {e}")
                return

        # Initialize web3 for on-chain queries
        self.web3 = AsyncWeb3(AsyncHTTPProvider(self.rpc_url))
        try:
            chain_id = await self.web3.eth.chain_id
            print(f"[INIT] Connected to chain {chain_id}")
        except Exception as e:
            print(f"[WARN] Web3 connection issue: {e}")

        # HTTP session
        self.session = aiohttp.ClientSession(headers={
            "User-Agent": "Atlas/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

        # Authenticate for live mode
        if self.live_mode:
            if await self._authenticate():
                print("[INIT] Authenticated with Limitless")
                balance = await self._check_balance()
                if balance <= 0:
                    print(f"[ERROR] Could not verify USDC balance (got ${balance:.2f})")
                    print("  Refusing to trade with unverified balance")
                    return
                if balance < self.position_size:
                    print(f"[ERROR] Insufficient USDC balance: ${balance:.2f}")
                    print(f"  Need at least ${self.position_size} USDC on Base")
                    return
                print(f"[INIT] USDC Balance: ${balance:.2f}")
                # Record starting balance for circuit breaker
                self.starting_balance = balance
                # Initialize nonce
                if self.wallet_address:
                    self.current_nonce = await self.web3.eth.get_transaction_count(self.wallet_address)
                    print(f"[INIT] Starting nonce: {self.current_nonce}")
            else:
                print("[ERROR] Authentication failed")
                return

        # Gas optimizer
        self.gas = GasOptimizer(self.rpc_url)
        await self.gas.start()
        print("[INIT] Gas optimizer started")

        # Discover markets
        await self._discover_markets()
        if not self.markets:
            print("[ERROR] No tradeable markets found")
            return

        print(f"[INIT] Tracking {len(self.markets)} markets")
        print(f"[INIT] Mode: {'LIVE' if self.live_mode else 'MONITOR'}")
        print(f"[INIT] Position size: ${self.position_size}")
        print(f"[INIT] Min edge: {self.min_edge:.0%}")

        # Start
        self.running = True
        self.start_time = time.time()

        print("\n[START] Atlas running. Press Ctrl+C to stop.\n")
        print("=" * 70)

        # Send Discord alert
        mode = "LIVE" if self.live_mode else "MONITOR"
        await self.alerts.bot_started(len(self.markets), mode)

        await self._run()

    async def stop(self):
        """Graceful shutdown."""
        self.running = False

        # Send Discord alert
        reason = self.halt_reason if self.halt_reason else "Manual stop"
        await self.alerts.bot_stopped(reason)

        # Disconnect WebSocket
        if self.ws_connected:
            try:
                await self.sio.disconnect()
            except Exception:
                pass

        if self.gas:
            await self.gas.stop()

        if self.session:
            await self.session.close()

        self._print_summary()

    def _print_banner(self):
        """Print startup banner."""
        print("\n" + "=" * 70)
        print("""
    ██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗
    ██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝
    ██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║
    ██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║
    ██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║
    ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝
                        ATLAS v{version}
                  Latency Arbitrage Engine
        """.format(version=self.VERSION))
        print("=" * 70)

    def _print_summary(self):
        """Print session summary."""
        runtime = time.time() - self.start_time

        print("\n" + "=" * 70)
        print("                      SESSION SUMMARY")
        print("=" * 70)
        print(f"  Runtime:           {runtime/60:.1f} minutes")
        print(f"  Markets tracked:   {len(self.markets)}")
        print(f"  Signals detected:  {self.signals_detected}")
        print(f"  Trades executed:   {self.trades_executed}")

        if self.trades:
            total_pnl = sum(t.pnl for t in self.trades)
            wins = sum(1 for t in self.trades if t.pnl > 0)
            print(f"  Win rate:          {wins}/{len(self.trades)}")
            print(f"  Total P/L:         ${total_pnl:+.2f}")

        print("=" * 70)

    # -------------------------------------------------------------------------
    # MARKET DISCOVERY
    # -------------------------------------------------------------------------

    async def _discover_markets(self):
        """Find all active BTC, ETH, and SOL prediction markets."""
        print("[SCAN] Discovering markets...")

        try:
            data = await self._fetch_json(f"{self.api_url}/markets/active?limit=25")
            now_ms = time.time() * 1000

            for m in data.get("data", []):
                title = m.get("title", "")
                slug = m.get("slug", "")
                exp_ts = m.get("expirationTimestamp", 0)

                # Determine asset type
                asset = None
                if "BTC" in title or "Bitcoin" in title:
                    asset = "BTC"
                elif "ETH" in title or "Ethereum" in title:
                    asset = "ETH"
                elif "SOL" in title or "Solana" in title:
                    asset = "SOL"

                if not asset:
                    continue
                if exp_ts < now_ms + 900000:  # 15 min minimum
                    continue
                if slug in self.markets:  # Already tracking
                    continue

                match = re.search(r"\$?([\d,]+\.?\d*)", title)
                if not match:
                    continue

                prices = m.get("prices", [0, 0])
                tokens = m.get("tokens", {})
                venue = m.get("venue", {})

                market = Market(
                    slug=slug,
                    title=title,
                    strike=float(match.group(1).replace(",", "")),
                    expiry_ts=exp_ts / 1000,
                    asset=asset,
                    no_price=float(prices[0]) if prices else 0,
                    yes_price=float(prices[1]) if len(prices) > 1 else 0,
                    updated_at=time.time(),
                    yes_token_id=str(tokens.get("yes", "")),
                    no_token_id=str(tokens.get("no", "")),
                    venue_address=venue.get("exchange", "") if isinstance(venue, dict) else "",
                )
                self.markets[slug] = market
                print(f"  [+] {asset} ${market.strike:,.0f} | {market.ttl_min:.0f}min | "
                      f"YES:{market.yes_price:.0%} NO:{market.no_price:.0%}")

        except Exception as e:
            print(f"[ERROR] Discovery failed: {e}")

    # -------------------------------------------------------------------------
    # WEBSOCKET CONNECTION
    # -------------------------------------------------------------------------

    async def _connect_websocket(self):
        """Connect to Limitless WebSocket for real-time prices."""
        print("[WS] Connecting to Limitless WebSocket...")

        @self.sio.event
        async def connect():
            print("[WS] Connected!")
            self.ws_connected = True
            # Subscribe to all tracked markets
            for slug in self.markets.keys():
                try:
                    await self.sio.emit("subscribe", {"market": slug})
                    await self.sio.emit("subscribe_market_prices", {"slug": slug})
                    await self.sio.emit("join", slug)
                    print(f"[WS] Subscribed to {slug[:30]}...")
                except Exception as e:
                    print(f"[WS] Subscribe error: {e}")

        @self.sio.event
        async def newPriceData(data):
            """Handle real-time price updates."""
            if isinstance(data, dict):
                slug = data.get("slug", "")
                prices = data.get("prices", [])
                if slug in self.markets and len(prices) >= 2:
                    market = self.markets[slug]
                    market.no_price = float(prices[0])
                    market.yes_price = float(prices[1])
                    market.updated_at = time.time()

        @self.sio.event
        async def orderbookUpdate(data):
            """Handle orderbook updates for CLOB markets."""
            if isinstance(data, dict):
                slug = data.get("slug", "")
                if slug in self.markets:
                    market = self.markets[slug]
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    if asks:
                        market.yes_price = float(asks[0].get("price", market.yes_price))
                    if bids:
                        market.no_price = float(bids[0].get("price", market.no_price))
                    market.updated_at = time.time()

        @self.sio.event
        async def disconnect():
            print("[WS] Disconnected")
            self.ws_connected = False

        try:
            # Try different namespace configurations
            await self.sio.connect(
                self.ws_url,
                transports=["websocket"],
                wait_timeout=10,
            )
            # Give time for subscriptions to process
            await asyncio.sleep(1)
            return True
        except Exception as e:
            print(f"[WS ERROR] {e}")
            print("[WS] Will use REST polling as fallback")
            return False

    async def _subscribe_new_market(self, slug: str):
        """Subscribe to a newly discovered market."""
        if self.ws_connected:
            try:
                await self.sio.emit("subscribe_market_prices", {"slug": slug})
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # AUTHENTICATION & TRADING
    # -------------------------------------------------------------------------

    async def _authenticate(self) -> bool:
        """Authenticate with Limitless API."""
        try:
            # Get signing message
            async with self.session.get(
                f"{self.api_url}/auth/signing-message"
            ) as resp:
                if resp.status != 200:
                    print(f"[AUTH] Failed to get signing message: {resp.status}")
                    return False
                data = await resp.json()
                message = data.get("message", "")
                nonce = data.get("nonce", "")

            if not message or not nonce:
                print("[AUTH] Invalid signing message response")
                return False

            # Sign with wallet
            account = Account.from_key(self.private_key)
            signed = account.sign_message(
                encode_typed_data(full_message={
                    "types": {
                        "EIP712Domain": [
                            {"name": "name", "type": "string"},
                            {"name": "version", "type": "string"},
                        ],
                        "Message": [
                            {"name": "content", "type": "string"},
                            {"name": "nonce", "type": "string"},
                        ],
                    },
                    "primaryType": "Message",
                    "domain": {"name": "Limitless", "version": "1"},
                    "message": {"content": message, "nonce": nonce},
                })
            )

            # Login
            async with self.session.post(
                f"{self.api_url}/auth/login",
                json={
                    "address": self.wallet_address,
                    "signature": "0x" + signed.signature.hex(),
                    "nonce": nonce,
                }
            ) as resp:
                if resp.status == 200:
                    self.authenticated = True
                    return True
                else:
                    text = await resp.text()
                    print(f"[AUTH] Login failed: {resp.status} - {text[:100]}")
                    return False

        except Exception as e:
            print(f"[AUTH ERROR] {e}")
            return False

    def _check_kill_switch(self) -> bool:
        """Check if kill switch file exists."""
        return KILL_SWITCH_PATH.exists()

    async def _check_balance(self) -> float:
        """
        Check USDC balance. Tries API first, falls back to on-chain.
        NEVER returns a fake balance - returns 0.0 if both methods fail.
        """
        # Method 1: Try Limitless API
        try:
            async with self.session.get(
                f"{self.api_url}/portfolio/positions"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    balance = float(data.get("availableBalance", 0))
                    if balance > 0:
                        self.usdc_balance = balance
                        return balance
        except Exception as e:
            print(f"[BALANCE] API check failed: {e}")

        # Method 2: Query on-chain USDC balance
        if self.web3 and self.wallet_address:
            try:
                usdc = self.web3.eth.contract(
                    address=self.web3.to_checksum_address(USDC_ADDRESS),
                    abi=USDC_ABI
                )
                raw_balance = await usdc.functions.balanceOf(self.wallet_address).call()
                # USDC has 6 decimals
                balance = raw_balance / 1e6
                if balance > 0:
                    print(f"[BALANCE] On-chain USDC: ${balance:.2f}")
                    self.usdc_balance = balance
                    return balance
            except Exception as e:
                print(f"[BALANCE] On-chain check failed: {e}")

        # CRITICAL: Never return fake balance - return 0 to halt trading
        print("[BALANCE] Could not verify balance - returning 0 for safety")
        return 0.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _fetch_json(self, url: str) -> dict:
        """Fetch JSON with retry logic."""
        async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _submit_order(self, signal: Signal) -> Optional[str]:
        """Submit order to Limitless API."""
        market = signal.market

        if not market.venue_address:
            print("[ORDER] No venue address - cannot submit")
            return None

        try:
            # Determine token and price
            if signal.action == "BUY_YES":
                token_id = market.yes_token_id
                price = signal.market_prob
            else:
                token_id = market.no_token_id
                price = market.no_price

            # Calculate amounts (in base units)
            size_base = int(self.position_size * (10 ** market.collateral_decimals))

            # Build order
            salt = secrets.randbelow(2**256)
            expiration = str(int(time.time()) + 3600)  # 1 hour expiry
            nonce = int(time.time() * 1000)

            order_data = {
                "salt": str(salt),
                "maker": self.wallet_address,
                "signer": self.wallet_address,
                "taker": "0x0000000000000000000000000000000000000000",
                "tokenId": token_id,
                "makerAmount": str(size_base),
                "takerAmount": str(int(size_base / price)) if price > 0 else "0",
                "expiration": expiration,
                "nonce": nonce,
                "feeRateBps": 100,  # 1% fee
                "side": 0,  # 0 = BUY
                "signatureType": 2,
            }

            # Sign order with EIP-712
            domain = {
                "name": "Limitless Exchange",
                "version": "1",
                "chainId": 8453,  # Base mainnet
                "verifyingContract": market.venue_address,
            }

            order_types = {
                "Order": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                    {"name": "signer", "type": "address"},
                    {"name": "taker", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "makerAmount", "type": "uint256"},
                    {"name": "takerAmount", "type": "uint256"},
                    {"name": "expiration", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "feeRateBps", "type": "uint256"},
                    {"name": "side", "type": "uint8"},
                    {"name": "signatureType", "type": "uint8"},
                ],
            }

            # Create signable message
            account = Account.from_key(self.private_key)
            typed_data = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"},
                    ],
                    **order_types,
                },
                "primaryType": "Order",
                "domain": domain,
                "message": {
                    "salt": int(order_data["salt"]),
                    "maker": order_data["maker"],
                    "signer": order_data["signer"],
                    "taker": order_data["taker"],
                    "tokenId": int(order_data["tokenId"]) if order_data["tokenId"] else 0,
                    "makerAmount": int(order_data["makerAmount"]),
                    "takerAmount": int(order_data["takerAmount"]),
                    "expiration": int(order_data["expiration"]),
                    "nonce": int(order_data["nonce"]),
                    "feeRateBps": int(order_data["feeRateBps"]),
                    "side": int(order_data["side"]),
                    "signatureType": int(order_data["signatureType"]),
                },
            }

            signed_order = account.sign_message(encode_typed_data(full_message=typed_data))
            order_data["signature"] = "0x" + signed_order.signature.hex()
            order_data["price"] = price

            # Submit order
            payload = {
                "order": order_data,
                "orderType": "GTC",  # Good til cancelled
                "marketSlug": market.slug,
            }

            async with self.session.post(
                f"{self.api_url}/orders",
                json=payload
            ) as resp:
                if resp.status in (200, 201):
                    result = await resp.json()
                    order_id = result.get("orderId", result.get("id", f"order_{int(time.time())}"))
                    print(f"[ORDER] Submitted: {order_id}")
                    return str(order_id)
                else:
                    text = await resp.text()
                    print(f"[ORDER] Failed: {resp.status} - {text[:200]}")
                    return None

        except Exception as e:
            print(f"[ORDER ERROR] {e}")
            return None

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    async def _run(self):
        """Main run loop."""
        # Try WebSocket first for real-time data
        ws_success = await self._connect_websocket()

        tasks = [
            asyncio.create_task(self._cex_stream()),
            asyncio.create_task(self._status_display()),
            asyncio.create_task(self._market_refresher()),
            asyncio.create_task(self._kill_switch_monitor()),  # Safety monitor
        ]

        # Always run a fast backup poller (checks staleness)
        tasks.append(asyncio.create_task(self._fast_poller()))

        if ws_success:
            print("[WS] WebSocket connected - using hybrid mode")
            tasks.append(asyncio.create_task(self._ws_keepalive()))
        else:
            print("[FALLBACK] WebSocket failed - using REST polling only")

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    async def _kill_switch_monitor(self):
        """Monitor for kill switch file - check every 5 seconds."""
        while self.running:
            if self._check_kill_switch():
                print("\n[KILL SWITCH] /tmp/atlas_stop detected - shutting down!")
                self.halted = True
                self.halt_reason = "Kill switch activated"
                await self.alerts.kill_switch()
                self.running = False
                break
            await asyncio.sleep(5)

    async def _cex_stream(self):
        """Stream CEX prices for BTC, ETH, SOL and detect velocity triggers."""
        exchange = ccxtpro.kraken({"enableRateLimit": True})
        print("[CEX] Streaming BTC/ETH/SOL from Kraken...")

        # Map symbols to assets
        symbols = {
            "BTC/USD": "BTC",
            "ETH/USD": "ETH",
            "SOL/USD": "SOL",
        }

        try:
            while self.running:
                # Watch all tickers simultaneously
                tickers = await exchange.watch_tickers(list(symbols.keys()))

                now = time.time()

                for symbol, asset in symbols.items():
                    ticker = tickers.get(symbol, {})
                    price = ticker.get("last")
                    if not price:
                        continue

                    self.cex_prices[asset] = price

                    # Velocity calculation per asset
                    self.price_histories[asset].append((now, price))
                    self.price_histories[asset] = [
                        (t, p) for t, p in self.price_histories[asset] if now - t < 5
                    ]

                    if len(self.price_histories[asset]) >= 2:
                        old_t, old_p = self.price_histories[asset][0]
                        dt = now - old_t
                        if dt > 0.3:
                            self.cex_velocities[asset] = (price - old_p) / dt

                            # Check momentum for this asset
                            await self._check_momentum(now, asset)

        except Exception as e:
            print(f"[CEX ERROR] {e}")
        finally:
            await exchange.close()

    async def _ws_keepalive(self):
        """Keep WebSocket connection alive and reconnect if needed."""
        while self.running:
            await asyncio.sleep(30)
            if not self.ws_connected:
                print("[WS] Reconnecting...")
                await self._connect_websocket()

    async def _fast_poller(self):
        """Fast backup poller - only polls stale markets."""
        while self.running:
            now = time.time()
            for slug, market in list(self.markets.items()):
                if not self.running:
                    break

                # Only poll if data is stale (>2 seconds old)
                if now - market.updated_at > 2.0:
                    try:
                        data = await self._fetch_json(f"{self.api_url}/markets/{slug}")
                        prices = data.get("prices", [0, 0])
                        market.no_price = float(prices[0])
                        market.yes_price = float(prices[1]) if len(prices) > 1 else 0
                        market.updated_at = time.time()
                    except Exception:
                        pass  # Retry logic already in _fetch_json

            await asyncio.sleep(0.5)  # Fast polling: 500ms

    async def _market_refresher(self):
        """Refresh market list periodically."""
        while self.running:
            await asyncio.sleep(120)

            # Remove expired
            expired = [s for s, m in self.markets.items() if m.ttl_min < 15]
            for s in expired:
                del self.markets[s]
                print(f"[EXPIRE] {s[:30]}...")

            # Track existing markets before refresh
            existing = set(self.markets.keys())

            await self._discover_markets()

            # Subscribe new markets to WebSocket
            if self.ws_connected:
                new_markets = set(self.markets.keys()) - existing
                for slug in new_markets:
                    await self._subscribe_new_market(slug)

    async def _status_display(self):
        """Display status periodically and update heartbeat."""
        heartbeat_path = Path("/tmp/atlas_heartbeat")

        while self.running:
            await asyncio.sleep(15)

            # Update heartbeat file (external monitors can check this)
            try:
                heartbeat_path.touch()
            except Exception:
                pass

            if self.gas and any(p > 0 for p in self.cex_prices.values()):
                gas_status = self.gas.get_status()
                ws_status = "WS" if self.ws_connected else "REST"

                # Count markets per asset
                asset_counts = {a: 0 for a in self.assets}
                for m in self.markets.values():
                    asset_counts[m.asset] += 1

                # Show prices for all assets
                price_str = " | ".join([
                    f"{a}: ${self.cex_prices[a]:,.0f}"
                    for a in self.assets if self.cex_prices[a] > 0
                ])

                print(f"\n[STATUS] {price_str} | "
                      f"Markets: {len(self.markets)} | "
                      f"Signals: {self.signals_detected} | "
                      f"Feed: {ws_status}")

                # Show markets grouped by asset
                for asset in self.assets:
                    markets_for_asset = [m for m in self.markets.values() if m.asset == asset]
                    if not markets_for_asset:
                        continue

                    cex_price = self.cex_prices[asset]
                    vel = self.cex_velocities[asset]
                    print(f"  {asset} (${cex_price:,.0f}, vel: ${vel:+.1f}/s):")

                    for m in markets_for_asset[:2]:  # Show top 2 per asset
                        diff = cex_price - m.strike
                        age_ms = (time.time() - m.updated_at) * 1000
                        print(f"    ${m.strike:,.0f} ({diff:+.0f}) "
                              f"YES:{m.yes_price:.0%} NO:{m.no_price:.0%} "
                              f"[{m.ttl_min:.0f}m] ({age_ms:.0f}ms)")

    # -------------------------------------------------------------------------
    # MOMENTUM CONFIRMATION
    # -------------------------------------------------------------------------

    async def _check_momentum(self, now: float, asset: str):
        """Check and confirm momentum before triggering for a specific asset."""
        velocity = self.cex_velocities[asset]
        current_direction = 1 if velocity > 0 else -1 if velocity < 0 else 0
        velocity_exceeds = abs(velocity) >= self.velocity_threshold

        # Cooldown: don't trigger too frequently (minimum 10 seconds between triggers per asset)
        if now - self.last_trigger_times[asset] < 10:
            return

        if velocity_exceeds:
            if self.momentum_directions[asset] == 0:
                # First time we see velocity above threshold
                self.momentum_start_times[asset] = now
                self.momentum_directions[asset] = current_direction
                print(f"[MOMENTUM] {asset} {'BULLISH' if current_direction > 0 else 'BEARISH'} "
                      f"(${velocity:+.1f}/s)")

            elif self.momentum_directions[asset] == current_direction:
                # Same direction - check if enough time has passed
                elapsed = now - self.momentum_start_times[asset]
                if elapsed >= self.momentum_confirm_sec:
                    # Momentum confirmed! Trigger
                    print(f"[MOMENTUM] {asset} confirmed after {elapsed:.1f}s")
                    self.last_trigger_times[asset] = now
                    self.momentum_directions[asset] = 0  # Reset
                    await self._on_trigger(asset)

            else:
                # Direction changed - reset
                print(f"[MOMENTUM] {asset} direction changed - resetting")
                self.momentum_directions[asset] = 0

        else:
            # Velocity dropped below threshold - reset
            if self.momentum_directions[asset] != 0:
                self.momentum_directions[asset] = 0

    # -------------------------------------------------------------------------
    # SIGNAL DETECTION & EXECUTION
    # -------------------------------------------------------------------------

    async def _on_trigger(self, asset: str):
        """Handle velocity trigger for a specific asset."""
        if self.halted:
            return

        velocity = self.cex_velocities[asset]
        price = self.cex_prices[asset]
        direction = "BULLISH" if velocity > 0 else "BEARISH"
        print(f"\n[TRIGGER] {asset} {direction} ${velocity:+.1f}/s @ ${price:,.2f}")

        # Find best opportunity for this asset
        best: Optional[Signal] = None

        for market in self.markets.values():
            if not market.tradeable:
                continue
            if market.asset != asset:
                continue

            signal = self._analyze(market)
            if signal and signal.edge >= self.min_edge:
                if not best or signal.edge > best.edge:
                    best = signal

        if not best:
            print(f"[TRIGGER] No {asset} opportunities above threshold")
            return

        self.signals_detected += 1

        print(f"\n[SIGNAL #{self.signals_detected}] {asset} {best.action}")
        print(f"  Market: {best.market.title[:45]}...")
        print(f"  Strike: ${best.market.strike:,.2f}")
        print(f"  Expected: {best.expected_prob:.0%} | Actual: {best.market_prob:.0%}")
        print(f"  Edge: {best.edge:+.1%}")
        print(f"  Est. Profit: ${best.profit_est:+.2f}")

        # Send Discord alert for signal
        await self.alerts.signal_alert(
            action=best.action,
            asset=asset,
            edge=best.edge,
            profit_est=best.profit_est,
            velocity=best.velocity,
        )

        if self.monitor_only:
            print("[MONITOR] Would execute (monitor mode)")
            self._log_signal(best)
        else:
            await self._execute(best)

    def _analyze(self, market: Market) -> Optional[Signal]:
        """Analyze market for opportunity."""
        asset = market.asset
        cex_price = self.cex_prices[asset]
        velocity = self.cex_velocities[asset]

        if cex_price <= 0:
            return None

        diff = cex_price - market.strike

        # Scale thresholds by asset (BTC ~$90k, ETH ~$3k, SOL ~$200)
        # Using percentage of price for consistency
        pct_diff = diff / cex_price if cex_price > 0 else 0

        # Expected probability based on % difference from strike
        if pct_diff > 0.01:      # >1% above
            exp = 0.98
        elif pct_diff > 0.005:   # >0.5% above
            exp = 0.85
        elif pct_diff > 0.002:   # >0.2% above
            exp = 0.70
        elif pct_diff > 0.0005:  # >0.05% above
            exp = 0.58
        elif pct_diff > -0.0005: # near strike
            exp = 0.50
        elif pct_diff > -0.002:  # >0.2% below
            exp = 0.42
        elif pct_diff > -0.005:  # >0.5% below
            exp = 0.30
        elif pct_diff > -0.01:   # >1% below
            exp = 0.15
        else:
            exp = 0.02

        if velocity > 0:  # Bullish -> buy YES
            edge = exp - market.yes_price
            if edge > 0:
                profit = self.position_size * edge - 1.0
                return Signal(
                    market=market,
                    action="BUY_YES",
                    cex_price=cex_price,
                    velocity=velocity,
                    expected_prob=exp,
                    market_prob=market.yes_price,
                    edge=edge,
                    profit_est=profit,
                )
        else:  # Bearish -> buy NO
            edge = (1 - exp) - market.no_price
            if edge > 0:
                profit = self.position_size * edge - 1.0
                return Signal(
                    market=market,
                    action="BUY_NO",
                    cex_price=cex_price,
                    velocity=velocity,
                    expected_prob=1 - exp,
                    market_prob=market.no_price,
                    edge=edge,
                    profit_est=profit,
                )

        return None

    async def _execute(self, signal: Signal):
        """Execute trade on signal."""
        print("[EXECUTE] Placing order...")

        # Circuit breaker: Check drawdown
        if self.starting_balance > 0:
            current = await self._check_balance()
            drawdown = (self.starting_balance - current) / self.starting_balance
            if drawdown >= self.max_drawdown_pct:
                print(f"\n[CIRCUIT BREAKER] Drawdown {drawdown:.1%} exceeds limit {self.max_drawdown_pct:.0%}!")
                print(f"  Started: ${self.starting_balance:.2f} | Current: ${current:.2f}")
                self.halted = True
                self.halt_reason = f"Circuit breaker: {drawdown:.1%} drawdown"
                await self.alerts.circuit_breaker(drawdown, current)
                return

        # Check balance first
        if self.usdc_balance < self.position_size:
            print(f"[EXECUTE] Insufficient balance: ${self.usdc_balance:.2f}")
            return

        # Get gas estimate
        gas_cost = self.gas.estimate_cost_usd()

        # Submit order to Limitless
        order_id = await self._submit_order(signal)

        if order_id:
            trade = Trade(
                signal=signal,
                order_id=order_id,
                status="submitted",
                entry_price=signal.market_prob,
                size_usd=self.position_size,
                gas_cost=gas_cost,
            )
            self.trades.append(trade)
            self.trades_executed += 1
            self.consecutive_losses = 0  # Reset on successful submission

            print(f"[EXECUTE] Order submitted: {order_id}")
            print(f"  Size: ${self.position_size}")
            print(f"  Gas cost: ${gas_cost:.4f}")
        else:
            # Order failed
            self.consecutive_losses += 1
            print(f"[EXECUTE] Order failed (loss #{self.consecutive_losses})")

            if self.consecutive_losses >= self.max_losses:
                print(f"[HALT] Max consecutive losses ({self.max_losses}) reached!")
                self.halted = True

            trade = Trade(
                signal=signal,
                order_id=f"failed_{int(time.time())}",
                status="failed",
                entry_price=signal.market_prob,
                size_usd=0,
                gas_cost=0,
            )
            self.trades.append(trade)

        self._log_trade(trade)

    def _log_signal(self, signal: Signal):
        """Log signal to file."""
        with open("atlas_signals.jsonl", "a") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(),
                "market": signal.market.slug,
                "action": signal.action,
                "cex_price": signal.cex_price,
                "velocity": signal.velocity,
                "edge": signal.edge,
                "profit_est": signal.profit_est,
            }) + "\n")

    def _log_trade(self, trade: Trade):
        """Log trade to file."""
        with open("atlas_trades.jsonl", "a") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(),
                "order_id": trade.order_id,
                "market": trade.signal.market.slug,
                "action": trade.signal.action,
                "entry": trade.entry_price,
                "size": trade.size_usd,
                "gas": trade.gas_cost,
                "status": trade.status,
            }) + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Project Atlas - Latency Arbitrage Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    args = parser.parse_args()

    bot = Atlas(live_mode=args.live)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
