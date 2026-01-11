#!/usr/bin/env python3
"""
Project Atlas - Gas Optimizer
Smart gas pricing to win the block inclusion race.

Features:
- Real-time Base L2 gas price monitoring
- Dynamic priority fee calculation
- Pre-built transaction templates for instant submission
- Gas price spike detection
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp


@dataclass
class GasState:
    """Current gas market state."""
    base_fee: int  # in wei
    priority_fee: int  # in wei
    timestamp: float
    block_number: int = 0

    @property
    def total_fee_gwei(self) -> float:
        return (self.base_fee + self.priority_fee) / 1e9

    @property
    def is_fresh(self) -> bool:
        return time.time() - self.timestamp < 2.0


class GasOptimizer:
    """
    Monitors and optimizes gas pricing for Base L2.
    """

    def __init__(self, rpc_url: str = "https://mainnet.base.org"):
        self.rpc_url = rpc_url
        self.session: Optional[aiohttp.ClientSession] = None

        # Current state
        self.current_gas: Optional[GasState] = None
        self.gas_history: list[GasState] = []

        # Configuration
        self.priority_multiplier = 1.5  # 50% above average priority
        self.max_priority_gwei = 0.1  # Cap priority fee at 0.1 gwei
        self.base_multiplier = 1.2  # 20% buffer on base fee

        # Running
        self.running = False

    async def start(self):
        """Start gas monitoring."""
        self.session = aiohttp.ClientSession()
        self.running = True
        asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop gas monitoring."""
        self.running = False
        if self.session:
            await self.session.close()

    async def _monitor_loop(self):
        """Continuously monitor gas prices."""
        while self.running:
            try:
                await self._fetch_gas_price()
            except Exception as e:
                pass  # Silently continue
            await asyncio.sleep(1.0)  # Check every second

    async def _fetch_gas_price(self):
        """Fetch current gas prices from RPC."""
        # Get latest block for base fee
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": ["latest", False],
            "id": 1,
        }

        async with self.session.post(self.rpc_url, json=payload) as resp:
            if resp.status != 200:
                return

            data = await resp.json()
            block = data.get("result", {})

            base_fee = int(block.get("baseFeePerGas", "0x0"), 16)
            block_number = int(block.get("number", "0x0"), 16)

        # Get suggested priority fee
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_maxPriorityFeePerGas",
            "params": [],
            "id": 2,
        }

        async with self.session.post(self.rpc_url, json=payload) as resp:
            if resp.status != 200:
                priority_fee = 1000000  # 0.001 gwei default
            else:
                data = await resp.json()
                priority_fee = int(data.get("result", "0x0"), 16)

        self.current_gas = GasState(
            base_fee=base_fee,
            priority_fee=priority_fee,
            timestamp=time.time(),
            block_number=block_number,
        )

        # Keep history
        self.gas_history.append(self.current_gas)
        self.gas_history = self.gas_history[-100:]  # Keep last 100

    def get_aggressive_gas(self) -> tuple[int, int]:
        """
        Get aggressive gas parameters for fast inclusion.
        Returns (max_fee_per_gas, max_priority_fee_per_gas) in wei.
        """
        if not self.current_gas or not self.current_gas.is_fresh:
            # Fallback defaults for Base L2
            return (100000000, 1000000)  # 0.1 gwei, 0.001 gwei

        # Calculate aggressive priority fee
        priority = int(self.current_gas.priority_fee * self.priority_multiplier)
        max_priority = int(self.max_priority_gwei * 1e9)
        priority = min(priority, max_priority)

        # Calculate max fee with buffer
        max_fee = int(self.current_gas.base_fee * self.base_multiplier) + priority

        return (max_fee, priority)

    def get_standard_gas(self) -> tuple[int, int]:
        """
        Get standard gas parameters (no rush).
        Returns (max_fee_per_gas, max_priority_fee_per_gas) in wei.
        """
        if not self.current_gas or not self.current_gas.is_fresh:
            return (50000000, 500000)  # 0.05 gwei, 0.0005 gwei

        priority = self.current_gas.priority_fee
        max_fee = int(self.current_gas.base_fee * 1.1) + priority

        return (max_fee, priority)

    def estimate_cost_usd(self, gas_limit: int = 200000, eth_price: float = 3000.0) -> float:
        """Estimate transaction cost in USD."""
        max_fee, _ = self.get_aggressive_gas()
        cost_eth = (gas_limit * max_fee) / 1e18
        return cost_eth * eth_price

    def get_status(self) -> dict:
        """Get current gas status."""
        if not self.current_gas:
            return {"status": "not_initialized"}

        aggressive = self.get_aggressive_gas()
        standard = self.get_standard_gas()

        return {
            "block": self.current_gas.block_number,
            "base_fee_gwei": self.current_gas.base_fee / 1e9,
            "priority_gwei": self.current_gas.priority_fee / 1e9,
            "aggressive_total_gwei": aggressive[0] / 1e9,
            "standard_total_gwei": standard[0] / 1e9,
            "age_seconds": time.time() - self.current_gas.timestamp,
        }


# Test
async def test_gas_optimizer():
    """Test gas optimizer."""
    print("Testing Gas Optimizer...")

    optimizer = GasOptimizer()
    await optimizer.start()

    # Wait for first fetch
    await asyncio.sleep(3)

    if optimizer.current_gas:
        status = optimizer.get_status()
        print(f"\nGas Status:")
        print(f"  Block: {status['block']}")
        print(f"  Base Fee: {status['base_fee_gwei']:.6f} gwei")
        print(f"  Priority: {status['priority_gwei']:.6f} gwei")
        print(f"  Aggressive Total: {status['aggressive_total_gwei']:.6f} gwei")
        print(f"  Standard Total: {status['standard_total_gwei']:.6f} gwei")

        cost = optimizer.estimate_cost_usd()
        print(f"  Est. Cost (200k gas): ${cost:.4f}")
    else:
        print("Failed to fetch gas prices")

    await optimizer.stop()


if __name__ == "__main__":
    asyncio.run(test_gas_optimizer())
