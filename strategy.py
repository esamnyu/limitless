"""
Project Atlas - Strategy Module
Contains the core velocity detection and arbitrage signal logic.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class PricePoint:
    """A timestamped price observation."""
    timestamp: float  # Unix timestamp
    price: float


@dataclass
class ArbitrageSignal:
    """Represents a detected arbitrage opportunity."""
    timestamp: float
    binance_price: float
    binance_velocity: float  # $/second
    limitless_best_ask: float
    limitless_best_bid: float
    signal_type: str  # "BULLISH" or "BEARISH"
    theoretical_profit: float
    market_prob: float  # Current on-chain probability
    true_prob: float    # Derived from CEX price


class VelocityDetector:
    """
    Detects rapid price movements on Binance.
    Calculates velocity = (price_now - price_1s_ago) / time_delta
    """

    def __init__(self, threshold_usd: float = 50.0, window_size: int = 10):
        self.threshold = threshold_usd
        self.price_history: deque[PricePoint] = deque(maxlen=window_size)
        self.last_signal_time: float = 0
        self.cooldown_seconds: float = 2.0  # Prevent signal spam

    def add_price(self, price: float) -> Optional[tuple[float, str]]:
        """
        Add a new price point and check for velocity trigger.

        Returns:
            Tuple of (velocity, direction) if threshold exceeded, None otherwise.
        """
        now = time.time()
        self.price_history.append(PricePoint(timestamp=now, price=price))

        if len(self.price_history) < 2:
            return None

        # Find price from ~1 second ago
        one_second_ago = now - 1.0
        old_price = None

        for pp in self.price_history:
            if pp.timestamp <= one_second_ago:
                old_price = pp
            else:
                break

        if old_price is None:
            # Not enough history yet
            old_price = self.price_history[0]

        # Calculate velocity
        time_delta = now - old_price.timestamp
        if time_delta < 0.1:  # Need at least 100ms of data
            return None

        velocity = (price - old_price.price) / time_delta

        # Check if threshold exceeded and cooldown passed
        if abs(velocity) >= self.threshold and (now - self.last_signal_time) > self.cooldown_seconds:
            self.last_signal_time = now
            direction = "BULLISH" if velocity > 0 else "BEARISH"
            return (velocity, direction)

        return None


class ArbitrageAnalyzer:
    """
    Analyzes potential arbitrage between CEX price and on-chain order book.
    """

    def __init__(self, strike_price: float = 90000.0):
        """
        Args:
            strike_price: The strike price of the prediction market (e.g., BTC > 90000)
        """
        self.strike_price = strike_price

    def calculate_true_probability(self, cex_price: float) -> float:
        """
        Estimate the "true" probability based on CEX price relative to strike.

        For a "BTC > X" market:
        - If current price >> strike: probability approaches 1.0
        - If current price << strike: probability approaches 0.0
        - Near strike: use a sigmoid-like model

        This is a simplified model. In production, you'd use implied vol.
        """
        # Simple linear model within $1000 of strike
        distance = cex_price - self.strike_price
        buffer = 1000.0  # $1000 buffer zone

        if distance > buffer:
            return 0.99
        elif distance < -buffer:
            return 0.01
        else:
            # Linear interpolation
            return 0.5 + (distance / buffer) * 0.49

    def analyze_opportunity(
        self,
        binance_price: float,
        binance_velocity: float,
        limitless_best_ask: float,
        limitless_best_bid: float,
        gas_cost_usd: float = 0.10,
        taker_fee_pct: float = 0.01,
        trade_size_usd: float = 100.0,
    ) -> Optional[ArbitrageSignal]:
        """
        Analyze if there's a profitable arbitrage opportunity.

        Args:
            binance_price: Current BTC price on Binance
            binance_velocity: Price velocity ($/second)
            limitless_best_ask: Best ask on Limitless (buy YES tokens)
            limitless_best_bid: Best bid on Limitless (buy NO / sell YES)
            gas_cost_usd: Estimated gas cost in USD
            taker_fee_pct: Taker fee percentage (0.01 = 1%)
            trade_size_usd: Position size

        Returns:
            ArbitrageSignal if profitable, None otherwise.
        """
        true_prob = self.calculate_true_probability(binance_price)
        signal_type = "BULLISH" if binance_velocity > 0 else "BEARISH"

        if signal_type == "BULLISH":
            # We want to BUY YES tokens (take the ask)
            market_prob = limitless_best_ask
            if market_prob <= 0 or market_prob >= 1:
                return None

            # Edge = true_prob - market_prob
            edge = true_prob - market_prob

        else:
            # BEARISH: We want to BUY NO tokens (equivalent to selling YES at bid)
            market_prob = limitless_best_bid
            if market_prob <= 0 or market_prob >= 1:
                return None

            # For NO tokens, edge is inverse
            edge = market_prob - true_prob

        # Calculate theoretical profit
        # Profit = Size * Edge - Costs
        taker_fee_usd = trade_size_usd * taker_fee_pct
        total_cost = gas_cost_usd + taker_fee_usd
        gross_profit = trade_size_usd * edge
        theoretical_profit = gross_profit - total_cost

        return ArbitrageSignal(
            timestamp=time.time(),
            binance_price=binance_price,
            binance_velocity=binance_velocity,
            limitless_best_ask=limitless_best_ask,
            limitless_best_bid=limitless_best_bid,
            signal_type=signal_type,
            theoretical_profit=theoretical_profit,
            market_prob=market_prob,
            true_prob=true_prob,
        )
