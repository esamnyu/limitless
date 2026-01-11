Here is the **Project Atlas** planning document. This is optimized for you to copy-paste directly into a new chat with Claude (or ChatGPT). It contains the full context, the mathematical edge, and the technical constraints so the AI can write high-quality code immediately.

***

# Project Atlas: High-Frequency Latency Arbitrage Bot

## 1. Executive Summary
**Objective:** Build a Python-based arbitrage bot that exploits time-latency discrepancies between Centralized Exchanges (Binance/Coinbase) and On-Chain Prediction Markets (Limitless on Base L2).
**Core Thesis:** On-chain order books lag behind CEX price feeds by 1–3 seconds due to block times and Market Maker latency. We use CEX volatility as a leading indicator to "snipe" stale limit orders on-chain.
**Primary Venue:** Limitless Exchange (Base Network).
**Secondary Venue:** PancakeSwap Prediction (BNB Chain).

## 2. Technical Architecture

### 2.1 Tech Stack
*   **Language:** Python 3.10+ (AsyncIO).
*   **CEX Data (The "Truth"):** `ccxt` (Pro/Async) using WebSockets.
*   **On-Chain Interaction:** `web3.py` (AsyncHTTPProvider).
*   **Signing:** `eth_account`.
*   **Environment:** Linux VPS (Ubuntu) in US-East (N. Virginia).

### 2.2 Infrastructure Requirements
*   **RPC Provider:** **Alchemy** (Base) or **QuickNode** (BNB). *Strict Requirement: Private/Paid endpoints to avoid rate limits and minimize latency.*
*   **Wallet:** Dedicated EOA (Externally Owned Account) for high-frequency signing.
*   **Chain:** Base Mainnet (Chain ID: 8453).

## 3. Strategy 1: Limitless Exchange (Order Book Sniping)

### 3.1 The Mechanism
Limitless uses a **CLOB (Central Limit Order Book)**. Market Makers place limit orders (Bids/Asks) representing probabilities (0.00 to 1.00).

### 3.2 The Trigger Logic
1.  **Monitor:** Listen to Binance Futures WebSocket (`BTC/USDT`).
2.  **Detect:** Calculate `Velocity = Price_Current - Price_1s_Ago`.
3.  **Signal:** `IF abs(Velocity) > Threshold` (e.g., $50 jump in 1s).
4.  **Scan:** Immediately query the Limitless Smart Contract for the "Best Ask" (if Bullish) or "Best Bid" (if Bearish).

### 3.3 The Profit Formula (EV)
Before executing, the bot must pass this strict EV check:

$$ \text{ProjectedProfit} = (\text{Size} \times (\text{TrueProb} - \text{MarketProb})) - \text{TotalCost} $$

*   **TrueProb:** Derived from live Binance Price (e.g., if Price > Strike, Prob $\approx$ 0.99).
*   **MarketProb:** The price of the stale Limit Order on Limitless.
*   **TotalCost:** `GasFee` + `Slippage` + **`AdaptiveTakerFee`**.

### 3.4 Critical Constraints (The "Kill Switches")
*   **Fee Warning:** Limitless uses **Adaptive Fees** that scale up to 3% near expiration.
    *   *Constraint:* `IF Time_To_Expiry < 15 Minutes: ABORT TRADE`.
*   **Liquidity Check:**
    *   *Constraint:* `IF Order_Depth_Amount < Trade_Size: ABORT` (Avoids slippage).

## 4. Strategy 2: PancakeSwap Prediction (Last-Block Sniping)

### 4.1 The Mechanism
Parimutuel "Up/Down" pools. The "Lock Price" is determined at the exact block the round starts.

### 4.2 The Trigger Logic
1.  **Wait:** Idle until `Time_Remaining < 5 Seconds`.
2.  **Compare:** Check `Prize_Pool_Ratio` vs. `Binance_Technical_Indicators` (RSI/Momentum).
3.  **Execute:** If `Pool_Odds > Calculated_Odds`, submit transaction with **High Priority Fee** to land in the final block.

## 5. Execution Strategy (The Gas War)
On Base (EIP-1559), speed is determined by the **Priority Fee**.
*   **Standard Tx:** `MaxFeePerGas = BaseFee + PriorityFee`.
*   **Atlas Tx:** `MaxFeePerGas = (BaseFee * 1.2) + (PriorityFee * 2.0)`.
*   *Goal:* We must pay a premium to be included in the very next block, ahead of other arb bots.

## 6. Implementation Roadmap

### Phase 1: The "Paper Sniper" (Data Collection)
*   **Goal:** Verify the lag exists without spending funds.
*   **Output:** A CSV log: `Timestamp | Binance_Price | Limitless_Best_Ask | Theoretical_Profit`.
*   **Requirement:** Use `web3.py` to read the contract state, do not rely on UI APIs.

### Phase 2: The "Hello World" (Small Cap)
*   **Goal:** Execute the first live trade.
*   **Limits:** Trade Size = $10.00 USDC.
*   **Safety:** Stop trading if 3 consecutive failures (Reverts/Losses).

### Phase 3: Production (Loop)
*   **Goal:** 24/7 Automation.
*   **Upgrade:** Dockerize the bot. Implement "Balance Checks" to auto-stop if ETH for gas runs low.

## 7. Developer Instructions (Prompts)
*   **Code Style:** Modular Python. Separate `strategy.py`, `execution.py`, and `config.py`.
*   **Error Handling:** Wrap all RPC calls in `try/except` blocks. If RPC fails, log it and continue (do not crash).
*   **ABIs:** If the Limitless ABI is not provided, instruct me on how to fetch it from BaseScan using the contract address.

***

### How to use this file:
1.  **Save** the text above into a file named `claude.md` (or `project_plan.md`) in your project folder.
2.  **Start a chat** with Claude (or ChatGPT o1/4o).
3.  **Attach** the file (or paste the text).
4.  **Type this prompt:**
    > "I am building this arbitrage bot. Read the attached `claude.md` for the full context, math, and strategy. Start by helping me write the Python script for **Phase 1 (The Paper Sniper)**. Ensure you use `ccxt` for Binance and `web3.py` for Base."