#!/bin/bash
# =============================================================================
# NYC WEATHER TRADING - DAILY OPERATIONS
# =============================================================================
# Single-Bot Strategy: NYC Sniper (sniper.py) - Latency Arbitrage
# =============================================================================

cd /Users/miqadmin/Documents/limitless

echo "=================================================="
echo "      NYC SNIPER - DAILY OPS"
echo "      $(date '+%Y-%m-%d %H:%M:%S ET')"
echo "=================================================="

# Log file
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_ops_$(date +%Y-%m-%d).log"

# =============================================================================
# NYC SNIPER (Latency Arbitrage)
# =============================================================================
echo ""
echo "[SNIPER] Starting Latency Arb Bot"
echo "--------------------------------------------------"
echo "Monitoring NWS KNYC for real-time opportunities..."
echo "Bot will run until 8 PM ET or Ctrl+C"
echo ""

# Use caffeinate to prevent sleep
export PYTHONUNBUFFERED=1
caffeinate -i python3 sniper.py --interval 60 2>&1 | tee -a "$LOG_FILE"
