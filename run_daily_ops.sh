#!/bin/bash
# =============================================================================
# NYC WEATHER TRADING - DAILY OPERATIONS
# =============================================================================
# Two-Bot Portfolio:
#   1. Strategist (weather_bot.py) - Model prediction scan
#   2. Sniper (nyc_weather_arb.py) - Latency arbitrage
# =============================================================================

cd /Users/miqadmin/Documents/limitless

echo "=================================================="
echo "      NYC WEATHER TRADING - DAILY OPS"
echo "      $(date '+%Y-%m-%d %H:%M:%S ET')"
echo "=================================================="

# Log file
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_ops_$(date +%Y-%m-%d).log"

# =============================================================================
# PHASE 1: THE STRATEGIST (Model Prediction)
# =============================================================================
echo ""
echo "[PHASE 1] STRATEGIST - Model vs Market Scan"
echo "--------------------------------------------------"
echo "Checking if GFS/ECMWF models disagree with market..."
echo ""

python3 weather_bot.py --once 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "--------------------------------------------------"
echo "[PHASE 1] Complete. Any trades logged above."
echo "--------------------------------------------------"

# =============================================================================
# PHASE 2: THE SNIPER (Latency Arbitrage)
# =============================================================================
echo ""
echo "[PHASE 2] SNIPER - Starting Latency Arb Bot"
echo "--------------------------------------------------"
echo "Monitoring NWS KNYC for real-time opportunities..."
echo "Bot will run until 8 PM ET or Ctrl+C"
echo ""

# Use caffeinate to prevent sleep
export PYTHONUNBUFFERED=1
caffeinate -i python3 nyc_weather_arb.py --interval 60 2>&1 | tee -a "$LOG_FILE"
