#!/bin/bash
# =============================================================================
# NYC Weather Arb Bot - Local Paper Trading Runner
# =============================================================================
#
# This script runs the bot in paper trading mode on your Mac.
# It logs all activity and can run in the background.
#
# Usage:
#   ./run_paper_trading.sh          # Run in foreground
#   ./run_paper_trading.sh &        # Run in background
#   ./run_paper_trading.sh status   # Check if running
#   ./run_paper_trading.sh stop     # Stop the bot
#   ./run_paper_trading.sh logs     # View recent logs
#   ./run_paper_trading.sh results  # Show paper trade summary
# =============================================================================

cd "$(dirname "$0")"

LOG_FILE="paper_trading.log"
PID_FILE=".paper_trading.pid"
TRADES_FILE="nyc_arb_trades.jsonl"

case "${1:-run}" in
    run)
        echo "Starting NYC Weather Arb Bot (Paper Mode)..."
        echo "Logs: $LOG_FILE"
        echo "Trades: $TRADES_FILE"
        echo ""
        echo "Press Ctrl+C to stop, or run: ./run_paper_trading.sh stop"
        echo ""

        # Run with unbuffered output for real-time logs
        PYTHONUNBUFFERED=1 python3 nyc_weather_arb.py --interval 60 2>&1 | tee -a "$LOG_FILE"
        ;;

    background)
        if [ -f "$PID_FILE" ]; then
            OLD_PID=$(cat "$PID_FILE")
            if kill -0 "$OLD_PID" 2>/dev/null; then
                echo "Bot already running (PID: $OLD_PID)"
                exit 1
            fi
        fi

        echo "Starting bot in background..."
        nohup python3 nyc_weather_arb.py --interval 60 >> "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "Started! PID: $(cat $PID_FILE)"
        echo "View logs: tail -f $LOG_FILE"
        ;;

    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "Stopping bot (PID: $PID)..."
                kill "$PID"
                rm "$PID_FILE"
                echo "Stopped."
            else
                echo "Bot not running (stale PID file)"
                rm "$PID_FILE"
            fi
        else
            echo "No PID file found. Bot not running."
            # Try to find and kill anyway
            pkill -f "nyc_weather_arb.py" 2>/dev/null && echo "Killed orphan process."
        fi
        ;;

    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "Bot is RUNNING (PID: $PID)"
                echo ""
                echo "Recent activity:"
                tail -5 "$LOG_FILE" 2>/dev/null
            else
                echo "Bot is NOT running (stale PID)"
            fi
        else
            echo "Bot is NOT running"
        fi
        ;;

    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -50 "$LOG_FILE"
        else
            echo "No log file yet."
        fi
        ;;

    logs-follow)
        tail -f "$LOG_FILE"
        ;;

    results)
        echo "=========================================="
        echo "     PAPER TRADING RESULTS"
        echo "=========================================="

        if [ ! -f "$TRADES_FILE" ]; then
            echo "No trades recorded yet."
            exit 0
        fi

        TOTAL=$(wc -l < "$TRADES_FILE" | tr -d ' ')
        echo "Total paper trades: $TOTAL"
        echo ""

        if [ "$TOTAL" -gt 0 ]; then
            echo "Recent trades:"
            echo "-------------------------------------------"
            tail -10 "$TRADES_FILE" | python3 -c "
import sys
import json
for line in sys.stdin:
    try:
        t = json.loads(line.strip())
        ts = t.get('ts', '')[:16]
        ticker = t.get('ticker', 'N/A')[-15:]
        side = t.get('side', 'yes').upper()
        temp = t.get('nws_temp', 0)
        price = t.get('entry_price', 0) * 100
        print(f'{ts} | {ticker} | {side:3} @ {price:2.0f}¢ | NWS: {temp}°F')
    except:
        pass
"
            echo ""
            echo "To analyze win rate, wait for settlements (next morning)"
        fi
        ;;

    *)
        echo "Usage: $0 {run|background|stop|status|logs|logs-follow|results}"
        exit 1
        ;;
esac
