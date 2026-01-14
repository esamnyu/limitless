#!/bin/bash
# =============================================================================
# Auto-start wrapper for NYC Weather Arb Bot
# Called by launchd at 9:55 AM ET on weekdays
# =============================================================================

cd /Users/miqadmin/Documents/limitless

# Log file with date
LOG_FILE="logs/arb_$(date +%Y-%m-%d).log"
mkdir -p logs

echo "========================================" >> "$LOG_FILE"
echo "Bot started at $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Run until 8 PM ET (market close), then exit
# The bot's internal logic handles market hours, but we'll let it run all day
export PYTHONUNBUFFERED=1

# Use caffeinate to prevent sleep during execution
caffeinate -i python3 nyc_weather_arb.py --interval 60 >> "$LOG_FILE" 2>&1 &
BOT_PID=$!

echo $BOT_PID > .arb_bot.pid
echo "Bot PID: $BOT_PID" >> "$LOG_FILE"

# Auto-stop at 8 PM ET (20:00)
# Calculate seconds until 8 PM
TARGET_HOUR=20
CURRENT_HOUR=$(date +%H)
CURRENT_MIN=$(date +%M)
CURRENT_SEC=$(date +%S)

if [ $CURRENT_HOUR -lt $TARGET_HOUR ]; then
    SECONDS_UNTIL_STOP=$(( (TARGET_HOUR - CURRENT_HOUR) * 3600 - CURRENT_MIN * 60 - CURRENT_SEC ))
    echo "Will auto-stop in $((SECONDS_UNTIL_STOP / 3600)) hours" >> "$LOG_FILE"

    # Sleep then kill
    (sleep $SECONDS_UNTIL_STOP && kill $BOT_PID 2>/dev/null && echo "Bot stopped at $(date)" >> "$LOG_FILE") &
fi
