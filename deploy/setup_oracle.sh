#!/bin/bash
# =============================================================================
# WEATHER EDGE — Oracle Cloud Free Tier Deployment
# ARM64 (Ampere A1) instance with Ubuntu 22.04+
# =============================================================================
set -euo pipefail

DEPLOY_DIR="/home/ubuntu/limitless"
VENV_DIR="$DEPLOY_DIR/.venv"
LOG_DIR="/var/log/weather-edge"

echo "═══════════════════════════════════════════"
echo "  Weather Edge — Oracle Cloud Setup"
echo "═══════════════════════════════════════════"

# ─────────────────────────────────────────────
# 1. SYSTEM UPDATES & DEPENDENCIES
# ─────────────────────────────────────────────
echo "[1/7] System packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3 python3-pip python3-venv \
    chrony curl git \
    logrotate

# ─────────────────────────────────────────────
# 2. TIME SYNCHRONIZATION (critical for API signing)
# ─────────────────────────────────────────────
echo "[2/7] Time sync (chrony)..."
sudo systemctl enable chrony
sudo systemctl start chrony
sudo chronyc makestep 2>/dev/null || true

# Set timezone to ET for readable logs (cron still uses ET times)
sudo timedatectl set-timezone America/New_York
echo "  Timezone: $(timedatectl show --value -p Timezone)"
echo "  Time sync: $(chronyc tracking 2>/dev/null | grep 'System time' || echo 'OK')"

# ─────────────────────────────────────────────
# 3. NETWORK TUNING
# ─────────────────────────────────────────────
echo "[3/7] Network optimizations..."
sudo tee /etc/sysctl.d/99-weather-edge.conf > /dev/null << 'EOF'
# Weather Edge — network tuning for API latency
net.ipv4.tcp_keepalive_time=60
net.ipv4.tcp_keepalive_intvl=10
net.ipv4.tcp_keepalive_probes=6
net.ipv4.tcp_fastopen=3
net.core.rmem_max=16777216
net.core.wmem_max=16777216
EOF
sudo sysctl --system > /dev/null 2>&1

# ─────────────────────────────────────────────
# 4. CREATE PROJECT STRUCTURE
# ─────────────────────────────────────────────
echo "[4/7] Project structure..."
sudo mkdir -p "$LOG_DIR"
sudo chown ubuntu:ubuntu "$LOG_DIR"
mkdir -p "$DEPLOY_DIR"

if [ ! -d "$DEPLOY_DIR/.git" ] && [ ! -f "$DEPLOY_DIR/config.py" ]; then
    echo ""
    echo "  ⚠ No code found at $DEPLOY_DIR"
    echo "  Upload your code first with deploy.sh, then re-run this script."
    echo "  Or: scp -r /path/to/limitless/* ubuntu@<ip>:~/limitless/"
    echo ""
fi

# ─────────────────────────────────────────────
# 5. PYTHON VIRTUAL ENVIRONMENT
# ─────────────────────────────────────────────
echo "[5/7] Python environment..."
cd "$DEPLOY_DIR"

python3 --version
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q
if [ -f requirements.txt ]; then
    pip install -r requirements.txt -q
    echo "  ✅ Dependencies installed"
else
    echo "  ⚠ requirements.txt not found — install deps after uploading code"
fi

# ─────────────────────────────────────────────
# 6. SYSTEMD SERVICES
# ─────────────────────────────────────────────
echo "[6/7] Installing systemd services..."

# Position Monitor — persistent service (every 5 min via timer)
sudo tee /etc/systemd/system/weather-edge-monitor.service > /dev/null << EOF
[Unit]
Description=Weather Edge Position Monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=$DEPLOY_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV_DIR/bin/python3 $DEPLOY_DIR/position_monitor.py --once
StandardOutput=append:$LOG_DIR/position_monitor.log
StandardError=append:$LOG_DIR/position_monitor.log
TimeoutStartSec=120
EOF

sudo tee /etc/systemd/system/weather-edge-monitor.timer > /dev/null << EOF
[Unit]
Description=Weather Edge Position Monitor Timer (every 5 min)

[Timer]
OnCalendar=*:0/5
Persistent=true
RandomizedDelaySec=10

[Install]
WantedBy=timers.target
EOF

# Watchdog — persistent health check (every 15 min via timer)
sudo tee /etc/systemd/system/weather-edge-watchdog.service > /dev/null << EOF
[Unit]
Description=Weather Edge Watchdog
After=network-online.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=$DEPLOY_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV_DIR/bin/python3 $DEPLOY_DIR/watchdog.py
StandardOutput=append:$LOG_DIR/watchdog.log
StandardError=append:$LOG_DIR/watchdog.log
TimeoutStartSec=60
EOF

sudo tee /etc/systemd/system/weather-edge-watchdog.timer > /dev/null << EOF
[Unit]
Description=Weather Edge Watchdog Timer (every 15 min)

[Timer]
OnCalendar=*:0/15
Persistent=true
RandomizedDelaySec=30

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable weather-edge-monitor.timer
sudo systemctl enable weather-edge-watchdog.timer

# ─────────────────────────────────────────────
# 7. CRON JOBS (for scheduled scans)
# ─────────────────────────────────────────────
echo "[7/7] Installing cron jobs..."

# Write crontab (preserves any existing non-weather-edge entries)
EXISTING_CRON=$(crontab -l 2>/dev/null | grep -v "weather-edge" | grep -v "auto_trader" | grep -v "backtest_collector" | grep -v "morning_check" | grep -v "^#.*Weather Edge" || true)

(
echo "$EXISTING_CRON"
cat << EOF

# ═══════════════════════════════════════════════════
# WEATHER EDGE — Automated Trading & Monitoring
# All times are ET (server timezone set to America/New_York)
# ═══════════════════════════════════════════════════

# Auto Trader at optimal scan windows (all --dry-run until enabled)
0 6 * * *  $VENV_DIR/bin/python3 $DEPLOY_DIR/auto_trader.py --dry-run >> $LOG_DIR/auto_trader.log 2>&1
0 8 * * *  $VENV_DIR/bin/python3 $DEPLOY_DIR/auto_trader.py --dry-run >> $LOG_DIR/auto_trader.log 2>&1
0 10 * * * $VENV_DIR/bin/python3 $DEPLOY_DIR/auto_trader.py --dry-run >> $LOG_DIR/auto_trader.log 2>&1
0 15 * * * $VENV_DIR/bin/python3 $DEPLOY_DIR/auto_trader.py --dry-run >> $LOG_DIR/auto_trader.log 2>&1
0 23 * * * $VENV_DIR/bin/python3 $DEPLOY_DIR/auto_trader.py --dry-run >> $LOG_DIR/auto_trader.log 2>&1

# Backtest Collector — 8:30 AM (after settlement)
30 8 * * * $VENV_DIR/bin/python3 $DEPLOY_DIR/backtest_collector.py >> $LOG_DIR/backtest_collector.log 2>&1

# Morning Check — 6:30 AM (position evaluation)
30 6 * * * $VENV_DIR/bin/python3 $DEPLOY_DIR/morning_check.py >> $LOG_DIR/morning_check.log 2>&1
EOF
) | crontab -

echo "  ✅ Cron jobs installed (auto_trader in --dry-run mode)"

# ─────────────────────────────────────────────
# LOG ROTATION
# ─────────────────────────────────────────────
sudo tee /etc/logrotate.d/weather-edge > /dev/null << EOF
$LOG_DIR/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 644 ubuntu ubuntu
}
EOF

# ─────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo "═══════════════════════════════════════════"
echo ""

if [ -f "$DEPLOY_DIR/.env" ] && [ -f "$DEPLOY_DIR/kalshi_private_key.pem" ]; then
    echo "  ✅ .env file found"
    echo "  ✅ Kalshi private key found"

    # Quick smoke test
    source "$VENV_DIR/bin/activate"
    if python3 -c "from config import STATIONS; print(f'  ✅ Config loaded: {len(STATIONS)} cities')" 2>/dev/null; then
        echo ""
    else
        echo "  ⚠ Config import failed — check code upload"
    fi
else
    echo "  ⚠ Missing files — upload with deploy.sh:"
    [ ! -f "$DEPLOY_DIR/.env" ] && echo "    - .env"
    [ ! -f "$DEPLOY_DIR/kalshi_private_key.pem" ] && echo "    - kalshi_private_key.pem"
fi

echo ""
echo "  NEXT STEPS:"
echo "  ─────────────────────────────────────────"
echo "  1. Upload code:      ./deploy/deploy.sh <server-ip>"
echo "  2. Verify dry-run:   ssh ubuntu@<ip> '$VENV_DIR/bin/python3 $DEPLOY_DIR/auto_trader.py --dry-run'"
echo "  3. Start monitors:   ssh ubuntu@<ip> 'sudo systemctl start weather-edge-monitor.timer weather-edge-watchdog.timer'"
echo "  4. Watch logs:       ssh ubuntu@<ip> 'tail -f $LOG_DIR/*.log'"
echo "  5. Go live:          Edit cron, remove --dry-run flags"
echo ""
echo "  EMERGENCY STOP:"
echo "  ssh ubuntu@<ip> 'touch $DEPLOY_DIR/PAUSE_TRADING'"
echo ""
echo "  VIEW STATUS:"
echo "  systemctl list-timers --all | grep weather"
echo "  crontab -l"
echo ""
