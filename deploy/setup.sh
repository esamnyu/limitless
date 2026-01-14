#!/bin/bash
# =============================================================================
# NYC Weather Arb Bot - HFT Production Deployment Script
# Optimized for: AWS Free Tier (us-east-1) or Oracle Cloud (Ashburn)
# =============================================================================

set -e

echo "=========================================="
echo "  NYC Weather Arb Bot - HFT Deployment"
echo "=========================================="

# -----------------------------------------------------------------------------
# 1. SYSTEM UPDATE & HFT ESSENTIALS
# -----------------------------------------------------------------------------
echo "[1/6] Installing system dependencies..."
sudo apt update
sudo apt install -y python3-pip python3-venv git chrony curl

# -----------------------------------------------------------------------------
# 2. TIME SYNCHRONIZATION (CRITICAL FOR HFT)
# -----------------------------------------------------------------------------
echo "[2/6] Configuring time sync (chrony)..."
sudo systemctl enable chrony
sudo systemctl start chrony

# Force immediate sync
sudo chronyc makestep 2>/dev/null || true

# Verify time sync
echo "Time sync status:"
chronyc tracking | grep -E "Ref time|System time|Last offset"

# -----------------------------------------------------------------------------
# 3. NETWORK TUNING FOR LOW LATENCY
# -----------------------------------------------------------------------------
echo "[3/6] Applying network optimizations..."

# Reduce TCP keepalive (detect dead connections faster)
sudo sysctl -w net.ipv4.tcp_keepalive_time=60
sudo sysctl -w net.ipv4.tcp_keepalive_intvl=10
sudo sysctl -w net.ipv4.tcp_keepalive_probes=6

# Enable TCP Fast Open (reduces handshake latency)
sudo sysctl -w net.ipv4.tcp_fastopen=3

# Increase socket buffer sizes
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216

# Make persistent
cat << 'EOF' | sudo tee -a /etc/sysctl.conf
# HFT Network Tuning
net.ipv4.tcp_keepalive_time=60
net.ipv4.tcp_keepalive_intvl=10
net.ipv4.tcp_keepalive_probes=6
net.ipv4.tcp_fastopen=3
net.core.rmem_max=16777216
net.core.wmem_max=16777216
EOF

# -----------------------------------------------------------------------------
# 4. PYTHON ENVIRONMENT
# -----------------------------------------------------------------------------
echo "[4/6] Setting up Python environment..."

cd /home/ubuntu
if [ ! -d "limitless" ]; then
    echo "Please clone your repo first: git clone <your-repo> limitless"
    echo "Or upload files via scp"
    exit 1
fi

cd limitless
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# -----------------------------------------------------------------------------
# 5. VERIFY CONFIGURATION
# -----------------------------------------------------------------------------
echo "[5/6] Verifying configuration..."

# Check .env exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Create it with: KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH"
    exit 1
fi

# Check private key exists
if [ ! -f "kalshi_private_key.pem" ]; then
    echo "ERROR: kalshi_private_key.pem not found!"
    exit 1
fi

# Test the bot
echo "Testing bot configuration..."
python3 nyc_weather_arb.py --test

# -----------------------------------------------------------------------------
# 6. INSTALL SYSTEMD SERVICE
# -----------------------------------------------------------------------------
echo "[6/6] Installing systemd service..."

cat << 'EOF' | sudo tee /etc/systemd/system/weather-arb.service
[Unit]
Description=NYC Weather Arb Bot (HFT)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/limitless

# CRITICAL: Unbuffered output for instant logs
Environment=PYTHONUNBUFFERED=1

# Path to python in venv
ExecStart=/home/ubuntu/limitless/venv/bin/python3 nyc_weather_arb.py

# Aggressive restart on crash
Restart=always
RestartSec=3

# Resource limits
LimitNOFILE=65535

# Nice priority (lower = higher priority, -20 to 19)
Nice=-10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable weather-arb

echo ""
echo "=========================================="
echo "  DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start weather-arb"
echo "  Stop:    sudo systemctl stop weather-arb"
echo "  Status:  sudo systemctl status weather-arb"
echo "  Logs:    journalctl -u weather-arb -f"
echo ""
echo "To enable LIVE trading, edit the service:"
echo "  sudo nano /etc/systemd/system/weather-arb.service"
echo "  Change: nyc_weather_arb.py --live"
echo "  Then:   sudo systemctl daemon-reload && sudo systemctl restart weather-arb"
echo ""
