#!/bin/bash
# =============================================================================
# WEATHER EDGE — Push code to remote server via rsync
#
# Usage:
#   ./deploy/deploy.sh <server-ip>              # Code only (fast)
#   ./deploy/deploy.sh <server-ip> --full       # Code + secrets + setup
#   ./deploy/deploy.sh <server-ip> --secrets     # Upload .env + key only
#
# Requirements:
#   - SSH key-based auth configured for ubuntu@<server-ip>
#   - Server already provisioned with setup_oracle.sh (or setup manually)
# =============================================================================
set -euo pipefail

# ─── Config ─────────────────────────────────
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # Project root
REMOTE_USER="ubuntu"
REMOTE_DIR="/home/ubuntu/limitless"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"  # Override with SSH_KEY env var

# ─── Args ───────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: $0 <server-ip> [--full|--secrets]"
    echo ""
    echo "  <server-ip>    SSH to this host"
    echo "  --full         Upload code + secrets + run setup"
    echo "  --secrets      Upload .env + private key only"
    echo ""
    echo "  Set SSH_KEY env var to use a custom SSH key"
    exit 1
fi

SERVER="$1"
MODE="${2:-code}"

SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
SCP_CMD="scp -i $SSH_KEY -o StrictHostKeyChecking=no"
RSYNC_SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"

echo "═══════════════════════════════════════════"
echo "  Weather Edge — Deploy to $SERVER"
echo "  Mode: $MODE"
echo "═══════════════════════════════════════════"

# ─── Upload secrets ─────────────────────────
upload_secrets() {
    echo ""
    echo "[secrets] Uploading .env and private key..."
    $SSH_CMD "$REMOTE_USER@$SERVER" "mkdir -p $REMOTE_DIR"

    if [ -f "$LOCAL_DIR/.env" ]; then
        $SCP_CMD "$LOCAL_DIR/.env" "$REMOTE_USER@$SERVER:$REMOTE_DIR/.env"
        echo "  ✅ .env uploaded"
    else
        echo "  ⚠ .env not found at $LOCAL_DIR/.env"
    fi

    if [ -f "$LOCAL_DIR/kalshi_private_key.pem" ]; then
        $SCP_CMD "$LOCAL_DIR/kalshi_private_key.pem" "$REMOTE_USER@$SERVER:$REMOTE_DIR/kalshi_private_key.pem"
        $SSH_CMD "$REMOTE_USER@$SERVER" "chmod 600 $REMOTE_DIR/kalshi_private_key.pem"
        echo "  ✅ Private key uploaded (chmod 600)"
    else
        echo "  ⚠ kalshi_private_key.pem not found"
    fi
}

# ─── Upload code ────────────────────────────
upload_code() {
    echo ""
    echo "[code] Syncing code to $SERVER..."

    rsync -avz --progress \
        --exclude '.git' \
        --exclude '.venv' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.env' \
        --exclude 'kalshi_private_key.pem' \
        --exclude 'positions.json' \
        --exclude '.positions.lock' \
        --exclude 'alerts_fallback.jsonl' \
        --exclude 'PAUSE_TRADING' \
        --exclude 'heartbeats/' \
        --exclude 'backtest/*.json' \
        --exclude 'logs/' \
        --exclude '.pytest_cache' \
        -e "$RSYNC_SSH" \
        "$LOCAL_DIR/" "$REMOTE_USER@$SERVER:$REMOTE_DIR/"

    echo "  ✅ Code synced"
}

# ─── Remote setup ───────────────────────────
run_remote_setup() {
    echo ""
    echo "[setup] Running remote setup..."
    $SSH_CMD "$REMOTE_USER@$SERVER" "cd $REMOTE_DIR && chmod +x deploy/setup_oracle.sh && bash deploy/setup_oracle.sh"
}

# ─── Install deps ───────────────────────────
install_deps() {
    echo ""
    echo "[deps] Installing Python dependencies on server..."
    $SSH_CMD "$REMOTE_USER@$SERVER" "
        cd $REMOTE_DIR
        if [ ! -d .venv ]; then
            python3 -m venv .venv
        fi
        source .venv/bin/activate
        pip install --upgrade pip -q
        pip install -r requirements.txt -q
    "
    echo "  ✅ Dependencies installed"
}

# ─── Run tests ──────────────────────────────
run_remote_tests() {
    echo ""
    echo "[test] Running tests on server..."
    $SSH_CMD "$REMOTE_USER@$SERVER" "
        cd $REMOTE_DIR
        source .venv/bin/activate
        python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20
    "
}

# ─── Verify ─────────────────────────────────
verify_deployment() {
    echo ""
    echo "[verify] Quick smoke test..."
    $SSH_CMD "$REMOTE_USER@$SERVER" "
        cd $REMOTE_DIR
        source .venv/bin/activate
        python3 -c \"
from config import STATIONS
print(f'  ✅ Config: {len(STATIONS)} cities')
from kalshi_client import KalshiClient
ts = KalshiClient._monotonic_ts_ms()
print(f'  ✅ Kalshi client: monotonic ts={ts}')
from trading_guards import check_kill_switch
ok, reason = check_kill_switch()
print(f'  ✅ Kill switch: {reason}')
print('  ✅ All imports clean')
\"
    "
}

# ─── Execute mode ───────────────────────────
case "$MODE" in
    --secrets)
        upload_secrets
        ;;
    --full)
        upload_code
        upload_secrets
        run_remote_setup
        run_remote_tests
        verify_deployment
        ;;
    *)
        upload_code
        install_deps
        run_remote_tests
        verify_deployment
        ;;
esac

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Deploy complete"
echo "═══════════════════════════════════════════"
echo ""
echo "  Quick commands:"
echo "  ssh ubuntu@$SERVER 'tail -f /var/log/weather-edge/*.log'"
echo "  ssh ubuntu@$SERVER '$REMOTE_DIR/.venv/bin/python3 $REMOTE_DIR/auto_trader.py --dry-run'"
echo "  ssh ubuntu@$SERVER 'touch $REMOTE_DIR/PAUSE_TRADING'  # Emergency stop"
echo ""
