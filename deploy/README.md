# Weather Edge — Oracle Cloud Free Tier Deployment

## Why Oracle Cloud?
- **$0/month forever** (not a trial — truly always-free)
- 4 ARM cores, 24 GB RAM, 200 GB storage
- Ashburn, VA region: 5-15ms to Kalshi (same coast)
- 99.9% uptime SLA

## What Gets Deployed

```
CRON (6/8/10/15/23 ET)
  auto_trader.py      → Scan + auto-execute 90+ confidence setups

SYSTEMD TIMERS
  position_monitor    → Every 5 min — exits, trailing stops, freeroll
  watchdog            → Every 15 min — cron health checks

CRON (one-shot daily)
  morning_check       → 6:30 AM — pre-settlement position evaluation
  backtest_collector  → 8:30 AM — settlement data collection
```

---

## Step-by-Step Setup

### 1. Create Oracle Cloud Account

1. Go to https://www.oracle.com/cloud/free/
2. Sign up (credit card required for identity verification — never charged for free tier)
3. Select **Home Region: US East (Ashburn)** — lowest latency to Kalshi

### 2. Create ARM Instance

In the Oracle Cloud Console:

```
Compute → Instances → Create Instance

Name:           weather-edge
Image:          Ubuntu 22.04 (or 24.04)
Shape:          VM.Standard.A1.Flex (Ampere ARM)
  OCPUs:        1 (free tier allows up to 4)
  Memory:       6 GB (free tier allows up to 24 GB)
Networking:     Create new VCN + public subnet
  Public IP:    Yes
Boot volume:    50 GB (free tier allows up to 200 GB)
SSH key:        Paste your public key (~/.ssh/id_rsa.pub)
```

> **Tip:** If you get "Out of host capacity," try a different Availability Domain
> or try again later (ARM instances are popular). Scripts exist to auto-retry.

### 3. Open SSH Port

```
Networking → Virtual Cloud Networks → [your VCN]
  → Security Lists → Default → Ingress Rules
  → Add: Source 0.0.0.0/0, TCP, Port 22
```

### 4. SSH In & Run Setup

```bash
# From your Mac
ssh ubuntu@<oracle-instance-ip>

# On the server — one-time setup
mkdir -p ~/limitless
```

### 5. Deploy Code

```bash
# From your Mac (project root)
chmod +x deploy/deploy.sh
./deploy/deploy.sh <oracle-instance-ip> --full
```

This will:
1. rsync all code (excluding secrets, .venv, positions.json)
2. Upload `.env` and `kalshi_private_key.pem`
3. Run `setup_oracle.sh` (installs Python, chrony, systemd services, cron)
4. Run tests on server
5. Smoke test the deployment

### 6. Verify

```bash
ssh ubuntu@<ip>

# Check cron
crontab -l

# Check timers
systemctl list-timers --all | grep weather

# Manual dry run
cd ~/limitless
source .venv/bin/activate
python3 auto_trader.py --dry-run

# Watch logs
tail -f /var/log/weather-edge/*.log
```

### 7. Go Live

When you're satisfied with dry-run results:

```bash
# Edit crontab to remove --dry-run flags
crontab -e
# Change: auto_trader.py --dry-run  →  auto_trader.py
```

---

## Day-to-Day Operations

### View logs
```bash
ssh ubuntu@<ip> 'tail -f /var/log/weather-edge/auto_trader.log'
ssh ubuntu@<ip> 'tail -f /var/log/weather-edge/position_monitor.log'
```

### Emergency stop
```bash
ssh ubuntu@<ip> 'touch ~/limitless/PAUSE_TRADING'     # HALT
ssh ubuntu@<ip> 'rm ~/limitless/PAUSE_TRADING'         # RESUME
```

### Push code updates
```bash
./deploy/deploy.sh <ip>           # Fast: code + deps + tests only
./deploy/deploy.sh <ip> --secrets  # Upload just .env + key
```

### Check positions
```bash
ssh ubuntu@<ip> 'cat ~/limitless/positions.json | python3 -m json.tool'
```

### Run a manual scan
```bash
ssh ubuntu@<ip> 'cd ~/limitless && source .venv/bin/activate && python3 auto_trader.py --dry-run --city NYC'
```

### Service status
```bash
ssh ubuntu@<ip> 'systemctl list-timers --all | grep weather'
ssh ubuntu@<ip> 'crontab -l'
```

---

## Architecture on Server

```
/home/ubuntu/limitless/
├── .env                    # Secrets (never rsync'd)
├── .venv/                  # Python venv
├── kalshi_private_key.pem  # RSA key (chmod 600)
├── positions.json          # Live positions (auto-created)
├── config.py               # All settings
├── auto_trader.py          # Main scan+trade loop (cron)
├── position_monitor.py     # Exit rules (systemd timer)
├── watchdog.py             # Health checks (systemd timer)
├── morning_check.py        # Pre-settlement (cron)
├── backtest_collector.py   # Data collection (cron)
├── edge_scanner_v2.py      # KDE + ensemble scanner
├── kalshi_client.py        # API client
├── execute_trade.py        # Order execution
├── position_store.py       # Atomic file store
├── trading_guards.py       # Safety checks
├── notifications.py        # Discord alerts
└── deploy/
    ├── setup_oracle.sh     # One-time server setup
    ├── deploy.sh           # Push code updates
    └── .env.example        # Template

/var/log/weather-edge/
├── auto_trader.log
├── position_monitor.log
├── watchdog.log
├── backtest_collector.log
└── morning_check.log       # 14-day rotation
```

---

## Cost Comparison

| Provider | Spec | Latency to Kalshi | Cost |
|----------|------|-------------------|------|
| **Oracle Cloud (Ashburn)** | 1 ARM core, 6 GB | 5-15ms | **$0/mo forever** |
| AWS EC2 t3.micro | 2 vCPU, 1 GB | <5ms | $0 for 12mo, then $8/mo |
| DigitalOcean (NYC) | 1 vCPU, 1 GB | 5-10ms | $6/mo |
| Vultr (NJ) | 1 vCPU, 1 GB | 5-10ms | $5/mo |

Oracle Ashburn is the best option: zero cost, sufficient specs, acceptable latency.
Latency doesn't matter much for this system — we place limit orders at scan time
(5x daily), not HFT. 10ms vs 5ms is irrelevant for a cron-based scanner.

---

## Troubleshooting

### "Out of host capacity" when creating instance
ARM instances are popular. Solutions:
- Try a different Availability Domain (AD-1, AD-2, AD-3)
- Try during off-peak hours (early morning US time)
- Use an auto-retry script: search "oracle cloud instance creation script"

### Tests fail on server
```bash
ssh ubuntu@<ip> 'cd ~/limitless && source .venv/bin/activate && python3 -m pytest tests/ -v'
```
Common issues: missing numpy on ARM (install via apt: `sudo apt install python3-numpy`)

### Discord alerts not working
```bash
ssh ubuntu@<ip> 'grep DISCORD ~/limitless/.env'
# Verify webhook URL is set
```

### Cron jobs not running
```bash
ssh ubuntu@<ip> 'grep CRON /var/log/syslog | tail -20'
# Check timezone
ssh ubuntu@<ip> 'timedatectl'
```

### Position monitor not running
```bash
ssh ubuntu@<ip> 'systemctl status weather-edge-monitor.timer'
ssh ubuntu@<ip> 'systemctl status weather-edge-monitor.service'
ssh ubuntu@<ip> 'journalctl -u weather-edge-monitor -n 20'
```
