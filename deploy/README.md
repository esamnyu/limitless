# NYC Weather Arb Bot - HFT Deployment Guide

## TL;DR - Best VPS for Latency Arbitrage

| Provider | Region | Latency to Kalshi | Cost | Verdict |
|----------|--------|-------------------|------|---------|
| **AWS Free Tier** | us-east-1 (N. Virginia) | <5ms | Free 12mo | **BEST** |
| Oracle Cloud | Ashburn, VA | 5-15ms | Free forever | Good backup |
| DigitalOcean | NYC1/NJ | 5-10ms | $4/mo | Alternative |
| Oracle Cloud | Other regions | 20-50ms | Free | Avoid |

**Why AWS wins:** Kalshi runs on AWS us-east-1. Same datacenter = <5ms latency.

---

## Quick Deploy (AWS Free Tier)

### 1. Create AWS Account
- Go to https://aws.amazon.com/free
- Create account (requires credit card, won't be charged)

### 2. Launch EC2 Instance
```
Region:        us-east-1 (N. Virginia) ← CRITICAL
Instance Type: t2.micro (free tier)
AMI:           Ubuntu 22.04 LTS
Storage:       8GB (default)
Security:      Allow SSH (port 22)
```

### 3. Connect & Deploy
```bash
# SSH into instance
ssh -i ~/your-key.pem ubuntu@<your-ec2-ip>

# Clone repo
git clone <your-repo-url> limitless
cd limitless

# Upload secrets (from local machine)
scp -i ~/your-key.pem .env ubuntu@<ip>:~/limitless/
scp -i ~/your-key.pem kalshi_private_key.pem ubuntu@<ip>:~/limitless/

# Run deployment script
chmod +x deploy/setup.sh
./deploy/setup.sh
```

### 4. Verify Latency
```bash
source venv/bin/activate
python3 deploy/latency_test.py
```

Expected on AWS us-east-1:
```
Kalshi API: ~3-5ms   ← OPTIMAL
NWS XML:    ~20-40ms ← Good
```

### 5. Start Bot
```bash
# Paper trading (default)
sudo systemctl start weather-arb

# Watch logs
journalctl -u weather-arb -f
```

### 6. Enable Live Trading
```bash
sudo nano /etc/systemd/system/weather-arb.service
# Change: nyc_weather_arb.py --live

sudo systemctl daemon-reload
sudo systemctl restart weather-arb
```

---

## Latency Matters: The Math

```
Scenario A: Your bot on AWS us-east-1
├── NWS observation received
├── Kalshi order sent: +3ms
├── Order filled: +2ms
└── Total: 5ms

Scenario B: Your bot on Oracle (Phoenix, AZ)
├── NWS observation received
├── Kalshi order sent: +45ms (cross-country + cloud hop)
├── Order filled: +2ms
└── Total: 47ms

Difference: 42ms

In 42ms, another bot on AWS can:
- See the same observation
- Place their order
- Get filled BEFORE you
```

---

## Monitoring Commands

```bash
# Service status
sudo systemctl status weather-arb

# Live logs
journalctl -u weather-arb -f

# Today's trades
cat nyc_arb_trades.jsonl | tail -20

# Check max temp tracking
cat nyc_daily_max_temp.json

# Time sync status
chronyc tracking

# Restart bot
sudo systemctl restart weather-arb
```

---

## Troubleshooting

### Bot crashes on startup
```bash
# Check logs for error
journalctl -u weather-arb -n 50

# Common fixes:
# 1. Missing .env file
# 2. Wrong path to private key
# 3. Python package missing
```

### High latency to Kalshi
```bash
# Run latency test
python3 deploy/latency_test.py

# If >50ms, you're in wrong region
# Redeploy to us-east-1
```

### Time sync issues
```bash
sudo chronyc makestep
chronyc tracking
```

---

## Cost After Free Tier Expires

| Option | Monthly Cost |
|--------|--------------|
| AWS t3.micro (on-demand) | ~$8/mo |
| AWS t3.micro (1yr reserved) | ~$4/mo |
| Oracle Cloud (always free) | $0 |
| Vultr NYC | $5/mo |
| DigitalOcean NYC | $4/mo |

**Recommendation:** Use AWS free tier for 12 months, then switch to Oracle Ashburn or a cheap NYC VPS.
