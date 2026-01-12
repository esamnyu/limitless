# Project: Kalshi Weather Trading Bot

## 1. Executive Summary

**Objective:** Automated trading bot for Kalshi's NYC high temperature markets (KXHIGHNY) that exploits mispricings between professional weather model forecasts and market prices.

**Core Thesis:** Professional weather models (ECMWF, GFS) are more accurate than retail trader guesses. When models agree but the market prices a different bracket, we have a high-probability edge.

**Target Win Rate:** 75-85%
**Primary Market:** KXHIGHNY (NYC Daily High Temperature)
**Legal Status:** CFTC-regulated, legal in all US states including NYC

## 2. The Strategy

### 2.1 The Edge
Weather prediction markets are inefficient because:
1. Retail traders often follow outdated NWS forecasts
2. Professional models (ECMWF, GFS) update 4x daily with higher accuracy
3. When both models agree, confidence reaches ~90%
4. Markets often misprice by 20-70 percentage points

### 2.2 Model Accuracy
| Model | Provider | Accuracy (1-day) | Update Frequency |
|-------|----------|------------------|------------------|
| ECMWF | European Centre | ~85% | 4x daily |
| GFS | NOAA | ~80% | 4x daily |
| Combined (when agree) | - | ~90% | - |

### 2.3 Trading Logic
```
1. Fetch GFS and ECMWF forecasts for NYC
2. Check if models agree (within 2°F)
3. Determine which bracket the average temp falls into
4. Get Kalshi market prices for that bracket
5. Calculate edge = model_confidence - market_price
6. If edge > 20% and models disagree with market favorite → TRADE
```

## 3. Market Structure

### 3.1 KXHIGHNY (NYC High Temperature)
- **Location:** Central Park, NYC (40.7829, -73.9654)
- **Settlement:** NWS CLI report next morning
- **Brackets:** 6 per day (dynamic based on forecast range)
- **Market Opens:** ~10 AM day before
- **Market Closes:** Evening before settlement

### 3.2 Bracket Examples
```
Typical winter day brackets:
├── T39: <39°F (extreme cold)
├── B39.5: 39-40°F
├── B41.5: 41-42°F
├── B43.5: 43-44°F
├── B45.5: 45-46°F
└── T46: >46°F (warm)

Note: Brackets are DYNAMIC - they change daily based on expected temp range
```

### 3.3 Ticker Format
`KXHIGHNY-{YY}{MMM}{DD}-{BRACKET}`

Examples:
- `KXHIGHNY-26JAN12-B39.5` → Jan 12, 2026, 39-40°F bracket
- `KXHIGHNY-26JAN12-T46` → Jan 12, 2026, >46°F bracket

## 4. Technical Architecture

### 4.1 Data Sources
| Source | API Endpoint | Cost | Purpose |
|--------|--------------|------|---------|
| Open-Meteo GFS | `api.open-meteo.com/v1/forecast` | Free | GFS forecasts |
| Open-Meteo ECMWF | `api.open-meteo.com/v1/ecmwf` | Free | ECMWF forecasts |
| Kalshi API | `trading-api.kalshi.com` | Free | Market data & trading |

### 4.2 File Structure
```
limitless/
├── weather_bot.py      # Main trading bot
├── weather_client.py   # Open-Meteo API wrapper
├── kalshi_client.py    # Kalshi API wrapper
├── alerts.py           # Discord notifications
├── .env                # API credentials
└── weather_paper_trades.jsonl  # Paper trade log
```

### 4.3 Authentication
Kalshi uses RSA-PSS signatures:
- Private key: `kalshi_private_key.pem`
- API Key ID: In `.env` as `KALSHI_API_KEY_ID`

## 5. Entry Criteria

All conditions must be true:
```
├── GFS and ECMWF agree within 2°F
├── Model bracket ≠ Market favorite bracket
├── Edge > 20% (model_confidence - market_price)
├── Time to settlement > 6 hours
└── Market has liquidity
```

## 6. Position Sizing & Risk

```
├── Max $25 per trade
├── Max 1 position per day per city
├── Stop trading if 3 consecutive losses
├── Scale up after 10 winning trades
└── Never risk >5% of bankroll per trade
```

## 7. Expected Performance

```
Per Trade:
├── Win Rate: 75-85%
├── Avg Win: ~10x (buy at 10c, win $1)
├── Avg Loss: -$25 max
├── Expected Value: +$0.60-0.70 per contract

Monthly (20 trades @ $25):
├── Wins: ~16 trades × $22.50 = +$360
├── Losses: ~4 trades × $25 = -$100
├── Net: ~+$260/month
├── ROI: ~50% monthly
```

## 8. Developer Instructions

### 8.1 Running the Bot
```bash
# Paper trading (default)
python3 weather_bot.py

# Single scan
python3 weather_bot.py --once

# Live trading (requires funded account)
python3 weather_bot.py --live
```

### 8.2 Key Functions

**weather_client.py:**
- `get_gfs_forecast()` - Fetch GFS model data
- `get_ecmwf_forecast()` - Fetch ECMWF model data
- `get_model_consensus()` - Check if models agree

**weather_bot.py:**
- `scan_once()` - Single scan for opportunities
- `_check_opportunity()` - Evaluate a specific market
- `_paper_trade()` - Record paper trade
- `_execute_trade()` - Place live order

**kalshi_client.py:**
- `get_markets(series_ticker="KXHIGHNY")` - Get weather markets
- `place_order()` - Execute trade
- `get_balance()` - Check account balance

### 8.3 Code Style
- Async Python (asyncio/aiohttp)
- Type hints where helpful
- Modular design (separate clients)
- Error handling with graceful degradation

### 8.4 Testing Changes
```bash
# Test weather API
python3 weather_client.py

# Test single scan
python3 weather_bot.py --once

# Check paper trades
cat weather_paper_trades.jsonl
```

## 9. Improvement Ideas

### High Priority
- [ ] Add HRRR model for same-day trades (more frequent updates)
- [ ] Track actual settlement temps for win rate validation
- [ ] Add boundary detection (avoid temps near bracket edges)

### Medium Priority
- [ ] Expand to other cities (Chicago, LA, Miami)
- [ ] Add precipitation markets
- [ ] Historical backtest with NWS data

### Low Priority
- [ ] Web dashboard for monitoring
- [ ] SMS alerts for opportunities
- [ ] Multi-account support

## 10. Important Constraints

1. **Legal:** Only trade on Kalshi (CFTC-regulated). Avoid Polymarket (blocked for US).

2. **Timing:** Don't trade within 6 hours of settlement (models less reliable).

3. **Liquidity:** Skip markets with no bids (can't execute).

4. **Edge Threshold:** Minimum 20% edge required. Don't chase small edges.

5. **Model Agreement:** Only trade when GFS and ECMWF agree within 2°F.

## 11. Quick Reference

### API Endpoints
```
GFS:   https://api.open-meteo.com/v1/forecast?latitude=40.7829&longitude=-73.9654&daily=temperature_2m_max&temperature_unit=fahrenheit
ECMWF: https://api.open-meteo.com/v1/ecmwf?latitude=40.7829&longitude=-73.9654&daily=temperature_2m_max&temperature_unit=fahrenheit
```

### Kalshi Series
- `KXHIGHNY` - NYC High Temperature
- `KXHIGHCHI` - Chicago High Temperature (future)
- `KXHIGHLA` - LA High Temperature (future)

### Settlement Source
NWS CLI Report: https://forecast.weather.gov/product.php?site=OKX&product=CLI&issuedby=NYC
