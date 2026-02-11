# POST-MORTEM: NYC Sniper Trade - Jan 19-20, 2026

**Result:** LOSS
**P&L:** -$113.44 (-41%)
**Saved by salvage:** $163

---

## THE TRADE THESIS

We bet on B24-26°F based on:
- NWS forecast: 25°F high at midnight (classic "Midnight High" pattern)
- Temperature dropping steadily: 30°F → 28°F → 27°F
- Physics: Strong W winds (gusts 29 mph) should accelerate cooling

## WHAT ACTUALLY HAPPENED

```
Temperature stopped dropping at 26.1°F
We needed 25.9°F or below
Missed by 0.2°F
```

---

## MISTAKES MADE

### 1. Chased the Price

```
Entry 1:  27¢ (initial target)
Entry 2:  32¢ (repriced to fill)
Entry 3:  36¢ (added more)
Entry 4:  44-47¢ (FOMO, market moving)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average cost: 43¢ (started at 27¢!)
```

**Lesson:** Once you miss your entry, don't chase. The edge erodes with every cent higher.

### 2. Over-Concentrated Position

- Put $269 into B24-26 (97% of capital)
- Left only $16 cash
- No ability to hedge or adjust

**Lesson:** Keep 30-40% cash reserve for adjustments.

### 3. Trusted the Forecast Too Much

- NWS said 25°F at midnight
- Actual: 26.1°F
- Forecast error: +1.1°F

**Lesson:** NWS hourly forecasts have ±1-2°F error. Don't bet on boundary cases.

### 4. Ignored the Boundary Risk

The bracket boundary at 26°F was critical:
- 25.9°F → rounds to B24-26 (WIN)
- 26.0°F → rounds to B26-28 (LOSE)

We were betting on a coin flip at the boundary.

**Lesson:** Avoid trades where success requires landing on the exact boundary. Look for 2-3°F margin of safety.

### 5. Cooling Rate Extrapolation Error

```
9:51 PM:  28.9°F
10:51 PM: 27.0°F  (-1.9°F/hr)
11:51 PM: 26.1°F  (-0.9°F/hr) ← Rate HALVED
```

We assumed linear cooling. It was logarithmic - slows as it approaches equilibrium.

**Lesson:** Cooling rate decreases as temp approaches the air mass temperature.

### 6. Late Recognition of Failure

- At 11:51 PM, temp was 26.1°F (already in losing bracket)
- Market was still bidding 25-28¢
- Should have sold immediately at first sign of stall
- Instead waited until bid dropped to 13¢

**Lesson:** Set a stop-loss trigger. If temp isn't tracking, exit early.

---

## WHAT WE DID RIGHT

1. **Salvaged the position** - Sold at avg 26¢ instead of holding to $0
2. **Saved $163** vs total loss of $276
3. **Recognized the physics** - The midnight high thesis was correct, just missed by 0.2°F
4. **Had a hedge** - B22-24 at $7 was the right instinct (wrong bracket though)

---

## RULES FOR NEXT TIME

| Rule | Description |
|------|-------------|
| **Entry Discipline** | If you miss your price by >5¢, PASS |
| **Position Sizing** | Max 60% of capital in one bracket |
| **Margin of Safety** | Need 2°F+ buffer from bracket boundary |
| **Stop Loss** | Exit if temp stalls 0.5°F above target for 30+ min |
| **Forecast Skepticism** | Treat NWS as ±2°F, not gospel |

---

## TRADE LOG

### Entries (B24-26)
| Time | Action | Qty | Price | Cost |
|------|--------|-----|-------|------|
| ~10:30 PM | BUY | 52 | 30.9¢ | $16.07 |
| ~10:45 PM | BUY | 100 | 36¢ | $36.00 |
| ~11:00 PM | BUY | 185 | 46-47¢ | $85.55 |
| ~11:15 PM | BUY | 141 | 47¢ | $66.27 |
| ~11:30 PM | BUY | 147 | 47¢ | $69.09 |
| **Total** | | **625** | **43.05¢ avg** | **$269.09** |

### Hedge (B22-24)
| Time | Action | Qty | Price | Cost |
|------|--------|-----|-------|------|
| ~10:40 PM | BUY | 700 | 1¢ | $7.00 |

### Exit (B24-26)
| Time | Action | Qty | Price | Proceeds |
|------|--------|-----|-------|----------|
| 12:02 AM | SELL | 89 | 31¢ | $27.59 |
| 12:02 AM | SELL | 536 | 13-26¢ | $135.06 |
| **Total** | | **625** | **26¢ avg** | **$162.65** |

---

## FINAL SCORE

```
Investment:  $276.09
Recovered:   $162.65
Net Loss:    -$113.44 (-41%)

But: Could have been -$276 (-100%)
Salvage saved: $163
```

---

## KEY TAKEAWAY

The physics was right. The execution was wrong. We chased, over-concentrated, and bet on a boundary. Expensive lesson, but we kept $171 to trade another day.

**Next time:** Respect the entry price. Respect position limits. Respect the boundary.
