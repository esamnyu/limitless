#!/usr/bin/env python3
"""
Project Atlas - Lag Analysis
Analyzes collected price data to determine if exploitable lag exists.

Usage:
    python3 analyze_lag.py lag_data_YYYYMMDD_HHMMSS.csv
"""

import csv
import sys
from datetime import datetime
from collections import defaultdict


def load_data(filename: str) -> list[dict]:
    """Load CSV data."""
    data = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "timestamp": row["Timestamp"],
                "kraken_price": float(row["Kraken_Price"]),
                "limitless_yes": float(row["Limitless_YES"]),
                "limitless_no": float(row["Limitless_NO"]),
                "price_vs_strike": float(row["Price_vs_Strike"]),
                "implied": row["Implied_Direction"],
                "limitless_age_ms": float(row["Limitless_Age_Ms"]),
            })
    return data


def analyze_lag(data: list[dict]) -> dict:
    """Analyze the lag patterns in the data."""
    results = {
        "total_samples": len(data),
        "duration_minutes": 0,
        "price_range": {"min": 0, "max": 0, "volatility": 0},
        "lag_analysis": {},
        "mispricings": [],
        "verdict": "",
    }

    if len(data) < 10:
        results["verdict"] = "INSUFFICIENT DATA - Need more samples"
        return results

    # Duration
    start = datetime.strptime(data[0]["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
    end = datetime.strptime(data[-1]["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
    results["duration_minutes"] = (end - start).total_seconds() / 60

    # Price range
    prices = [d["kraken_price"] for d in data]
    results["price_range"]["min"] = min(prices)
    results["price_range"]["max"] = max(prices)
    results["price_range"]["volatility"] = max(prices) - min(prices)

    # Analyze mispricings
    # A mispricing occurs when:
    # - Kraken says price >> strike (should be YES ~0.95+) but Limitless YES is low
    # - Kraken says price << strike (should be NO ~0.95+) but Limitless NO is low
    mispricings = []

    for i, d in enumerate(data):
        price_diff = d["price_vs_strike"]

        # Calculate what probability "should" be based on Kraken price
        if price_diff > 1000:
            expected_yes = 0.98
        elif price_diff > 500:
            expected_yes = 0.90
        elif price_diff > 200:
            expected_yes = 0.75
        elif price_diff > 0:
            expected_yes = 0.55
        elif price_diff > -200:
            expected_yes = 0.45
        elif price_diff > -500:
            expected_yes = 0.25
        elif price_diff > -1000:
            expected_yes = 0.10
        else:
            expected_yes = 0.02

        actual_yes = d["limitless_yes"]
        discrepancy = expected_yes - actual_yes

        # Significant mispricing: >10% difference
        if abs(discrepancy) > 0.10:
            mispricings.append({
                "timestamp": d["timestamp"],
                "kraken_price": d["kraken_price"],
                "price_vs_strike": price_diff,
                "expected_yes": expected_yes,
                "actual_yes": actual_yes,
                "discrepancy": discrepancy,
                "direction": "BUY_YES" if discrepancy > 0 else "BUY_NO",
                "limitless_age_ms": d["limitless_age_ms"],
            })

    results["mispricings"] = mispricings

    # Calculate statistics
    if mispricings:
        discrepancies = [abs(m["discrepancy"]) for m in mispricings]
        results["lag_analysis"] = {
            "mispricing_count": len(mispricings),
            "mispricing_rate": len(mispricings) / len(data) * 100,
            "avg_discrepancy": sum(discrepancies) / len(discrepancies),
            "max_discrepancy": max(discrepancies),
            "avg_limitless_age_ms": sum(m["limitless_age_ms"] for m in mispricings) / len(mispricings),
        }
    else:
        results["lag_analysis"] = {
            "mispricing_count": 0,
            "mispricing_rate": 0,
            "avg_discrepancy": 0,
            "max_discrepancy": 0,
            "avg_limitless_age_ms": 0,
        }

    # Verdict
    mispricing_rate = results["lag_analysis"]["mispricing_rate"]
    avg_discrepancy = results["lag_analysis"]["avg_discrepancy"]

    if mispricing_rate > 20 and avg_discrepancy > 0.15:
        results["verdict"] = "STRONG OPPORTUNITY - Significant lag detected!"
    elif mispricing_rate > 10 and avg_discrepancy > 0.10:
        results["verdict"] = "MODERATE OPPORTUNITY - Some exploitable lag exists"
    elif mispricing_rate > 5:
        results["verdict"] = "WEAK OPPORTUNITY - Occasional mispricings, marginal edge"
    else:
        results["verdict"] = "NO OPPORTUNITY - Markets are well-synced"

    return results


def print_report(results: dict, show_mispricings: bool = True):
    """Print analysis report."""
    print("\n" + "=" * 70)
    print("               PROJECT ATLAS - LAG ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n[DATA SUMMARY]")
    print(f"  Total samples:     {results['total_samples']}")
    print(f"  Duration:          {results['duration_minutes']:.1f} minutes")
    print(f"  Price range:       ${results['price_range']['min']:,.2f} - ${results['price_range']['max']:,.2f}")
    print(f"  Volatility:        ${results['price_range']['volatility']:,.2f}")

    lag = results["lag_analysis"]
    print(f"\n[LAG ANALYSIS]")
    print(f"  Mispricings found: {lag.get('mispricing_count', 0)}")
    print(f"  Mispricing rate:   {lag.get('mispricing_rate', 0):.1f}%")
    print(f"  Avg discrepancy:   {lag.get('avg_discrepancy', 0):.1%}")
    print(f"  Max discrepancy:   {lag.get('max_discrepancy', 0):.1%}")
    print(f"  Avg data age:      {lag.get('avg_limitless_age_ms', 0):.0f}ms")

    # Show sample mispricings
    if show_mispricings and results["mispricings"]:
        print(f"\n[SAMPLE MISPRICINGS] (showing first 10)")
        print("-" * 70)
        for m in results["mispricings"][:10]:
            print(f"  {m['timestamp']}")
            print(f"    Kraken: ${m['kraken_price']:,.2f} ({m['price_vs_strike']:+.0f} vs strike)")
            print(f"    Expected YES: {m['expected_yes']:.0%} | Actual: {m['actual_yes']:.0%}")
            print(f"    Discrepancy: {m['discrepancy']:+.1%} -> {m['direction']}")
            print()

    print("\n" + "=" * 70)
    print(f"  VERDICT: {results['verdict']}")
    print("=" * 70)

    # Recommendations
    print("\n[RECOMMENDATIONS]")
    if "STRONG" in results["verdict"]:
        print("  1. The lag is real and exploitable")
        print("  2. Proceed to Phase 2 (small live trades)")
        print("  3. Focus on high-volatility periods")
    elif "MODERATE" in results["verdict"]:
        print("  1. Lag exists but may be marginal after fees")
        print("  2. Collect more data during volatile periods")
        print("  3. Consider if gas + fees eat the edge")
    elif "WEAK" in results["verdict"]:
        print("  1. Edge is thin - probably not worth the risk")
        print("  2. Try during extreme volatility (news events)")
        print("  3. Consider other prediction markets")
    else:
        print("  1. No exploitable lag detected")
        print("  2. Markets are efficiently priced")
        print("  3. Strategy may not be viable on Limitless")
    print()


def main():
    if len(sys.argv) < 2:
        # Find most recent lag_data file
        import glob
        files = sorted(glob.glob("lag_data_*.csv"), reverse=True)
        if files:
            filename = files[0]
            print(f"[INFO] Using most recent file: {filename}")
        else:
            print("Usage: python3 analyze_lag.py <lag_data_file.csv>")
            print("       Or run lag_analyzer.py first to collect data")
            sys.exit(1)
    else:
        filename = sys.argv[1]

    print(f"[INFO] Loading {filename}...")
    data = load_data(filename)
    print(f"[INFO] Loaded {len(data)} samples")

    results = analyze_lag(data)
    print_report(results)


if __name__ == "__main__":
    main()
