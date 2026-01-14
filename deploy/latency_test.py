#!/usr/bin/env python3
"""
Latency Test Script - Measure round-trip time to Kalshi and NWS APIs.

Run this on your VPS to verify you have low latency.
Target: <10ms to Kalshi, <50ms to NWS

Usage:
    python3 latency_test.py
"""

import asyncio
import time
import socket
import statistics
from typing import List, Tuple

import aiohttp


async def measure_latency(url: str, count: int = 10) -> Tuple[float, float, float]:
    """
    Measure HTTP latency to a URL.
    Returns: (min_ms, avg_ms, max_ms)
    """
    latencies: List[float] = []

    # Use connection pooling like the real bot
    connector = aiohttp.TCPConnector(
        limit=5,
        ttl_dns_cache=300,
        keepalive_timeout=60,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(count):
            start = time.perf_counter()
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    await resp.read()
                    elapsed = (time.perf_counter() - start) * 1000  # ms
                    latencies.append(elapsed)
            except Exception as e:
                print(f"  Request {i+1} failed: {e}")

            # Small delay between requests
            await asyncio.sleep(0.1)

    if not latencies:
        return (0, 0, 0)

    return (
        min(latencies),
        statistics.mean(latencies),
        max(latencies)
    )


def dns_lookup_time(hostname: str) -> float:
    """Measure DNS lookup time in ms."""
    start = time.perf_counter()
    try:
        socket.gethostbyname(hostname)
    except:
        pass
    return (time.perf_counter() - start) * 1000


async def main():
    print("=" * 60)
    print("        LATENCY TEST - NYC Weather Arb Bot")
    print("=" * 60)
    print()

    # Test endpoints
    endpoints = [
        ("Kalshi API", "https://api.elections.kalshi.com/trade-api/v2/exchange/status"),
        ("NWS JSON", "https://api.weather.gov/stations/KNYC/observations/latest"),
        ("NWS XML", "https://www.weather.gov/xml/current_obs/KNYC.xml"),
    ]

    # DNS Tests
    print("[1] DNS Lookup Times:")
    print("-" * 40)
    dns_hosts = [
        "api.elections.kalshi.com",
        "api.weather.gov",
        "www.weather.gov",
    ]
    for host in dns_hosts:
        dns_time = dns_lookup_time(host)
        print(f"  {host}: {dns_time:.2f}ms")
    print()

    # HTTP Latency Tests
    print("[2] HTTP Round-Trip Latency (10 requests each):")
    print("-" * 40)

    results = {}
    for name, url in endpoints:
        print(f"  Testing {name}...")
        min_ms, avg_ms, max_ms = await measure_latency(url, count=10)
        results[name] = avg_ms

        # Color-code results
        if avg_ms < 10:
            grade = "EXCELLENT"
        elif avg_ms < 50:
            grade = "GOOD"
        elif avg_ms < 100:
            grade = "OK"
        else:
            grade = "SLOW"

        print(f"    Min: {min_ms:.1f}ms | Avg: {avg_ms:.1f}ms | Max: {max_ms:.1f}ms [{grade}]")

    print()
    print("=" * 60)
    print("                    VERDICT")
    print("=" * 60)

    kalshi_latency = results.get("Kalshi API", 999)
    nws_latency = min(results.get("NWS JSON", 999), results.get("NWS XML", 999))

    if kalshi_latency < 10:
        print("  Kalshi: OPTIMAL (<10ms)")
        print("    -> You likely share the same datacenter (AWS us-east-1)")
    elif kalshi_latency < 30:
        print("  Kalshi: GOOD (<30ms)")
        print("    -> Acceptable for arbitrage")
    else:
        print(f"  Kalshi: SLOW ({kalshi_latency:.0f}ms)")
        print("    -> Consider switching to AWS us-east-1 for lower latency")

    print()

    if nws_latency < 50:
        print("  NWS: EXCELLENT (<50ms)")
    elif nws_latency < 100:
        print("  NWS: GOOD (<100ms)")
    else:
        print(f"  NWS: SLOW ({nws_latency:.0f}ms)")
        print("    -> NWS CDN may be congested")

    print()
    total_loop = kalshi_latency + nws_latency
    print(f"  Estimated scan loop overhead: {total_loop:.0f}ms")
    print()

    if kalshi_latency > 50:
        print("  RECOMMENDATION: Move to AWS Free Tier (us-east-1)")
        print("  Expected improvement: 10-50ms faster order execution")
    else:
        print("  Your setup is optimized for latency arbitrage!")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
