#!/usr/bin/env python3
"""Wind Penalty Verification for Jan 22 Recovery Trade."""

import asyncio
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import aiohttp


async def main():
    tz = ZoneInfo('America/New_York')
    now = datetime.now(tz)
    tomorrow = (now + timedelta(days=1)).date()

    print(f"=== JAN 22 WIND PENALTY VERIFICATION ===")
    print(f"Current time: {now.strftime('%Y-%m-%d %I:%M %p ET')}")
    print(f"Target date: {tomorrow}")
    print()

    # Fetch NWS hourly forecast
    url = "https://api.weather.gov/gridpoints/OKX/33,37/forecast/hourly"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={"User-Agent": "WeatherSniper/3.0"}) as resp:
            data = await resp.json()
            periods = data.get("properties", {}).get("periods", [])

            print("=" * 60)
            print("1. OFFICIAL NWS FORECAST HIGH (Jan 22)")
            print("=" * 60)

            max_temp = 0
            max_temp_time = None
            tomorrow_forecasts = []

            for p in periods:
                time = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00"))
                local_time = time.astimezone(tz)

                if local_time.date() == tomorrow:
                    temp = p.get("temperature", 0)
                    tomorrow_forecasts.append((local_time, p))
                    if temp > max_temp:
                        max_temp = temp
                        max_temp_time = local_time

            print(f"NWS Forecast High: {max_temp}°F")
            print(f"Peak Hour: {max_temp_time.strftime('%I:%M %p ET') if max_temp_time else 'N/A'}")
            print()

            print("=" * 60)
            print("2. WIND PROFILE (12 PM - 4 PM)")
            print("=" * 60)

            for local_time, p in tomorrow_forecasts:
                hour = local_time.hour
                if 12 <= hour <= 16:
                    temp = p.get("temperature", 0)
                    wind_str = p.get("windSpeed", "0 mph")
                    wind_dir = p.get("windDirection", "")

                    # Parse wind speed
                    match = re.search(r"(\d+)\s*(?:to\s*(\d+))?\s*mph", wind_str, re.I)
                    if match:
                        wind_low = int(match.group(1))
                        wind_high = int(match.group(2)) if match.group(2) else wind_low
                    else:
                        wind_low = wind_high = 0

                    # Estimate gusts (1.5x sustained if > 10 mph)
                    gust_estimate = int(wind_high * 1.5) if wind_high > 10 else wind_high

                    forecast = p.get("shortForecast", "")

                    print(f"{local_time.strftime('%I %p')}: {temp}°F | Wind: {wind_str} {wind_dir} | Gusts: ~{gust_estimate} mph")
                    print(f"         Sky: {forecast}")

            print()
            print("=" * 60)
            print("3. WIND PENALTY ASSESSMENT")
            print("=" * 60)

            # Check peak heating hours (12-4 PM)
            peak_winds = []
            peak_gusts = []
            sky_conditions = []

            for local_time, p in tomorrow_forecasts:
                hour = local_time.hour
                if 12 <= hour <= 16:
                    wind_str = p.get("windSpeed", "0 mph")
                    match = re.search(r"(\d+)\s*(?:to\s*(\d+))?\s*mph", wind_str, re.I)
                    if match:
                        wind_high = int(match.group(2)) if match.group(2) else int(match.group(1))
                        peak_winds.append(wind_high)
                        gust = int(wind_high * 1.5) if wind_high > 10 else wind_high
                        peak_gusts.append(gust)

                    forecast = p.get("shortForecast", "").lower()
                    sky_conditions.append(forecast)

            max_sustained = max(peak_winds) if peak_winds else 0
            max_gust = max(peak_gusts) if peak_gusts else 0

            print(f"Max Sustained Wind (12-4 PM): {max_sustained} mph")
            print(f"Est. Max Gust (12-4 PM): {max_gust} mph")
            print()

            # Determine if sunny
            is_sunny = any("sunny" in s or "clear" in s for s in sky_conditions)
            is_cloudy = any("cloudy" in s or "overcast" in s for s in sky_conditions)

            print(f"Sky Condition: {'SUNNY/PARTLY SUNNY' if is_sunny else 'CLOUDY' if is_cloudy else 'MIXED'}")
            print()

            # Calculate wind penalty
            if max_gust > 25:
                penalty = 2.0
                penalty_reason = "Heavy gusts >25 mph"
            elif max_gust > 15:
                penalty = 1.0
                penalty_reason = "Moderate gusts 15-25 mph"
            else:
                penalty = 0.0
                penalty_reason = "Light winds <15 mph"

            print("-" * 60)
            print(f"WIND PENALTY: -{penalty}°F ({penalty_reason})")
            print("-" * 60)

            # Sun + Wind analysis
            if is_sunny and penalty > 0:
                print("WARNING: SUN + WIND = MIXING PENALTY VALID")
                print("    Mechanical turbulence will suppress surface heating")
            elif is_cloudy:
                print("NOTE: CLOUDY = PENALTY REDUCED")
                print("    Cloud cover already limits heating, wind effect muted")
            else:
                print("OK: Conditions neutral")

            print()
            print("=" * 60)
            print("PHYSICS TARGET")
            print("=" * 60)
            physics_high = max_temp - penalty
            print(f"NWS Forecast:  {max_temp}°F")
            print(f"Wind Penalty:  -{penalty}°F")
            print(f"Physics High:  {physics_high}°F")
            print()

            # Bracket recommendation
            if physics_high >= 46:
                target_bracket = "46-47° (B46.5)"
            elif physics_high >= 44:
                target_bracket = "44-45° (B44.5)"
            elif physics_high >= 42:
                target_bracket = "42-43° (B42.5)"
            elif physics_high >= 40:
                target_bracket = "40-41° (B40.5)"
            else:
                target_bracket = "Below 40°"

            print(f">>> TARGET BRACKET: {target_bracket}")


if __name__ == "__main__":
    asyncio.run(main())
