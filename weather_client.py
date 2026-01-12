#!/usr/bin/env python3
"""
Weather Client - Fetches professional weather model forecasts from Open-Meteo.

Provides GFS and ECMWF forecasts for high-temperature trading on Kalshi.
Both models are free via Open-Meteo's API.

Usage:
    from weather_client import WeatherClient
    client = WeatherClient()
    gfs, ecmwf = await client.get_nyc_forecasts()
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import aiohttp


@dataclass
class Forecast:
    """A weather forecast for a specific date."""

    date: str  # YYYY-MM-DD
    high_temp_f: float  # Daily high in Fahrenheit
    model: str  # "GFS" or "ECMWF"
    location: str  # "NYC" etc.
    fetched_at: datetime

    @property
    def bracket(self) -> str:
        """Return the Kalshi bracket this forecast falls into."""
        temp = self.high_temp_f
        if temp < 39:
            return "T39"  # <39
        elif temp < 41:
            return "B39.5"  # 39-40
        elif temp < 43:
            return "B41.5"  # 41-42
        elif temp < 45:
            return "B43.5"  # 43-44
        elif temp < 47:
            return "B45.5"  # 45-46
        else:
            return "T46"  # >46

    @property
    def bracket_range(self) -> str:
        """Human-readable bracket range."""
        brackets = {
            "T39": "<39°F",
            "B39.5": "39-40°F",
            "B41.5": "41-42°F",
            "B43.5": "43-44°F",
            "B45.5": "45-46°F",
            "T46": ">46°F",
        }
        return brackets.get(self.bracket, "Unknown")


class WeatherClient:
    """
    Async client for Open-Meteo weather API.

    Fetches forecasts from:
    - GFS (Global Forecast System) - NOAA, ~80% accuracy
    - ECMWF (European Centre) - ~85% accuracy, gold standard
    """

    # Open-Meteo API endpoints (free, no auth required)
    GFS_URL = "https://api.open-meteo.com/v1/forecast"
    ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"

    # NYC Central Park coordinates (where NWS CLI reports from)
    NYC_LAT = 40.7829
    NYC_LON = -73.9654

    # Model accuracy estimates
    GFS_ACCURACY = 0.80
    ECMWF_ACCURACY = 0.85
    COMBINED_ACCURACY = 0.90  # When both agree

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Initialize the client session."""
        self.session = aiohttp.ClientSession()
        print("[WEATHER] Client initialized")

    async def stop(self):
        """Close the client session."""
        if self.session:
            await self.session.close()

    async def get_gfs_forecast(
        self,
        lat: float = None,
        lon: float = None,
        days: int = 7
    ) -> list[Forecast]:
        """
        Get GFS model forecast.

        Args:
            lat: Latitude (default: NYC)
            lon: Longitude (default: NYC)
            days: Number of forecast days

        Returns:
            List of Forecast objects
        """
        lat = lat or self.NYC_LAT
        lon = lon or self.NYC_LON

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "forecast_days": days,
        }

        try:
            async with self.session.get(self.GFS_URL, params=params) as resp:
                if resp.status != 200:
                    print(f"[WEATHER] GFS API error: {resp.status}")
                    return []

                data = await resp.json()
                daily = data.get("daily", {})
                dates = daily.get("time", [])
                temps = daily.get("temperature_2m_max", [])

                forecasts = []
                for date, temp in zip(dates, temps):
                    if temp is not None:
                        forecasts.append(Forecast(
                            date=date,
                            high_temp_f=temp,
                            model="GFS",
                            location="NYC",
                            fetched_at=datetime.now(timezone.utc),
                        ))

                return forecasts

        except Exception as e:
            print(f"[WEATHER] GFS error: {e}")
            return []

    async def get_ecmwf_forecast(
        self,
        lat: float = None,
        lon: float = None,
        days: int = 7
    ) -> list[Forecast]:
        """
        Get ECMWF model forecast (higher accuracy).

        Args:
            lat: Latitude (default: NYC)
            lon: Longitude (default: NYC)
            days: Number of forecast days

        Returns:
            List of Forecast objects
        """
        lat = lat or self.NYC_LAT
        lon = lon or self.NYC_LON

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }

        try:
            async with self.session.get(self.ECMWF_URL, params=params) as resp:
                if resp.status != 200:
                    print(f"[WEATHER] ECMWF API error: {resp.status}")
                    return []

                data = await resp.json()
                daily = data.get("daily", {})
                dates = daily.get("time", [])
                temps = daily.get("temperature_2m_max", [])

                forecasts = []
                for date, temp in zip(dates, temps):
                    if temp is not None:
                        forecasts.append(Forecast(
                            date=date,
                            high_temp_f=temp,
                            model="ECMWF",
                            location="NYC",
                            fetched_at=datetime.now(timezone.utc),
                        ))

                return forecasts

        except Exception as e:
            print(f"[WEATHER] ECMWF error: {e}")
            return []

    async def get_nyc_forecasts(self) -> Tuple[list[Forecast], list[Forecast]]:
        """
        Get both GFS and ECMWF forecasts for NYC.

        Returns:
            Tuple of (gfs_forecasts, ecmwf_forecasts)
        """
        gfs, ecmwf = await asyncio.gather(
            self.get_gfs_forecast(),
            self.get_ecmwf_forecast(),
        )
        return gfs, ecmwf

    def get_model_consensus(
        self,
        gfs: Forecast,
        ecmwf: Forecast,
        max_divergence: float = 2.0
    ) -> Optional[dict]:
        """
        Check if models agree and return consensus forecast.

        Args:
            gfs: GFS forecast
            ecmwf: ECMWF forecast
            max_divergence: Max temp difference in °F for agreement

        Returns:
            Consensus dict or None if models disagree
        """
        if gfs.date != ecmwf.date:
            return None

        divergence = abs(gfs.high_temp_f - ecmwf.high_temp_f)

        if divergence > max_divergence:
            print(f"[WEATHER] Models disagree by {divergence:.1f}°F - skipping")
            return None

        avg_temp = (gfs.high_temp_f + ecmwf.high_temp_f) / 2

        # Determine bracket from average
        if avg_temp < 39:
            bracket = "T39"
            bracket_range = "<39°F"
        elif avg_temp < 41:
            bracket = "B39.5"
            bracket_range = "39-40°F"
        elif avg_temp < 43:
            bracket = "B41.5"
            bracket_range = "41-42°F"
        elif avg_temp < 45:
            bracket = "B43.5"
            bracket_range = "43-44°F"
        elif avg_temp < 47:
            bracket = "B45.5"
            bracket_range = "45-46°F"
        else:
            bracket = "T46"
            bracket_range = ">46°F"

        return {
            "date": gfs.date,
            "gfs_temp": gfs.high_temp_f,
            "ecmwf_temp": ecmwf.high_temp_f,
            "avg_temp": avg_temp,
            "divergence": divergence,
            "bracket": bracket,
            "bracket_range": bracket_range,
            "confidence": self.COMBINED_ACCURACY if divergence < 1.0 else self.ECMWF_ACCURACY,
        }


async def test_client():
    """Test the weather client."""
    client = WeatherClient()
    await client.start()

    print("\n[TEST] Fetching NYC weather forecasts...")
    print("=" * 60)

    gfs_forecasts, ecmwf_forecasts = await client.get_nyc_forecasts()

    print(f"\n[GFS] {len(gfs_forecasts)} days of forecasts:")
    for f in gfs_forecasts[:5]:
        print(f"  {f.date}: {f.high_temp_f:.1f}°F → {f.bracket_range}")

    print(f"\n[ECMWF] {len(ecmwf_forecasts)} days of forecasts:")
    for f in ecmwf_forecasts[:5]:
        print(f"  {f.date}: {f.high_temp_f:.1f}°F → {f.bracket_range}")

    # Check consensus for tomorrow
    if gfs_forecasts and ecmwf_forecasts:
        print("\n[CONSENSUS] Model Agreement Check:")
        print("-" * 60)

        # Match forecasts by date
        gfs_by_date = {f.date: f for f in gfs_forecasts}
        ecmwf_by_date = {f.date: f for f in ecmwf_forecasts}

        for date in sorted(set(gfs_by_date.keys()) & set(ecmwf_by_date.keys()))[:3]:
            gfs = gfs_by_date[date]
            ecmwf = ecmwf_by_date[date]

            consensus = client.get_model_consensus(gfs, ecmwf)

            if consensus:
                print(f"\n  {date}:")
                print(f"    GFS:   {consensus['gfs_temp']:.1f}°F")
                print(f"    ECMWF: {consensus['ecmwf_temp']:.1f}°F")
                print(f"    Avg:   {consensus['avg_temp']:.1f}°F")
                print(f"    Bracket: {consensus['bracket_range']} ({consensus['bracket']})")
                print(f"    Confidence: {consensus['confidence']:.0%}")
            else:
                print(f"\n  {date}: Models disagree - no consensus")

    await client.stop()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_client())
