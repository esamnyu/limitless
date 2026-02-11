#!/usr/bin/env python3
"""
WEATHER SNIPER - Shared NWS Client Module

Consolidated NWS API client for all weather trading bots.
Handles hourly forecasts, current observations, and MOS data.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp

from config import (
    StationConfig,
    NWS_TIMEOUT_TOTAL_SEC,
    NWS_TIMEOUT_CONNECT_SEC,
    FORECAST_HOURS_AHEAD,
    WIND_GUST_MULTIPLIER,
    WIND_GUST_THRESHOLD_MPH,
)
from strategies import HourlyForecast, MOSForecast

logger = logging.getLogger(__name__)


class NWSClient:
    """NWS API client for observations and hourly forecasts."""

    def __init__(self, station_config: StationConfig):
        self.station_config = station_config
        self.session: Optional[aiohttp.ClientSession] = None
        self.gridpoint_url: str = station_config.nws_hourly_forecast_url
        self.observation_url: str = station_config.nws_observation_url
        self.tz = ZoneInfo(station_config.timezone)

    async def start(self):
        """Initialize the HTTP session."""
        self.session = aiohttp.ClientSession(
            headers={
                "User-Agent": f"WeatherSniper/3.0 ({self.station_config.city_code})",
                "Accept": "application/geo+json",
            },
            timeout=aiohttp.ClientTimeout(
                total=NWS_TIMEOUT_TOTAL_SEC,
                connect=NWS_TIMEOUT_CONNECT_SEC
            ),
        )
        logger.info(
            f"NWS client initialized for {self.station_config.station_id} "
            f"(gridpoint: {self.gridpoint_url})"
        )

    async def stop(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            logger.debug("NWS client stopped")

    async def get_current_temp(self) -> Optional[float]:
        """Get current temperature from station observation."""
        try:
            async with self.session.get(self.observation_url) as resp:
                if resp.status != 200:
                    logger.warning(f"NWS observation returned status {resp.status}")
                    return None
                props = (await resp.json()).get("properties", {})
                temp_c = props.get("temperature", {}).get("value")
                if temp_c is None:
                    return None
                return round((temp_c * 1.8) + 32, 1)
        except asyncio.TimeoutError:
            logger.error("NWS observation request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"NWS observation HTTP error: {e}")
            return None
        except Exception as e:
            logger.exception(f"NWS observation unexpected error: {e}")
            return None

    async def get_hourly_forecast(self) -> list[HourlyForecast]:
        """Get hourly forecast including wind, precip, and dewpoint data."""
        try:
            async with self.session.get(self.gridpoint_url) as resp:
                if resp.status != 200:
                    logger.error(f"NWS hourly forecast returned status {resp.status}")
                    return []
                data = await resp.json()
                periods = data.get("properties", {}).get("periods", [])

                forecasts = []
                for p in periods[:FORECAST_HOURS_AHEAD]:
                    try:
                        time = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00"))
                        temp_f = float(p.get("temperature", 0))

                        # Parse wind speed
                        wind_str = p.get("windSpeed", "0 mph")
                        wind_match = re.search(r"(\d+)\s*(?:to\s*(\d+))?\s*mph", wind_str, re.I)
                        if wind_match:
                            wind_speed = float(wind_match.group(2) or wind_match.group(1))
                        else:
                            wind_speed = 0.0

                        # Estimate gusts if not provided
                        wind_gust = (
                            wind_speed * WIND_GUST_MULTIPLIER
                            if wind_speed > WIND_GUST_THRESHOLD_MPH
                            else wind_speed
                        )

                        # Parse precipitation probability
                        precip_val = p.get("probabilityOfPrecipitation", {}).get("value")
                        precip_prob = int(precip_val) if precip_val is not None else 0

                        # Parse dewpoint
                        dew_val = p.get("dewpoint", {}).get("value")
                        dew_c = float(dew_val) if dew_val is not None else 0.0
                        dew_f = (dew_c * 1.8) + 32

                        forecasts.append(HourlyForecast(
                            time=time,
                            temp_f=temp_f,
                            wind_speed_mph=wind_speed,
                            wind_gust_mph=wind_gust,
                            short_forecast=p.get("shortForecast", ""),
                            is_daytime=p.get("isDaytime", False),
                            precip_prob=precip_prob,
                            dewpoint_f=dew_f,
                        ))
                    except (KeyError, ValueError) as e:
                        logger.debug(f"Skipping malformed forecast period: {e}")
                        continue

                logger.info(f"Fetched {len(forecasts)} hourly forecast periods")
                return forecasts

        except asyncio.TimeoutError:
            logger.error("NWS hourly forecast request timed out")
            return []
        except aiohttp.ClientError as e:
            logger.error(f"NWS hourly forecast HTTP error: {e}")
            return []
        except Exception as e:
            logger.exception(f"NWS hourly forecast unexpected error: {e}")
            return []


class MOSClient:
    """Client for fetching MOS (Model Output Statistics) data."""

    def __init__(self, station_config: StationConfig):
        self.station_config = station_config
        self.session: Optional[aiohttp.ClientSession] = None
        self.tz = ZoneInfo(station_config.timezone)

    async def start(self):
        """Initialize the HTTP session."""
        self.session = aiohttp.ClientSession(
            headers={
                "User-Agent": "WeatherSniper/3.0 (contact: weather-sniper@example.com)",
                "Accept": "text/plain",
            },
            timeout=aiohttp.ClientTimeout(
                total=NWS_TIMEOUT_TOTAL_SEC,
                connect=NWS_TIMEOUT_CONNECT_SEC
            ),
        )
        logger.info(f"MOS client initialized for {self.station_config.station_id}")

    async def stop(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()

    async def fetch_mos(self, url: str, source: str) -> Optional[MOSForecast]:
        """Fetch and parse MOS bulletin."""
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(f"MOS {source} fetch returned {resp.status}")
                    return None
                text = await resp.text()
                return self._parse_mos(text, source)
        except asyncio.TimeoutError:
            logger.error(f"MOS {source} request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"MOS {source} HTTP error: {e}")
            return None
        except Exception as e:
            logger.exception(f"MOS {source} unexpected error: {e}")
            return None

    def _parse_mos(self, text: str, source: str) -> Optional[MOSForecast]:
        """Parse MOS text bulletin to extract max temperature."""
        try:
            lines = text.strip().split('\n')

            temp_line = None

            for line in lines:
                if line.strip().startswith('X/N') or line.strip().startswith('N/X'):
                    temp_line = line
                    break

            if not temp_line:
                logger.debug(f"Could not find X/N line in {source} MOS")
                return None

            parts = temp_line.split()
            if len(parts) < 2:
                return None

            temps = []
            for p in parts[1:]:
                try:
                    temps.append(int(p))
                except ValueError:
                    continue

            if not temps:
                return None

            max_temp = temps[0]
            min_temp = temps[1] if len(temps) > 1 else None

            valid_date = datetime.now(self.tz).date() + timedelta(days=1)

            return MOSForecast(
                source=source,
                valid_date=datetime(
                    valid_date.year, valid_date.month, valid_date.day,
                    tzinfo=self.tz
                ),
                max_temp_f=float(max_temp),
                min_temp_f=float(min_temp) if min_temp else 0.0,
            )

        except Exception as e:
            logger.debug(f"Error parsing {source} MOS: {e}")
            return None

    async def get_mav(self) -> Optional[MOSForecast]:
        """Get GFS MOS (MAV) forecast."""
        return await self.fetch_mos(self.station_config.mos_mav_url, "MAV")

    async def get_met(self) -> Optional[MOSForecast]:
        """Get NAM MOS (MET) forecast."""
        return await self.fetch_mos(self.station_config.mos_met_url, "MET")
