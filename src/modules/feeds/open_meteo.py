"""
Open-Meteo external feed adapter.
Fetches free weather forecast and current weather observations and converts them to CityEvent dictionaries.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import requests

from ..hazard_common import load_hazard_config


class OpenMeteoAdapter:
    """Adapts Open-Meteo API payload into weather_forecast and weather_observation CityEvents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_hazard_config()
        self.center = self.config.get("city_center", {"lat": 36.8065, "lon": 10.1815})
        self.url = self.config.get("feeds", {}).get("open_meteo", {}).get(
            "forecast_url", "https://api.open-meteo.com/v1/forecast"
        )

    def parse_payload(
        self,
        payload: Dict[str, Any],
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Parses Open-Meteo JSON response into standard CityEvent dictionaries."""
        events = []
        now = time.time()
        c_lat = lat or self.center.get("lat", 36.8065)
        c_lon = lon or self.center.get("lon", 10.1815)

        # 1. Current observation
        current = payload.get("current", payload.get("current_weather", {}))
        if current:
            rain_rate = current.get("precipitation", current.get("rain", 0.0))
            wind_speed = current.get("wind_speed_10m", current.get("windspeed", 0.0))
            wind_gust = current.get("wind_gusts_10m", 0.0)
            pressure = current.get("surface_pressure", current.get("pressure_msl", 1013.25))

            events.append({
                "id": f"om-obs-{int(now)}",
                "source": "open_meteo_live",
                "kind": "weather_observation",
                "ts": now,
                "lat": c_lat,
                "lon": c_lon,
                "data": {
                    "rain_rate_mm_h": float(rain_rate),
                    "wind_speed_kmh": float(wind_speed),
                    "gust_kmh": float(wind_gust),
                    "pressure_hpa": float(pressure),
                },
            })

        # 2. Hourly/Daily forecast
        daily = payload.get("daily", {})
        if "precipitation_sum" in daily and daily["precipitation_sum"]:
            r24 = float(daily["precipitation_sum"][0])
            events.append({
                "id": f"om-fc-{int(now)}",
                "source": "open_meteo_forecast",
                "kind": "weather_forecast",
                "ts": now,
                "lat": c_lat,
                "lon": c_lon,
                "data": {
                    "rain_24h_forecast_mm": r24,
                    "horizon_h": 24,
                },
            })

        return events

    def fetch(self, lat: Optional[float] = None, lon: Optional[float] = None) -> List[Dict[str, Any]]:
        """Fetches live Open-Meteo API data and returns CityEvent dictionaries."""
        c_lat = lat or self.center.get("lat", 36.8065)
        c_lon = lon or self.center.get("lon", 10.1815)

        params = {
            "latitude": c_lat,
            "longitude": c_lon,
            "current": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,surface_pressure",
            "daily": "precipitation_sum,wind_gusts_10m_max",
            "timezone": "auto",
        }
        resp = requests.get(self.url, params=params, timeout=10.0)
        resp.raise_for_status()
        return self.parse_payload(resp.json(), lat=c_lat, lon=c_lon)

