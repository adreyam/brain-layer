"""
National Meteorological Institute Weather Alert Stub Adapter.
Normalizes official severe weather bulletins into official_weather_alert CityEvents.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..hazard_common import load_hazard_config


class WeatherAlertAdapter:
    """Adapter for official civil protection and meteorological institute alert bulletins."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_hazard_config()

    def parse_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses national weather bulletin into official_weather_alert CityEvent."""
        now = time.time()
        events = []

        events.append({
            "id": payload.get("alert_id", f"met-alert-{int(now)}"),
            "source": payload.get("issuer", "inm_tunisia"),
            "kind": "official_weather_alert",
            "ts": now,
            "lat": float(payload.get("lat", 36.80)),
            "lon": float(payload.get("lon", 10.18)),
            "data": {
                "alert_type": payload.get("alert_type", "severe_storm"),
                "alert_level": payload.get("alert_level", "orange"),
                "valid_until": payload.get("valid_until", now + 7200),
                "issuer": payload.get("issuer", "Institut National de la Meteorologie (INM)"),
                "description": payload.get("description", "Heavy precipitation and convective squalls expected."),
            },
        })
        return events

