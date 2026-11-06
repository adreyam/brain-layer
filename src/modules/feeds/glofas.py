"""
Copernicus GloFAS (Global Flood Awareness System) Stub Adapter.
Normalizes river discharge and hydrological flood forecasts into hydro_observation CityEvents.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..hazard_common import load_hazard_config


class GloFASAdapter:
    """Stub adapter for Copernicus GloFAS river discharge forecast service."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_hazard_config()

    def parse_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses GloFAS hydrological forecast record into hydro_observation CityEvent."""
        events = []
        now = time.time()
        station_id = payload.get("station_id", "glofas-gauge-01")
        lat = float(payload.get("lat", 36.80))
        lon = float(payload.get("lon", 10.18))

        events.append({
            "id": f"glofas-{station_id}-{int(now)}",
            "source": "copernicus_glofas",
            "kind": "hydro_observation",
            "ts": now,
            "lat": lat,
            "lon": lon,
            "data": {
                "river_level_m": float(payload.get("river_level_m", 2.5)),
                "flood_stage_m": float(payload.get("flood_stage_m", 4.0)),
                "rate_of_rise_m_h": float(payload.get("rate_of_rise_m_h", 0.1)),
                "drainage_load": float(payload.get("drainage_load", 0.5)),
            },
        })
        return events

