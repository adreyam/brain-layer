"""
EMSC (Euro-Mediterranean Seismological Centre) Feed Adapter.
Fetches seismic events in the Mediterranean basin and parses them into seismic_catalog CityEvents.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import requests

from ..hazard_common import load_hazard_config


class EMSCAdapter:
    """Adapts EMSC Mediterranean seismic catalog into seismic_catalog CityEvents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_hazard_config()
        self.url = self.config.get("feeds", {}).get("emsc", {}).get(
            "url", "https://www.seismicportal.eu/fdsnws/event/1/query?format=json"
        )

    def parse_payload(self, emsc_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses EMSC GeoJSON response into CityEvents."""
        events = []
        features = emsc_payload.get("features", [])

        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [0.0, 0.0, 10.0])

            lon = float(coords[0])
            lat = float(coords[1])
            depth = float(coords[2]) if len(coords) > 2 else 10.0
            mag = float(props.get("mag") or 0.0)
            event_id = str(props.get("unid") or feat.get("id") or f"emsc-{int(time.time())}")

            events.append({
                "id": event_id,
                "source": "emsc_catalog",
                "kind": "seismic_catalog",
                "ts": time.time(),
                "lat": lat,
                "lon": lon,
                "data": {
                    "magnitude": mag,
                    "depth_km": depth,
                    "epicenter_lat": lat,
                    "epicenter_lon": lon,
                    "catalog_source": "emsc",
                    "flynn_region": props.get("flynn_region", ""),
                },
            })

        return events

    def fetch(self, min_mag: float = 4.0) -> List[Dict[str, Any]]:
        """Polls EMSC web service for recent Mediterranean earthquakes."""
        params = {"minmag": min_mag, "limit": 20}
        resp = requests.get(self.url, params=params, timeout=10.0)
        resp.raise_for_status()
        return self.parse_payload(resp.json())

