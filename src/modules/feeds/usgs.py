"""
USGS Earthquake API Feed Adapter.
Fetches GeoJSON seismic events and parses them into seismic_catalog CityEvents.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import requests

from ..hazard_common import load_hazard_config


class USGSAdapter:
    """Adapts USGS GeoJSON earthquake feed into seismic_catalog CityEvents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_hazard_config()
        self.url = self.config.get("feeds", {}).get("usgs", {}).get(
            "url", "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
        )

    def parse_payload(self, geojson_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses USGS FeatureCollection GeoJSON into CityEvents."""
        events = []
        features = geojson_payload.get("features", [])

        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [0.0, 0.0, 10.0])

            lon = float(coords[0])
            lat = float(coords[1])
            depth = float(coords[2]) if len(coords) > 2 else 10.0
            mag = float(props.get("mag") or 0.0)
            event_time = props.get("time")
            ts = (float(event_time) / 1000.0) if event_time else time.time()
            event_id = feat.get("id", f"usgs-{int(ts)}")

            events.append({
                "id": event_id,
                "source": "usgs_catalog",
                "kind": "seismic_catalog",
                "ts": ts,
                "lat": lat,
                "lon": lon,
                "data": {
                    "magnitude": mag,
                    "depth_km": depth,
                    "epicenter_lat": lat,
                    "epicenter_lon": lon,
                    "event_time": ts,
                    "catalog_source": "usgs",
                    "title": props.get("title", ""),
                },
            })

        return events

    def fetch(self) -> List[Dict[str, Any]]:
        """Polls USGS API and returns list of seismic_catalog CityEvent dictionaries."""
        resp = requests.get(self.url, timeout=10.0)
        resp.raise_for_status()
        return self.parse_payload(resp.json())

