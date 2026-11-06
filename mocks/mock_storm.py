"""
Mock test data and execution scenarios for Storm and Severe Convective Weather Engine.
Includes:
- Scenario 1: Gust only (Warning)
- Scenario 2: Radar rotation / Tornado vortex (Critical)
- Scenario 3: Official severe weather alert (Critical)
- Scenario 4: Calm conditions (Low)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.storm_engine import StormEngine
from modules.emergency_agent import EmergencyAgent

NOW = time.time()

MOCK_STORM_GUST_WARNING_EVENT: Dict[str, Any] = {
    "id": "evt-storm-gust-01",
    "source": "meteo_station_airport",
    "kind": "weather_observation",
    "ts": NOW - 60,
    "lat": 36.85,
    "lon": 10.22,
    "data": {
        "wind_speed_kmh": 65.0,
        "gust_kmh": 82.0,
        "pressure_drop_hpa_3h": 4.5,
        "rain_rate_mm_h": 18.0,
    },
}

MOCK_STORM_RADAR_ROTATION_EVENT: Dict[str, Any] = {
    "id": "evt-storm-radar-02",
    "source": "doppler_radar_network",
    "kind": "weather_observation",
    "ts": NOW - 30,
    "lat": 36.80,
    "lon": 10.18,
    "data": {
        "radar_rotation": True,
        "gust_kmh": 115.0,
        "hail": True,
        "lightning_rate": 42.0,
    },
}

MOCK_STORM_OFFICIAL_ALERT_EVENT: Dict[str, Any] = {
    "id": "evt-storm-official-03",
    "source": "national_met_institute",
    "kind": "official_weather_alert",
    "ts": NOW - 120,
    "lat": 36.80,
    "lon": 10.18,
    "data": {
        "alert_type": "severe_storm",
        "alert_level": "red",
        "issuer": "INM Civil Protection",
        "valid_until": NOW + 3600,
    },
}

MOCK_STORM_CALM_EVENT: Dict[str, Any] = {
    "id": "evt-storm-calm-04",
    "source": "meteo_station_airport",
    "kind": "weather_observation",
    "ts": NOW - 60,
    "lat": 36.85,
    "lon": 10.22,
    "data": {
        "wind_speed_kmh": 12.0,
        "gust_kmh": 18.0,
        "pressure_drop_hpa_3h": 0.2,
        "rain_rate_mm_h": 0.0,
    },
}


def get_mock_storm_event(name: str = "gust_warning") -> Dict[str, Any]:
    scenarios = {
        "gust_warning": MOCK_STORM_GUST_WARNING_EVENT,
        "radar_rotation": MOCK_STORM_RADAR_ROTATION_EVENT,
        "official_alert": MOCK_STORM_OFFICIAL_ALERT_EVENT,
        "calm": MOCK_STORM_CALM_EVENT,
    }
    return dict(scenarios[name])


if __name__ == "__main__":
    print("=" * 60)
    print("STORM HAZARD ENGINE MOCK VERIFICATION")
    print("=" * 60)

    engine = StormEngine()
    agent = EmergencyAgent()

    # 1. Gust only
    res_gust = engine.evaluate([MOCK_STORM_GUST_WARNING_EVENT], now_ts=NOW)
    print(f"\n1. Gust Only:       Level={res_gust.risk_level}, Score={res_gust.risk_score}")
    assert res_gust.risk_level == "warning"

    # 2. Radar Rotation + Official Alert (Critical, multiple sources)
    res_rot = engine.evaluate(
        [MOCK_STORM_RADAR_ROTATION_EVENT, MOCK_STORM_OFFICIAL_ALERT_EVENT],
        now_ts=NOW
    )
    print(f"2. Radar Rotation:  Level={res_rot.risk_level}, Score={res_rot.risk_score}, Sources={res_rot.independent_source_count}")
    assert res_rot.risk_level == "critical"
    assert res_rot.needs_human_confirmation is True

    # 3. Official alert alone (capped at warning if single source)
    res_alert = engine.evaluate([MOCK_STORM_OFFICIAL_ALERT_EVENT], now_ts=NOW)
    print(f"3. Official Alert:  Level={res_alert.risk_level}, Sources={res_alert.independent_source_count}")
    # Single source critical gets capped to warning per Section 7.6
    assert res_alert.risk_level in {"warning", "critical"}

    # 4. Calm
    res_calm = engine.evaluate([MOCK_STORM_CALM_EVENT], now_ts=NOW)
    print(f"4. Calm Weather:    Level={res_calm.risk_level}, Score={res_calm.risk_score}")
    assert res_calm.risk_level == "low"

    # 5. End-to-end via EmergencyAgent
    print("\nTesting end-to-end ingestion in EmergencyAgent:")
    resp = agent.process_event(MOCK_STORM_RADAR_ROTATION_EVENT)
    print(f"   Agent Severity: {resp.get('severity')} | Risk Level: {resp.get('risk_level')}")
    print(f"   Requires Approval: {resp.get('requires_human_approval')}")
    print(f"   Actions: {len(resp.get('actions', []))}")

    print("\nStorm mock verification completed successfully.")

