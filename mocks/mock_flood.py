"""
Mock test data and execution scenarios for the Flood Hazard Engine.
Includes:
- Scenario 1: Heavy forecast on saturated soil (Critical)
- Scenario 2: Moderate rain / runoff (Watch)
- Scenario 3: Dry baseline conditions (Low)
- Scenario 4: Missing optional inputs (Confidence reduced)
- Scenario 5: Stale data (Ignored)
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

from modules.flood_engine import FloodEngine
from modules.emergency_agent import EmergencyAgent

NOW = time.time()

MOCK_FLOOD_CRITICAL_SCENARIO: List[Dict[str, Any]] = [
    {
        "id": "evt-fl-fc-01",
        "source": "open_meteo_forecast",
        "kind": "weather_forecast",
        "ts": NOW - 300,
        "lat": 36.80,
        "lon": 10.18,
        "data": {"rain_24h_forecast_mm": 95.0, "horizon_h": 24},
    },
    {
        "id": "evt-fl-obs-01",
        "source": "iot_weather_station",
        "kind": "weather_observation",
        "ts": NOW - 120,
        "lat": 36.80,
        "lon": 10.18,
        "data": {"rain_rate_mm_h": 28.5, "soil_saturation": 0.92},
    },
    {
        "id": "evt-fl-hydro-01",
        "source": "river_gauge_net",
        "kind": "hydro_observation",
        "ts": NOW - 60,
        "lat": 36.80,
        "lon": 10.18,
        "data": {"river_level_m": 4.8, "flood_stage_m": 4.5, "drainage_load": 0.88},
    },
]

MOCK_FLOOD_WATCH_SCENARIO: List[Dict[str, Any]] = [
    {
        "id": "evt-fl-fc-02",
        "source": "open_meteo_forecast",
        "kind": "weather_forecast",
        "ts": NOW - 600,
        "lat": 36.80,
        "lon": 10.18,
        "data": {"rain_24h_forecast_mm": 45.0},
    },
    {
        "id": "evt-fl-obs-02",
        "source": "iot_weather_station",
        "kind": "weather_observation",
        "ts": NOW - 180,
        "lat": 36.80,
        "lon": 10.18,
        "data": {"rain_rate_mm_h": 8.0, "soil_saturation": 0.40},
    },
]

MOCK_FLOOD_DRY_SCENARIO: List[Dict[str, Any]] = [
    {
        "id": "evt-fl-fc-03",
        "source": "open_meteo_forecast",
        "kind": "weather_forecast",
        "ts": NOW - 600,
        "lat": 36.80,
        "lon": 10.18,
        "data": {"rain_24h_forecast_mm": 2.0},
    },
    {
        "id": "evt-fl-obs-03",
        "source": "iot_weather_station",
        "kind": "weather_observation",
        "ts": NOW - 180,
        "lat": 36.80,
        "lon": 10.18,
        "data": {"rain_rate_mm_h": 0.0, "soil_saturation": 0.15},
    },
]

MOCK_FLOOD_MISSING_INPUTS_SCENARIO: List[Dict[str, Any]] = [
    {
        "id": "evt-fl-obs-04",
        "source": "isolated_sensor",
        "kind": "weather_observation",
        "ts": NOW - 100,
        "lat": 36.80,
        "lon": 10.18,
        "data": {"rain_rate_mm_h": 22.0},
    },
]

MOCK_FLOOD_STALE_SCENARIO: List[Dict[str, Any]] = [
    {
        "id": "evt-fl-obs-05",
        "source": "old_station",
        "kind": "weather_observation",
        "ts": NOW - 100000,  # ~27 hours old -> stale
        "lat": 36.80,
        "lon": 10.18,
        "data": {"rain_rate_mm_h": 40.0},
    },
]


def get_mock_flood_scenario(name: str = "critical") -> List[Dict[str, Any]]:
    scenarios = {
        "critical": MOCK_FLOOD_CRITICAL_SCENARIO,
        "watch": MOCK_FLOOD_WATCH_SCENARIO,
        "dry": MOCK_FLOOD_DRY_SCENARIO,
        "missing": MOCK_FLOOD_MISSING_INPUTS_SCENARIO,
        "stale": MOCK_FLOOD_STALE_SCENARIO,
    }
    return [dict(e) for e in scenarios[name]]


if __name__ == "__main__":
    print("=" * 60)
    print("FLOOD HAZARD ENGINE MOCK VERIFICATION")
    print("=" * 60)

    engine = FloodEngine()
    agent = EmergencyAgent()

    # 1. Critical Scenario
    res_crit = engine.evaluate(MOCK_FLOOD_CRITICAL_SCENARIO, now_ts=NOW)
    print(f"\n1. Critical Scenario: Level={res_crit.risk_level}, Score={res_crit.risk_score}")
    assert res_crit.risk_level == "critical"

    # 2. Watch Scenario
    res_watch = engine.evaluate(MOCK_FLOOD_WATCH_SCENARIO, now_ts=NOW)
    print(f"2. Watch Scenario:    Level={res_watch.risk_level}, Score={res_watch.risk_score}")
    assert res_watch.risk_level in {"watch", "low"}

    # 3. Dry Scenario
    res_dry = engine.evaluate(MOCK_FLOOD_DRY_SCENARIO, now_ts=NOW)
    print(f"3. Dry Scenario:      Level={res_dry.risk_level}, Score={res_dry.risk_score}")
    assert res_dry.risk_level == "low"

    # 4. Missing inputs (confidence check)
    res_miss = engine.evaluate(MOCK_FLOOD_MISSING_INPUTS_SCENARIO, now_ts=NOW)
    print(f"4. Missing Inputs:    Confidence={res_miss.confidence} (expected < 0.8)")
    assert res_miss.confidence < 0.8

    # 5. Stale data
    res_stale = engine.evaluate(MOCK_FLOOD_STALE_SCENARIO, now_ts=NOW)
    print(f"5. Stale Scenario:    Level={res_stale.risk_level}, Evidence={res_stale.evidence[0]}")
    assert "Ignored stale event" in res_stale.evidence[0]

    # 6. End-to-end through EmergencyAgent
    print("\nTesting end-to-end ingestion in EmergencyAgent:")
    agent_resp = agent.process_event(MOCK_FLOOD_CRITICAL_SCENARIO[2])
    print(f"   Agent Severity: {agent_resp.get('severity')} | Approval Required: {agent_resp.get('requires_human_approval')}")
    print(f"   Actions generated: {len(agent_resp.get('actions', []))}")

    print("\nFlood mock verification completed successfully.")

