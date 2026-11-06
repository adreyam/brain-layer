"""
Mock test data and execution scenarios for Tsunami Hazard Engine.
Includes:
- Offshore M7.0 shallow without tide gauge confirmation (Warning)
- Offshore M7.0 with sea-level anomaly confirmation (Critical + Human Approval)
- Onshore or deep earthquake (No tsunami / Low)
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

from modules.hazard_common import HazardAssessment
from modules.tsunami_engine import TsunamiEngine
from modules.emergency_agent import EmergencyAgent

NOW = time.time()

# Upstream offshore quake candidate
MOCK_TSUNAMI_OFFSHORE_QUAKE_EVENT: Dict[str, Any] = {
    "id": "evt-tsunami-quake-01",
    "source": "emsc_feed",
    "kind": "seismic_catalog",
    "ts": NOW - 300,
    "lat": 37.50,
    "lon": 10.40,
    "data": {
        "magnitude": 7.2,
        "depth_km": 14.0,
        "epicenter_lat": 37.50,
        "epicenter_lon": 10.40,
        "offshore": True,
        "catalog_source": "emsc",
    },
}

# Sea level gauge anomaly confirmation
MOCK_SEA_LEVEL_ANOMALY_EVENT: Dict[str, Any] = {
    "id": "evt-tide-anomaly-01",
    "source": "la_goulette_tide_station",
    "kind": "sea_level_observation",
    "ts": NOW - 60,
    "lat": 36.81,
    "lon": 10.31,
    "data": {
        "station_id": "TUN-TG-01",
        "sea_level_m": 1.65,
        "baseline_m": 0.85,
        "change_m_per_min": 0.45,
    },
}

# Onshore earthquake (negative case)
MOCK_ONSHORE_QUAKE_EVENT: Dict[str, Any] = {
    "id": "evt-onshore-quake-02",
    "source": "usgs_feed",
    "kind": "seismic_catalog",
    "ts": NOW - 300,
    "lat": 36.50,
    "lon": 9.80,  # Inland
    "data": {
        "magnitude": 6.8,
        "depth_km": 10.0,
        "epicenter_lat": 36.50,
        "epicenter_lon": 9.80,
        "offshore": False,
        "catalog_source": "usgs",
    },
}


def get_mock_tsunami_event(name: str = "offshore_quake") -> Dict[str, Any]:
    scenarios = {
        "offshore_quake": MOCK_TSUNAMI_OFFSHORE_QUAKE_EVENT,
        "sea_level_anomaly": MOCK_SEA_LEVEL_ANOMALY_EVENT,
        "onshore_quake": MOCK_ONSHORE_QUAKE_EVENT,
    }
    return dict(scenarios[name])


if __name__ == "__main__":
    print("=" * 60)
    print("TSUNAMI HAZARD ENGINE MOCK VERIFICATION")
    print("=" * 60)

    engine = TsunamiEngine()
    agent = EmergencyAgent()

    fake_quake_assessment = HazardAssessment(
        hazard="earthquake",
        risk_level="critical",
        risk_score=0.90,
        confidence=0.92,
        metadata={
            "magnitude": 7.2,
            "depth_km": 14.0,
            "epicenter_lat": 37.50,
            "epicenter_lon": 10.40,
            "offshore": True,
            "catalog_source": "emsc",
        },
    )

    # 1. Candidate without sea level confirmation
    res_warn = engine.evaluate(fake_quake_assessment, now_ts=NOW)
    print(f"\n1. Candidate Tsunami (No Sea Level): Level={res_warn.risk_level}, Score={res_warn.risk_score}, ETA={res_warn.eta_minutes}min")
    assert res_warn.risk_level == "warning"

    # 2. Confirmed with Sea Level Anomaly
    res_crit = engine.evaluate(fake_quake_assessment, [MOCK_SEA_LEVEL_ANOMALY_EVENT], now_ts=NOW)
    print(f"2. Confirmed Tsunami (With Gauge): Level={res_crit.risk_level}, Score={res_crit.risk_score}, Sources={res_crit.independent_source_count}")
    assert res_crit.risk_level == "critical"
    assert res_crit.needs_human_confirmation is True

    # 3. Onshore quake check
    onshore_assessment = HazardAssessment(
        hazard="earthquake",
        risk_level="warning",
        risk_score=0.70,
        metadata={
            "magnitude": 6.8,
            "depth_km": 10.0,
            "epicenter_lat": 36.50,
            "epicenter_lon": 9.80,
            "offshore": False,
        },
    )
    res_onshore = engine.evaluate(onshore_assessment, now_ts=NOW)
    print(f"3. Onshore Quake:                  Level={res_onshore.risk_level}, Score={res_onshore.risk_score}")
    assert res_onshore.risk_level == "low"

    # 4. End-to-end via EmergencyAgent
    print("\nTesting end-to-end ingestion in EmergencyAgent:")
    agent.process_event(MOCK_TSUNAMI_OFFSHORE_QUAKE_EVENT)
    agent_resp = agent.process_event(MOCK_SEA_LEVEL_ANOMALY_EVENT)
    print(f"   Agent Severity: {agent_resp.get('severity')} | Risk Level: {agent_resp.get('risk_level')}")
    print(f"   Requires Approval: {agent_resp.get('requires_human_approval')}")
    print(f"   Actions: {len(agent_resp.get('actions', []))}")

    print("\nTsunami mock verification completed successfully.")

