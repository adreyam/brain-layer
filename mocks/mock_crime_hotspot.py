"""
Mock test data and execution scenarios for the Crime Hotspot Predictive Engine
(based on Alhammadi (2025) Random Forest & Risk Terrain Architecture).

Includes:
- High-risk acoustic gunshot triangulation with adverse risk-terrain factors (Score >= 0.75, Severity 5)
- Elevated hotspot risk scenario (Score ~ 0.50)
- Low-risk ambient noise / baseline monitoring scenario (Score < 0.50)
- Multi-feature risk terrain context (guardianship distance, temporal night/weekend, historical baseline)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure parent and src directory are in python path for standalone execution
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.crime_hotspot import (
    CrimeHotspotEngine,
    IncidentContext,
    RawCrimeEvent,
    RiskTerrainContext,
)
from modules.emergency_agent import EmergencyAgent


# --------------------------------------------------------------------------- #
# Scenario 1: High-Risk Violent Incident (Gunshot + Adverse Risk Terrain)
# --------------------------------------------------------------------------- #
MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT: Dict[str, Any] = {
    "id": "evt-crime-gunshot-001",
    "source": "twin_urban_event_stream",
    "kind": "acoustic_gunshot_triangulation",
    "lat": 36.8050,
    "lon": 10.1830,
    "data": {
        "decibelLevel": 118.2,
        "severity": 5,
        "associatedContext": {
            "dist_to_police_station_m": 2500.0,   # >2km guardianship gap (+0.10)
            "dist_to_hospital_m": 1800.0,
            "historical_crime_count": 28,          # >20 high historical crime cell (+0.30)
            "is_weekend": True,                    # Weekend night multiplier (+0.15)
            "is_night": True,
            "street_lighting_status": "malfunctioning",  # Environmental vulnerability (+0.05)
        },
    },
}

# --------------------------------------------------------------------------- #
# Scenario 2: Elevated Hotspot Risk (Historical Crime Baseline Trigger)
# --------------------------------------------------------------------------- #
MOCK_CRIME_ELEVATED_RISK_EVENT: Dict[str, Any] = {
    "id": "evt-crime-elevated-002",
    "source": "twin_urban_event_stream",
    "kind": "crime_hotspot",
    "lat": 36.8080,
    "lon": 10.1860,
    "data": {
        "decibelLevel": 95.0,
        "severity": 3,
        "associatedContext": {
            "dist_to_police_station_m": 2200.0,   # >2km (+0.10)
            "dist_to_hospital_m": 1200.0,
            "historical_crime_count": 25,          # >20 (+0.30)
            "is_weekend": False,
            "is_night": True,
            "street_lighting_status": "malfunctioning",  # (+0.05)
        },
    },
}

# --------------------------------------------------------------------------- #
# Scenario 3: Low-Risk Ambient Noise (Below Critical Thresholds)
# --------------------------------------------------------------------------- #
MOCK_CRIME_LOW_RISK_EVENT: Dict[str, Any] = {
    "id": "evt-crime-low-003",
    "source": "twin_urban_event_stream",
    "kind": "crime_hotspot",
    "lat": 36.7990,
    "lon": 10.1750,
    "data": {
        "decibelLevel": 68.0,
        "severity": 1,
        "associatedContext": {
            "dist_to_police_station_m": 450.0,     # close guardianship
            "dist_to_hospital_m": 600.0,
            "historical_crime_count": 5,           # low baseline
            "is_weekend": False,
            "is_night": False,
            "street_lighting_status": "ok",
        },
    },
}


def get_mock_crime_event(scenario: str = "high_risk") -> Dict[str, Any]:
    """Retrieve a mock Crime Hotspot event dictionary by scenario key."""
    scenarios = {
        "high_risk": MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT,
        "gunshot": MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT,
        "elevated": MOCK_CRIME_ELEVATED_RISK_EVENT,
        "low_risk": MOCK_CRIME_LOW_RISK_EVENT,
    }
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(scenarios.keys())}")
    return scenarios[scenario].copy()


if __name__ == "__main__":
    print("=" * 60)
    print("CRIME HOTSPOT MOCK DATA & ALGORITHM VERIFICATION")
    print("=" * 60)

    # 1. Direct Engine Evaluation
    print("\n1. Testing CrimeHotspotEngine direct feature evaluation (High-Risk):")
    ctx_high = RiskTerrainContext(**MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT["data"]["associatedContext"])
    score, confidence, evidence = CrimeHotspotEngine.evaluate_incident_risk(
        decibel_level=MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT["data"]["decibelLevel"],
        context=ctx_high,
    )
    print(f"   Composite Hotspot Risk Score: {score:.2f} (Confidence: {confidence})")
    print("   Evidence trail:")
    for ev in evidence:
        print(f"     - {ev}")

    # 2. Raw Event Analysis Test
    print("\n2. Testing CrimeHotspotEngine.analyze_event:")
    analysis_score, classification, analysis_evidence = CrimeHotspotEngine.analyze_event(
        RawCrimeEvent(
            precursorEventId=MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT["id"],
            category=MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT["kind"],
            sensorSource=MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT["source"],
            decibelLevel=MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT["data"]["decibelLevel"],
            associatedContext=IncidentContext(**MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT["data"]["associatedContext"]),
        )
    )
    print(f"   Classification: {classification} (Score: {analysis_score})")

    # 3. Emergency Agent Integration Test
    print("\n3. Testing EmergencyAgent ingestion with mock gunshot event:")
    agent = EmergencyAgent()
    resp_high = agent.process_event(MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT)
    print(f"   Agent Severity:   {resp_high.get('severity')} | Confidence: {resp_high.get('confidence')}")
    print("   Agent Directives:")
    for act in resp_high.get("actions", []):
        print(f"     [{act.get('priority')}] {act.get('type')} on {act.get('resource')}: {act.get('detail')}")

    # 4. Low-Risk Test
    print("\n4. Testing EmergencyAgent ingestion with low-risk mock event:")
    resp_low = agent.process_event(MOCK_CRIME_LOW_RISK_EVENT)
    print(f"   Low Risk Severity: {resp_low.get('severity')} | Actions: {len(resp_low.get('actions', []))}")
    for act in resp_low.get("actions", []):
        print(f"     [{act.get('priority')}] {act.get('type')} on {act.get('resource')}: {act.get('detail')}")

    print("\nCrime Hotspot mock verification finished successfully.")
