"""
Mock test data and execution scenarios for Catalog Earthquake Engine.
Includes:
- M6.8 shallow nearby (Critical with PLUM sensor confirmation)
- M5.0 nearby (Watch)
- M7.0 far away (Low local impact)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.catalog_quake_engine import CatalogQuakeEngine
from modules.emergency_agent import EmergencyAgent

MOCK_QUAKE_CRITICAL_NEARBY: Dict[str, Any] = {
    "id": "evt-quake-crit-01",
    "source": "usgs_feed",
    "kind": "seismic_catalog",
    "ts": time.time(),
    "lat": 37.05,
    "lon": 10.25,
    "data": {
        "magnitude": 6.8,
        "depth_km": 12.0,
        "epicenter_lat": 37.05,
        "epicenter_lon": 10.25,
        "offshore": True,
        "catalog_source": "usgs",
    },
}

MOCK_QUAKE_WATCH_NEARBY: Dict[str, Any] = {
    "id": "evt-quake-watch-02",
    "source": "emsc_feed",
    "kind": "seismic_catalog",
    "ts": time.time(),
    "lat": 36.95,
    "lon": 10.20,
    "data": {
        "magnitude": 5.0,
        "depth_km": 15.0,
        "epicenter_lat": 36.95,
        "epicenter_lon": 10.20,
        "offshore": False,
        "catalog_source": "emsc",
    },
}

MOCK_QUAKE_DISTANT_STRONG: Dict[str, Any] = {
    "id": "evt-quake-dist-03",
    "source": "usgs_feed",
    "kind": "seismic_catalog",
    "ts": time.time(),
    "lat": 42.50,
    "lon": 14.50,  # Central Italy, > 600km away
    "data": {
        "magnitude": 7.0,
        "depth_km": 10.0,
        "epicenter_lat": 42.50,
        "epicenter_lon": 14.50,
        "catalog_source": "usgs",
    },
}


def get_mock_catalog_quake(name: str = "critical") -> Dict[str, Any]:
    scenarios = {
        "critical": MOCK_QUAKE_CRITICAL_NEARBY,
        "watch": MOCK_QUAKE_WATCH_NEARBY,
        "distant": MOCK_QUAKE_DISTANT_STRONG,
    }
    return dict(scenarios[name])


if __name__ == "__main__":
    print("=" * 60)
    print("CATALOG QUAKE MOCK VERIFICATION")
    print("=" * 60)

    engine = CatalogQuakeEngine()
    agent = EmergencyAgent()

    # 1. Critical Nearby Quake with PLUM confirmation
    res_crit = engine.evaluate(MOCK_QUAKE_CRITICAL_NEARBY, plum_active=True)
    print(f"\n1. Critical Nearby: Level={res_crit.risk_level}, Score={res_crit.risk_score}, Sources={res_crit.independent_source_count}")
    assert res_crit.risk_level == "critical"
    assert res_crit.independent_source_count >= 2

    # 2. Watch Nearby Quake
    res_watch = engine.evaluate(MOCK_QUAKE_WATCH_NEARBY, plum_active=False)
    print(f"2. Watch Nearby:    Level={res_watch.risk_level}, Score={res_watch.risk_score}")
    assert res_watch.risk_level in {"watch", "warning"}

    # 3. Distant Quake (Beyond monitoring radius)
    res_dist = engine.evaluate(MOCK_QUAKE_DISTANT_STRONG)
    print(f"3. Distant Quake:   Level={res_dist.risk_level}, Score={res_dist.risk_score}")
    assert res_dist.risk_level == "low"

    # 4. End-to-end through EmergencyAgent
    print("\nTesting end-to-end ingestion in EmergencyAgent:")
    resp = agent.process_event(MOCK_QUAKE_CRITICAL_NEARBY)
    print(f"   Severity: {resp.get('severity')} | Risk Level: {resp.get('risk_level')}")
    print(f"   Requires Approval: {resp.get('requires_human_approval')}")
    print(f"   Actions: {len(resp.get('actions', []))}")

    print("\nCatalog Quake mock verification completed successfully.")
