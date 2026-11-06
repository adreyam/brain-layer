"""
Unit tests for TsunamiEngine.
"""
import time
import pytest
from src.modules.tsunami_engine import TsunamiEngine
from src.modules.hazard_common import HazardAssessment


def test_tsunami_candidate_criteria():
    engine = TsunamiEngine()

    # Candidate: Offshore, M7.1, 15km depth in Mediterranean
    candidate_quake = HazardAssessment(
        hazard="earthquake",
        risk_level="critical",
        risk_score=0.90,
        metadata={
            "magnitude": 7.1,
            "depth_km": 15.0,
            "epicenter_lat": 37.8,
            "epicenter_lon": 10.5,
            "offshore": True,
            "catalog_source": "emsc",
        },
    )

    res_warn = engine.evaluate(candidate_quake)
    assert res_warn.risk_level == "warning"
    assert res_warn.metadata["candidate"] is True
    assert res_warn.eta_minutes is not None
    assert res_warn.eta_minutes > 0
    assert res_warn.needs_human_confirmation is True


def test_tsunami_elevated_to_critical_with_sea_level():
    engine = TsunamiEngine()
    now = time.time()

    candidate_quake = HazardAssessment(
        hazard="earthquake",
        risk_level="critical",
        risk_score=0.90,
        metadata={
            "magnitude": 7.3,
            "depth_km": 12.0,
            "epicenter_lat": 37.8,
            "epicenter_lon": 10.5,
            "offshore": True,
            "catalog_source": "emsc",
        },
    )

    sea_level_event = {
        "kind": "sea_level_observation",
        "source": "la_goulette_tide_station",
        "ts": now,
        "data": {
            "station_id": "TG-01",
            "sea_level_m": 1.70,
            "baseline_m": 0.80,
            "change_m_per_min": 0.50,
        },
    }

    res_crit = engine.evaluate(candidate_quake, [sea_level_event], now_ts=now)
    assert res_crit.risk_level == "critical"
    assert res_crit.independent_source_count >= 2
    assert res_crit.needs_human_confirmation is True


def test_tsunami_non_candidates():
    engine = TsunamiEngine()

    # Case A: Onshore
    quake_onshore = HazardAssessment(
        hazard="earthquake",
        risk_level="critical",
        risk_score=0.85,
        metadata={"magnitude": 7.2, "depth_km": 10.0, "epicenter_lat": 36.5, "epicenter_lon": 9.8, "offshore": False},
    )
    assert engine.evaluate(quake_onshore).risk_level == "low"

    # Case B: Deep (> 70 km)
    quake_deep = HazardAssessment(
        hazard="earthquake",
        risk_level="warning",
        risk_score=0.70,
        metadata={"magnitude": 7.2, "depth_km": 120.0, "epicenter_lat": 37.5, "epicenter_lon": 10.5, "offshore": True},
    )
    assert engine.evaluate(quake_deep).risk_level == "low"

    # Case C: Low magnitude (< 6.5)
    quake_small = HazardAssessment(
        hazard="earthquake",
        risk_level="watch",
        risk_score=0.50,
        metadata={"magnitude": 5.8, "depth_km": 10.0, "epicenter_lat": 37.5, "epicenter_lon": 10.5, "offshore": True},
    )
    assert engine.evaluate(quake_small).risk_level == "low"

