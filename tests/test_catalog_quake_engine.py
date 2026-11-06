"""
Unit tests for CatalogQuakeEngine.
"""
import pytest
from src.modules.catalog_quake_engine import CatalogQuakeEngine, is_mediterranean_offshore


def test_is_mediterranean_offshore():
    # Northern Tunisian / Central Mediterranean waters
    assert is_mediterranean_offshore(37.5, 10.5) is True
    # Inland Sahara
    assert is_mediterranean_offshore(32.0, 9.0) is False


def test_quake_magnitude_thresholds():
    engine = CatalogQuakeEngine()

    # M3.8: Below min threshold (4.5)
    res_sub = engine.evaluate({
        "kind": "seismic_catalog",
        "data": {"magnitude": 3.8, "epicenter_lat": 36.8, "epicenter_lon": 10.2},
    })
    assert res_sub.risk_level == "low"
    assert "below minimum monitoring threshold" in res_sub.evidence[-1]

    # M5.0 nearby: Watch
    res_watch = engine.evaluate({
        "kind": "seismic_catalog",
        "data": {"magnitude": 5.0, "depth_km": 15.0, "epicenter_lat": 36.9, "epicenter_lon": 10.2},
    })
    assert res_watch.risk_level in {"watch", "warning"}

    # M6.8 nearby + PLUM co-detection: Critical
    res_crit = engine.evaluate(
        {
            "kind": "seismic_catalog",
            "source": "usgs",
            "data": {"magnitude": 6.8, "depth_km": 10.0, "epicenter_lat": 37.0, "epicenter_lon": 10.2},
        },
        plum_active=True,
    )
    assert res_crit.risk_level == "critical"
    assert res_crit.independent_source_count >= 2


def test_quake_distance_attenuation():
    engine = CatalogQuakeEngine()

    # M7.0 at 800km away (Beyond radius)
    res_far = engine.evaluate({
        "kind": "seismic_catalog",
        "data": {"magnitude": 7.0, "epicenter_lat": 44.0, "epicenter_lon": 15.0},
    })
    assert res_far.risk_level == "low"
    assert "exceeds monitoring radius" in res_far.evidence[-1]


def test_quake_single_source_critical_cap():
    engine = CatalogQuakeEngine()
    # Major M7.2 nearby, but PLUM is NOT active (single source)
    res = engine.evaluate(
        {
            "kind": "seismic_catalog",
            "source": "usgs_feed",
            "data": {"magnitude": 7.2, "depth_km": 10.0, "epicenter_lat": 36.9, "epicenter_lon": 10.2},
        },
        plum_active=False,
    )
    assert res.independent_source_count == 1
    # Must be capped at warning per Section 7.6
    assert res.risk_level == "warning"

