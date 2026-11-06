"""
Unit tests for FloodEngine.
"""
import time
import pytest
from src.modules.flood_engine import FloodEngine
from src.modules.hazard_common import HazardAssessment


def test_flood_engine_empty_events():
    engine = FloodEngine()
    assessment = engine.evaluate([])
    assert assessment.hazard == "flood"
    assert assessment.risk_level == "low"
    assert assessment.risk_score == 0.0


def test_flood_level_boundaries():
    engine = FloodEngine()
    now = time.time()

    # Dry: Low
    res_low = engine.evaluate([
        {"kind": "weather_forecast", "source": "s1", "ts": now, "data": {"rain_24h_forecast_mm": 5.0}},
        {"kind": "weather_observation", "source": "s2", "ts": now, "data": {"rain_rate_mm_h": 1.0, "soil_saturation": 0.1}},
    ])
    assert res_low.risk_level == "low"
    assert res_low.risk_score < 0.30

    # Watch
    res_watch = engine.evaluate([
        {"kind": "weather_forecast", "source": "s1", "ts": now, "data": {"rain_24h_forecast_mm": 40.0}},
        {"kind": "weather_observation", "source": "s2", "ts": now, "data": {"rain_rate_mm_h": 10.0, "soil_saturation": 0.5}},
    ])
    assert res_watch.risk_level == "watch"
    assert 0.30 <= res_watch.risk_score < 0.50

    # Warning
    res_warn = engine.evaluate([
        {"kind": "weather_forecast", "source": "s1", "ts": now, "data": {"rain_24h_forecast_mm": 70.0}},
        {"kind": "weather_observation", "source": "s2", "ts": now, "data": {"rain_rate_mm_h": 20.0, "soil_saturation": 0.7}},
    ])
    assert res_warn.risk_level == "warning"
    assert 0.50 <= res_warn.risk_score < 0.70

    # Critical (Needs 2+ sources)
    res_crit = engine.evaluate([
        {"kind": "weather_forecast", "source": "src1", "ts": now, "data": {"rain_24h_forecast_mm": 95.0}},
        {"kind": "weather_observation", "source": "src2", "ts": now, "data": {"rain_rate_mm_h": 30.0, "soil_saturation": 0.95}},
        {"kind": "hydro_observation", "source": "src3", "ts": now, "data": {"river_level_m": 5.0, "flood_stage_m": 4.0, "drainage_load": 0.9}},
    ])
    assert res_crit.risk_level == "critical"
    assert res_crit.risk_score >= 0.70


def test_flood_missing_inputs_confidence_reduction():
    engine = FloodEngine()
    now = time.time()
    # Only single factor available
    res = engine.evaluate([
        {"kind": "weather_observation", "source": "s1", "ts": now, "data": {"rain_rate_mm_h": 25.0}}
    ])
    # Full factor suite yields confidence ~0.95; single factor should have confidence < 0.80
    assert res.confidence < 0.80


def test_flood_stale_inputs_ignored():
    engine = FloodEngine()
    now = time.time()
    stale_ts = now - (500 * 60)  # 500 minutes ago > 60m max age

    res = engine.evaluate([
        {"kind": "hydro_observation", "source": "s1", "ts": stale_ts, "data": {"river_level_m": 10.0, "flood_stage_m": 2.0}},
    ], now_ts=now)

    assert res.risk_level == "low"
    assert any("Ignored stale event" in e for e in res.evidence)


def test_flood_critical_capped_when_single_source():
    engine = FloodEngine()
    now = time.time()
    # Single source reporting catastrophic rainfall
    res = engine.evaluate([
        {
            "kind": "weather_forecast",
            "source": "single_feed",
            "ts": now,
            "data": {"rain_24h_forecast_mm": 120.0},
        },
        {
            "kind": "weather_observation",
            "source": "single_feed",  # Same source name!
            "ts": now,
            "data": {"rain_rate_mm_h": 50.0, "soil_saturation": 0.99},
        },
        {
            "kind": "hydro_observation",
            "source": "single_feed",  # Same source name!
            "ts": now,
            "data": {"river_level_m": 8.0, "flood_stage_m": 3.0, "drainage_load": 1.0},
        },
    ], now_ts=now)

    assert res.independent_source_count == 1
    # Must be capped at warning due to Section 7.6
    assert res.risk_level == "warning"
    assert any("Safety constraint applied: Capped from 'critical' to 'warning'" in e for e in res.evidence)

