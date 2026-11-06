"""
Unit tests for StormEngine.
"""
import time
import pytest
from src.modules.storm_engine import StormEngine


def test_storm_calm():
    engine = StormEngine()
    now = time.time()
    res = engine.evaluate([{
        "kind": "weather_observation",
        "source": "station_1",
        "ts": now,
        "data": {"wind_speed_kmh": 15.0, "gust_kmh": 22.0, "pressure_drop_hpa_3h": 0.5},
    }])
    assert res.risk_level == "low"
    assert res.risk_score < 0.30


def test_storm_gust_warning():
    engine = StormEngine()
    now = time.time()
    res = engine.evaluate([{
        "kind": "weather_observation",
        "source": "station_1",
        "ts": now,
        "data": {"gust_kmh": 85.0, "pressure_drop_hpa_3h": 4.0},
    }])
    assert res.risk_level == "warning"


def test_storm_radar_rotation_critical():
    engine = StormEngine()
    now = time.time()
    # 2 independent sources: radar + weather alert
    res = engine.evaluate([
        {
            "kind": "weather_observation",
            "source": "doppler_radar",
            "ts": now,
            "data": {"radar_rotation": True, "gust_kmh": 110.0},
        },
        {
            "kind": "official_weather_alert",
            "source": "inm_civil_protection",
            "ts": now,
            "data": {"alert_type": "severe_storm", "alert_level": "red"},
        },
    ])
    assert res.risk_level == "critical"
    assert res.needs_human_confirmation is True
    assert res.independent_source_count >= 2


def test_storm_single_source_critical_cap():
    engine = StormEngine()
    now = time.time()
    # Only single source reporting radar rotation
    res = engine.evaluate([
        {
            "kind": "weather_observation",
            "source": "doppler_radar",
            "ts": now,
            "data": {"radar_rotation": True},
        }
    ])
    assert res.independent_source_count == 1
    # Must be capped at warning per Section 7.6
    assert res.risk_level == "warning"
    assert any("Safety constraint applied: Capped from 'critical' to 'warning'" in e for e in res.evidence)

