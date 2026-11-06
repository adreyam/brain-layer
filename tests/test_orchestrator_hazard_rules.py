"""
Unit and integration tests for Orchestrator rules with Catastrophic Hazard extensions.
"""
import pytest
from src.orchestrator import (
    CityEvent,
    decide,
    rule_cascades,
    rule_emergency,
)


def test_rule_emergency_seismic_catalog():
    ev_major = CityEvent(
        source="catalog",
        kind="seismic_catalog",
        data={"magnitude": 6.8, "offshore": True},
    )
    scores = rule_emergency(ev_major)
    assert any(s[0] == "emergency" and s[1] >= 0.90 for s in scores)

    ev_minor = CityEvent(
        source="catalog",
        kind="seismic_catalog",
        data={"magnitude": 4.6},
    )
    scores_minor = rule_emergency(ev_minor)
    assert any(s[0] == "emergency" and s[1] >= 0.50 for s in scores_minor)


def test_rule_emergency_hydro_observation():
    ev_flood = CityEvent(
        source="gauge",
        kind="hydro_observation",
        data={"river_level_m": 5.2, "flood_stage_m": 4.5},
    )
    scores = rule_emergency(ev_flood)
    assert any(s[0] == "emergency" and s[1] >= 0.80 for s in scores)


def test_rule_emergency_weather_observation():
    ev_tornado = CityEvent(
        source="radar",
        kind="weather_observation",
        data={"radar_rotation": True},
    )
    scores = rule_emergency(ev_tornado)
    assert any(s[0] == "emergency" and s[1] >= 0.90 for s in scores)

    ev_gust = CityEvent(
        source="station",
        kind="weather_observation",
        data={"gust_kmh": 85.0},
    )
    scores_gust = rule_emergency(ev_gust)
    assert any(s[0] == "emergency" and s[1] >= 0.60 for s in scores_gust)


def test_rule_cascades_hazards():
    # 1. Seismic cascade
    ev_seismic = CityEvent(source="catalog", kind="seismic_catalog", data={"magnitude": 6.5})
    cascades = rule_cascades(ev_seismic)
    axes = {c[0] for c in cascades}
    assert "mobility" in axes
    assert "energy" in axes
    assert "environment" in axes

    # 2. Flood cascade
    ev_flood = CityEvent(source="hydro", kind="hydro_observation", data={"river_level_m": 5.0, "flood_stage_m": 4.0})
    cascades_fl = rule_cascades(ev_flood)
    axes_fl = {c[0] for c in cascades_fl}
    assert "mobility" in axes_fl
    assert "energy" in axes_fl
    assert "environment" in axes_fl

    # 3. Storm cascade
    ev_storm = CityEvent(source="weather", kind="weather_observation", data={"gust_kmh": 90.0})
    cascades_st = rule_cascades(ev_storm)
    axes_st = {c[0] for c in cascades_st}
    assert "energy" in axes_st
    assert "mobility" in axes_st


def test_orchestrator_decide_hazard_end_to_end():
    ev = {
        "source": "sensor",
        "kind": "hydro_observation",
        "lat": 36.80,
        "lon": 10.18,
        "data": {"river_level_m": 4.9, "flood_stage_m": 4.5, "drainage_load": 0.85},
    }
    decision = decide(ev)
    assert decision["primary_axis"] == "emergency"
    assert "emergency" in decision["activated_axes"]
    assert len(decision["actions"]) > 0

