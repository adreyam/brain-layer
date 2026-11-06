"""
Integration tests for EmergencyAgent handling Catastrophic Hazards.
"""
import time
import pytest
from src.modules.emergency_agent import EmergencyAgent
from mocks.mock_plum import MOCK_PLUM_STRONG_MOTION_EVENT
from mocks.mock_crime_hotspot import MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT


def test_emergency_agent_regression_existing_kinds():
    agent = EmergencyAgent()

    # 1. PLUM seismic sensor
    resp_plum = agent.process_event(MOCK_PLUM_STRONG_MOTION_EVENT)
    assert resp_plum["plumAlertTriggered"] is True
    assert resp_plum["severity"] >= 4
    assert resp_plum["status"] == "ok"

    # 2. Crime gunshot triangulation
    resp_crime = agent.process_event(MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT)
    assert resp_crime["severity"] == 5
    assert resp_crime["status"] == "ok"

    # 3. Direct trigger: accident
    resp_acc = agent.process_event({"kind": "accident", "source": "police", "data": {"severity": 4}})
    assert resp_acc["severity"] == 4
    assert any(a["type"] == "dispatch_units" for a in resp_acc["actions"])


def test_emergency_agent_flood_dispatch():
    agent = EmergencyAgent()
    now = time.time()

    # Send multi-event flood inputs
    agent.process_event({
        "kind": "weather_forecast",
        "source": "open_meteo",
        "ts": now,
        "data": {"rain_24h_forecast_mm": 95.0},
    })
    agent.process_event({
        "kind": "weather_observation",
        "source": "iot_station",
        "ts": now,
        "data": {"rain_rate_mm_h": 28.0, "soil_saturation": 0.90},
    })
    resp_hydro = agent.process_event({
        "kind": "hydro_observation",
        "source": "river_gauge",
        "ts": now,
        "data": {"river_level_m": 4.9, "flood_stage_m": 4.5, "drainage_load": 0.85},
    })

    assert resp_hydro["status"] == "ok"
    assert resp_hydro["hazard"] == "flood"
    assert resp_hydro["risk_level"] == "critical"
    assert resp_hydro["severity"] == 5
    assert resp_hydro["requires_human_approval"] is True
    # Action items should include evacuation and CAP payload
    assert any(a["type"] == "evacuate" for a in resp_hydro["actions"])
    assert any(a["type"] == "alert_cap" for a in resp_hydro["actions"])


def test_emergency_agent_tsunami_dispatch():
    agent = EmergencyAgent()
    now = time.time()

    # Step 1: Undersea earthquake catalog event
    agent.process_event({
        "kind": "seismic_catalog",
        "source": "emsc_feed",
        "ts": now,
        "lat": 37.60,
        "lon": 10.45,
        "data": {
            "magnitude": 7.3,
            "depth_km": 12.0,
            "epicenter_lat": 37.60,
            "epicenter_lon": 10.45,
            "offshore": True,
            "catalog_source": "emsc",
        },
    })

    # Step 2: Tide gauge sea-level anomaly
    resp_tide = agent.process_event({
        "kind": "sea_level_observation",
        "source": "tide_gauge_station",
        "ts": now,
        "data": {
            "station_id": "TG-PORT",
            "sea_level_m": 1.75,
            "baseline_m": 0.90,
            "change_m_per_min": 0.45,
        },
    })

    assert resp_tide["hazard"] == "tsunami"
    assert resp_tide["risk_level"] == "critical"
    assert resp_tide["requires_human_approval"] is True
    assert any(a["type"] == "evacuate" for a in resp_tide["actions"])


def test_emergency_agent_storm_dispatch():
    agent = EmergencyAgent()
    now = time.time()

    # Severe storm with rotation
    agent.process_event({
        "kind": "official_weather_alert",
        "source": "national_met",
        "ts": now,
        "data": {"alert_type": "severe_storm", "alert_level": "red"},
    })
    resp_storm = agent.process_event({
        "kind": "weather_observation",
        "source": "radar",
        "ts": now,
        "data": {"radar_rotation": True, "gust_kmh": 115.0},
    })

    assert resp_storm["hazard"] == "storm"
    assert resp_storm["risk_level"] == "critical"
    assert resp_storm["requires_human_approval"] is True
    assert any("interior room" in a["detail"].lower() for a in resp_storm["actions"])

