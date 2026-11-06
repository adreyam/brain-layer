"""
Common models, configuration loader, and utilities for Catastrophic Hazard engines.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field

RiskLevel = Literal["low", "watch", "warning", "critical"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "city_center": {"lat": 36.8065, "lon": 10.1815},
    "source_agreement": {"min_sources_for_critical": 2},
    "severity_mapping": {"low": 1, "watch": 2, "warning": 4, "critical": 5},
    "max_age_minutes": {
        "weather_forecast": 360,
        "weather_observation": 60,
        "hydro_observation": 60,
        "seismic_catalog": 120,
        "sea_level_observation": 60,
        "official_weather_alert": 180,
    },
    "flood": {
        "rain_24h_norm_mm": 100.0,
        "rain_now_norm_mm_h": 30.0,
        "weights": {
            "rain_24h_forecast": 0.35,
            "rain_now": 0.20,
            "soil_saturation": 0.20,
            "river_level": 0.15,
            "drainage_load": 0.10,
        },
        "cutoffs": {"low": 0.30, "watch": 0.50, "warning": 0.70},
    },
    "earthquake": {
        "min_magnitude": 4.5,
        "radius_km": 350.0,
        "shallow_depth_km": 30.0,
        "magnitude_thresholds": {"watch": 4.5, "warning": 5.5, "critical": 6.5},
    },
    "tsunami": {
        "min_magnitude": 6.5,
        "max_depth_km": 70.0,
        "wave_speed_kmh": 700.0,
        "sea_level_change_threshold_m": 0.30,
        "sea_level_window_min": 30,
        "region_bbox": {
            "lat_min": 30.0,
            "lat_max": 46.0,
            "lon_min": -6.0,
            "lon_max": 37.0,
        },
    },
    "storm": {
        "gust_warning_kmh": 75.0,
        "gust_critical_kmh": 105.0,
        "pressure_drop_hpa_3h": 5.0,
        "rain_rate_warning_mm_h": 25.0,
        "rain_rate_critical_mm_h": 50.0,
        "lightning_rate_critical": 30.0,
    },
    "feeds": {
        "open_meteo": {
            "forecast_url": "https://api.open-meteo.com/v1/forecast",
            "polling_interval_s": 900,
        },
        "usgs": {
            "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
            "polling_interval_s": 300,
        },
        "emsc": {
            "url": "https://www.seismicportal.eu/fdsnws/event/1/query?format=json",
            "polling_interval_s": 300,
        },
    },
}


def load_hazard_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads hazard configuration from YAML file or falls back to built-in defaults.
    Path lookup precedence:
      1. Explicit config_path argument
      2. CITY_BRAIN_HAZARD_CONFIG environment variable
      3. config/hazard_thresholds.yaml relative to project root
    """
    path_to_try = config_path or os.getenv("CITY_BRAIN_HAZARD_CONFIG")

    if not path_to_try:
        # Search relative to this file or current working directory
        current_dir = Path(__file__).resolve().parent
        candidate = current_dir.parent.parent / "config" / "hazard_thresholds.yaml"
        if candidate.exists():
            path_to_try = str(candidate)
        else:
            local_cand = Path("config/hazard_thresholds.yaml")
            if local_cand.exists():
                path_to_try = str(local_cand)

    if path_to_try and os.path.exists(path_to_try):
        try:
            with open(path_to_try, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    # Merge with defaults for any missing keys
                    merged = DEFAULT_CONFIG.copy()
                    for k, v in loaded.items():
                        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                            merged[k] = {**merged[k], **v}
                        else:
                            merged[k] = v
                    return merged
        except Exception:
            pass

    return DEFAULT_CONFIG.copy()


class RiskDriver(BaseModel):
    factor: str
    value: Union[float, int, str, bool]
    contribution: float = 0.0


class GeoPoint(BaseModel):
    lat: float
    lon: float


class AffectedArea(BaseModel):
    center: GeoPoint
    radius_km: float


class HazardAssessment(BaseModel):
    """
    Standard output of deterministic hazard assessment engines.
    
    IMPORTANT:
    The `risk_score` is a normalized 0.0 - 1.0 composite risk index;
    it is strictly NOT a probability.
    """
    hazard: str
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    drivers: List[RiskDriver] = Field(default_factory=list)
    affected_area: Optional[AffectedArea] = None
    eta_minutes: Optional[int] = None
    independent_source_count: int = 1
    needs_human_confirmation: bool = False
    evidence: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance in kilometers using the Haversine formula."""
    r = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def normalize_units(data: Dict[str, Any], evidence: List[str]) -> Dict[str, Any]:
    """
    Normalizes incoming measurements to canonical units:
      - rain/precipitation: mm, mm/h
      - speed: km/h
      - pressure: hPa
      - distance/elevation: meters, km
    Records any unit conversions in the evidence list.
    """
    norm = dict(data)

    # Precipitation rate: in/hr -> mm/h
    if "rain_rate_in_h" in norm:
        val_mm_h = norm.pop("rain_rate_in_h") * 25.4
        norm["rain_rate_mm_h"] = round(val_mm_h, 2)
        evidence.append(f"Normalized rain_rate_in_h to rain_rate_mm_h: {val_mm_h:.2f} mm/h")

    # Wind speed: mph or m/s or knots -> km/h
    if "wind_speed_mph" in norm:
        val_kmh = norm.pop("wind_speed_mph") * 1.60934
        norm["wind_speed_kmh"] = round(val_kmh, 1)
        evidence.append(f"Normalized wind_speed_mph to wind_speed_kmh: {val_kmh:.1f} km/h")
    elif "wind_speed_ms" in norm:
        val_kmh = norm.pop("wind_speed_ms") * 3.6
        norm["wind_speed_kmh"] = round(val_kmh, 1)
        evidence.append(f"Normalized wind_speed_ms to wind_speed_kmh: {val_kmh:.1f} km/h")
    elif "wind_speed_knots" in norm:
        val_kmh = norm.pop("wind_speed_knots") * 1.852
        norm["wind_speed_kmh"] = round(val_kmh, 1)
        evidence.append(f"Normalized wind_speed_knots to wind_speed_kmh: {val_kmh:.1f} km/h")

    # Gust speed: mph or knots -> km/h
    if "gust_mph" in norm:
        val_kmh = norm.pop("gust_mph") * 1.60934
        norm["gust_kmh"] = round(val_kmh, 1)
        evidence.append(f"Normalized gust_mph to gust_kmh: {val_kmh:.1f} km/h")
    elif "gust_knots" in norm:
        val_kmh = norm.pop("gust_knots") * 1.852
        norm["gust_kmh"] = round(val_kmh, 1)
        evidence.append(f"Normalized gust_knots to gust_kmh: {val_kmh:.1f} km/h")

    # Pressure: bar, mbar, atm, inHg -> hPa
    if "pressure_inhg" in norm:
        val_hpa = norm.pop("pressure_inhg") * 33.8639
        norm["pressure_hpa"] = round(val_hpa, 1)
        evidence.append(f"Normalized pressure_inhg to pressure_hpa: {val_hpa:.1f} hPa")
    elif "pressure_bar" in norm:
        val_hpa = norm.pop("pressure_bar") * 1000.0
        norm["pressure_hpa"] = round(val_hpa, 1)
        evidence.append(f"Normalized pressure_bar to pressure_hpa: {val_hpa:.1f} hPa")

    return norm


def apply_critical_source_cap(
    assessment: HazardAssessment,
    min_sources: int = 2
) -> HazardAssessment:
    """
    Enforces Section 7.6 safety rule:
    An assessment may not reach 'critical' unless independent_source_count >= min_sources.
    Otherwise, cap at 'warning' and record the constraint in evidence.
    """
    if assessment.risk_level == "critical" and assessment.independent_source_count < min_sources:
        assessment.evidence.append(
            f"Safety constraint applied: Capped from 'critical' to 'warning' because "
            f"independent_source_count={assessment.independent_source_count} < {min_sources}."
        )
        assessment.risk_level = "warning"
        # Scale risk_score down to warning ceiling (0.69) if it was in critical band
        if assessment.risk_score >= 0.70:
            assessment.risk_score = 0.69

    if assessment.risk_level == "critical":
        assessment.needs_human_confirmation = True

    return assessment


def map_risk_level_to_severity(level: RiskLevel, config: Optional[Dict[str, Any]] = None) -> int:
    """Maps risk level to standard Emergency Agent severity (1 to 5)."""
    cfg = config or load_hazard_config()
    mapping = cfg.get("severity_mapping", {"low": 1, "watch": 2, "warning": 4, "critical": 5})
    return mapping.get(level, 1)

