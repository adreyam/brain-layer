"""
Deterministic Flood Hazard Assessment Engine.
Computes a composite flood risk index based on hydrological telemetry, current rainfall,
soil saturation, and forecasted precipitation.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Union

from .hazard_common import (
    AffectedArea,
    GeoPoint,
    HazardAssessment,
    RiskDriver,
    apply_critical_source_cap,
    load_hazard_config,
    normalize_units,
)


class FloodEngine:
    """
    Deterministic rule-based assessment for pluvial and fluvial flood risk.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_hazard_config()
        self.flood_cfg = self.config.get("flood", {})
        self.weights = self.flood_cfg.get("weights", {
            "rain_24h_forecast": 0.35,
            "rain_now": 0.20,
            "soil_saturation": 0.20,
            "river_level": 0.15,
            "drainage_load": 0.10,
        })
        self.cutoffs = self.flood_cfg.get("cutoffs", {
            "low": 0.30,
            "watch": 0.50,
            "warning": 0.70,
        })
        self.max_ages = self.config.get("max_age_minutes", {
            "weather_forecast": 360,
            "weather_observation": 60,
            "hydro_observation": 60,
        })
        self.min_sources_critical = self.config.get("source_agreement", {}).get(
            "min_sources_for_critical", 2
        )

    def evaluate(
        self,
        events: Sequence[Union[Dict[str, Any], Any]],
        now_ts: Optional[float] = None
    ) -> HazardAssessment:
        """
        Evaluates a collection of recent events (weather_forecast, weather_observation, hydro_observation).
        """
        now = now_ts or time.time()
        evidence: List[str] = []
        drivers: List[RiskDriver] = []
        sources = set()

        if not events:
            return HazardAssessment(
                hazard="flood",
                risk_level="low",
                risk_score=0.0,
                confidence=0.5,
                drivers=[],
                evidence=["Insufficient data: No hydrological or weather events provided."],
            )

        # Extracted parameters with their source timestamps
        rain_24h_val: Optional[float] = None
        rain_now_val: Optional[float] = None
        soil_sat_val: Optional[float] = None
        river_ratio_val: Optional[float] = None
        drainage_val: Optional[float] = None

        center_lat: Optional[float] = None
        center_lon: Optional[float] = None

        for raw in events:
            ev = raw if isinstance(raw, dict) else (raw.model_dump() if hasattr(raw, "model_dump") else raw.__dict__)
            kind = ev.get("kind", "")
            ts = float(ev.get("ts", now))
            source = ev.get("source", "unknown")
            data = ev.get("data", {})
            lat = ev.get("lat")
            lon = ev.get("lon")

            if lat is not None and lon is not None and center_lat is None:
                center_lat, center_lon = float(lat), float(lon)

            # Check age
            max_age_min = self.max_ages.get(kind, 120)
            age_min = (now - ts) / 60.0
            if age_min > max_age_min:
                evidence.append(
                    f"Ignored stale event {kind} from {source} (age {age_min:.1f}m > max {max_age_min}m)."
                )
                continue

            # Record independent source
            sources.add(source)

            norm_data = normalize_units(data, evidence)

            if kind == "weather_forecast":
                if "rain_24h_forecast_mm" in norm_data:
                    rain_24h_val = float(norm_data["rain_24h_forecast_mm"])
                elif "rain_6h_forecast_mm" in norm_data:
                    # Scale 6h forecast to 24h approximation
                    rain_24h_val = float(norm_data["rain_6h_forecast_mm"]) * 2.0
                    evidence.append("Extrapolated rain_6h_forecast_mm to 24h baseline.")

            elif kind == "weather_observation":
                if "rain_rate_mm_h" in norm_data:
                    rain_now_val = float(norm_data["rain_rate_mm_h"])
                if "soil_saturation" in norm_data:
                    soil_sat_val = min(1.0, max(0.0, float(norm_data["soil_saturation"])))
                if "rain_24h_mm" in norm_data and rain_24h_val is None:
                    rain_24h_val = float(norm_data["rain_24h_mm"])

            elif kind == "hydro_observation":
                river_m = norm_data.get("river_level_m")
                stage_m = norm_data.get("flood_stage_m")
                if river_m is not None and stage_m is not None and stage_m > 0:
                    river_ratio_val = float(river_m) / float(stage_m)
                if "drainage_load" in norm_data:
                    drainage_val = min(1.0, max(0.0, float(norm_data["drainage_load"])))

        # Factor definitions: (name, normalized_value_0_1, raw_value, configured_weight)
        rain_24h_norm = self.flood_cfg.get("rain_24h_norm_mm", 100.0)
        rain_now_norm = self.flood_cfg.get("rain_now_norm_mm_h", 30.0)

        factor_candidates = [
            (
                "rain_24h_forecast_mm",
                min(rain_24h_val / rain_24h_norm, 1.0) if rain_24h_val is not None else None,
                rain_24h_val,
                self.weights.get("rain_24h_forecast", 0.35),
            ),
            (
                "rain_now_mm_h",
                min(rain_now_val / rain_now_norm, 1.0) if rain_now_val is not None else None,
                rain_now_val,
                self.weights.get("rain_now", 0.20),
            ),
            (
                "soil_saturation",
                soil_sat_val,
                soil_sat_val,
                self.weights.get("soil_saturation", 0.20),
            ),
            (
                "river_level_ratio",
                min(river_ratio_val, 2.0) / 1.0 if river_ratio_val is not None else None,  # >= 1.0 means at or above flood stage
                river_ratio_val,
                self.weights.get("river_level", 0.15),
            ),
            (
                "drainage_load",
                drainage_val,
                drainage_val,
                self.weights.get("drainage_load", 0.10),
            ),
        ]

        active_factors = [fc for fc in factor_candidates if fc[1] is not None]

        if not active_factors:
            return HazardAssessment(
                hazard="flood",
                risk_level="low",
                risk_score=0.0,
                confidence=0.4,
                drivers=[],
                evidence=evidence + ["No valid or recent hydrological factors extracted."],
            )

        sum_active_weights = sum(fc[3] for fc in active_factors)
        total_possible_weight = sum(fc[3] for fc in factor_candidates)

        # Weight redistribution for missing optional inputs
        score = 0.0
        for name, norm_v, raw_v, w in active_factors:
            effective_weight = w / sum_active_weights
            contribution = round(effective_weight * norm_v, 3)
            score += contribution
            drivers.append(RiskDriver(factor=name, value=round(raw_v, 2), contribution=contribution))

        risk_score = min(1.0, max(0.0, round(score, 2)))

        # Confidence lowers if many inputs are missing
        base_confidence = 0.95 * (sum_active_weights / max(total_possible_weight, 0.01))
        confidence = round(max(0.40, min(0.95, base_confidence)), 2)

        # Level determination
        if risk_score >= self.cutoffs.get("warning", 0.70):
            risk_level = "critical"
        elif risk_score >= self.cutoffs.get("watch", 0.50):
            risk_level = "warning"
        elif risk_score >= self.cutoffs.get("low", 0.30):
            risk_level = "watch"
        else:
            risk_level = "low"

        evidence.append(
            f"Flood composite risk index: {risk_score:.2f} -> {risk_level.upper()} "
            f"(active factors: {len(active_factors)}/{len(factor_candidates)}, confidence: {confidence})"
        )

        affected_area = None
        if center_lat is not None and center_lon is not None:
            affected_area = AffectedArea(
                center=GeoPoint(lat=center_lat, lon=center_lon),
                radius_km=15.0,
            )

        assessment = HazardAssessment(
            hazard="flood",
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=confidence,
            drivers=drivers,
            affected_area=affected_area,
            eta_minutes=120 if risk_level in {"warning", "critical"} else None,
            independent_source_count=len(sources),
            evidence=evidence,
        )

        return apply_critical_source_cap(assessment, self.min_sources_critical)


if __name__ == "__main__":
    print("Running FloodEngine self-test...")
    engine = FloodEngine()

    test_events = [
        {
            "kind": "weather_forecast",
            "source": "open_meteo_forecast",
            "ts": time.time(),
            "lat": 36.80,
            "lon": 10.18,
            "data": {"rain_24h_forecast_mm": 95.0},
        },
        {
            "kind": "weather_observation",
            "source": "iot_weather_station",
            "ts": time.time(),
            "lat": 36.80,
            "lon": 10.18,
            "data": {"rain_rate_mm_h": 28.0, "soil_saturation": 0.90},
        },
        {
            "kind": "hydro_observation",
            "source": "river_gauge_network",
            "ts": time.time(),
            "lat": 36.80,
            "lon": 10.18,
            "data": {"river_level_m": 4.8, "flood_stage_m": 4.5, "drainage_load": 0.85},
        },
    ]

    res = engine.evaluate(test_events)
    print(f"Risk Score: {res.risk_score} | Level: {res.risk_level} | Sources: {res.independent_source_count}")
    for d in res.drivers:
        print(f" - {d.factor}: {d.value} (contribution: {d.contribution})")
    assert res.risk_level == "critical", "Expected critical flood"
    print("FloodEngine self-test passed successfully.")

