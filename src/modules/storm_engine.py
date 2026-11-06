"""
Deterministic Storm and Severe Weather Hazard Assessment Engine.
Short-term nowcasting and incident assessment for convective storms, severe windstorms,
downbursts, and tornadic rotation.
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


class StormEngine:
    """
    Evaluates meteorological telemetry and official meteorological alerts
    to determine severe convective or cyclonic storm risk.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_hazard_config()
        self.storm_cfg = self.config.get("storm", {})
        self.gust_warning_kmh = self.storm_cfg.get("gust_warning_kmh", 75.0)
        self.gust_critical_kmh = self.storm_cfg.get("gust_critical_kmh", 105.0)
        self.pressure_drop_thresh = self.storm_cfg.get("pressure_drop_hpa_3h", 5.0)
        self.rain_rate_warning = self.storm_cfg.get("rain_rate_warning_mm_h", 25.0)
        self.rain_rate_critical = self.storm_cfg.get("rain_rate_critical_mm_h", 50.0)
        self.lightning_rate_crit = self.storm_cfg.get("lightning_rate_critical", 30.0)
        self.min_sources_critical = self.config.get("source_agreement", {}).get(
            "min_sources_for_critical", 2
        )

    def evaluate(
        self,
        events: Sequence[Union[Dict[str, Any], Any]],
        now_ts: Optional[float] = None
    ) -> HazardAssessment:
        """
        Evaluates a set of recent weather_observation and official_weather_alert events.
        """
        now = now_ts or time.time()
        evidence: List[str] = []
        drivers: List[RiskDriver] = []
        sources = set()

        if not events:
            return HazardAssessment(
                hazard="storm",
                risk_level="low",
                risk_score=0.0,
                confidence=0.5,
                drivers=[],
                evidence=["No meteorological events provided for storm assessment."],
            )

        max_gust = 0.0
        max_wind = 0.0
        max_pressure_drop = 0.0
        max_rain_rate = 0.0
        radar_rotation = False
        hail_detected = False
        max_lightning_rate = 0.0
        official_alert_level: Optional[str] = None
        official_alert_type: Optional[str] = None

        center_lat: Optional[float] = None
        center_lon: Optional[float] = None

        for raw in events:
            ev = raw if isinstance(raw, dict) else (raw.model_dump() if hasattr(raw, "model_dump") else raw.__dict__)
            kind = ev.get("kind", "")
            source = ev.get("source", "weather_station")
            data = ev.get("data", {})
            lat = ev.get("lat")
            lon = ev.get("lon")

            if lat is not None and lon is not None and center_lat is None:
                center_lat, center_lon = float(lat), float(lon)

            sources.add(source)
            norm_data = normalize_units(data, evidence)

            if kind in {"weather_observation", "weather_forecast"}:
                gust = norm_data.get("gust_kmh") or norm_data.get("gust_forecast_kmh")
                if gust is not None and float(gust) > max_gust:
                    max_gust = float(gust)

                wind = norm_data.get("wind_speed_kmh")
                if wind is not None and float(wind) > max_wind:
                    max_wind = float(wind)

                p_drop = norm_data.get("pressure_drop_hpa_3h")
                if p_drop is not None and float(p_drop) > max_pressure_drop:
                    max_pressure_drop = float(p_drop)

                r_rate = norm_data.get("rain_rate_mm_h")
                if r_rate is not None and float(r_rate) > max_rain_rate:
                    max_rain_rate = float(r_rate)

                if norm_data.get("radar_rotation"):
                    radar_rotation = True

                if norm_data.get("hail"):
                    hail_detected = True

                l_rate = norm_data.get("lightning_rate")
                if l_rate is not None and float(l_rate) > max_lightning_rate:
                    max_lightning_rate = float(l_rate)

            elif kind == "official_weather_alert":
                al_type = str(norm_data.get("alert_type", "")).lower()
                al_level = str(norm_data.get("alert_level", "")).lower()
                issuer = norm_data.get("issuer", "National Met Service")
                official_alert_type = al_type
                official_alert_level = al_level
                evidence.append(
                    f"Official weather alert active from {issuer}: type='{al_type}', level='{al_level}'."
                )

        # Evaluate risk conditions
        is_critical_trigger = False
        is_warning_trigger = False

        score = 0.10

        # Radar rotation -> immediate critical severe convective / tornado threat
        if radar_rotation:
            is_critical_trigger = True
            score = max(score, 0.95)
            drivers.append(RiskDriver(factor="radar_rotation", value=True, contribution=0.50))
            evidence.append("Doppler radar indicates mesocyclonic rotation signature (Tornado / Severe Vortex).")

        # Official severe alert
        if official_alert_level in {"red", "critical", "extreme"} or official_alert_type in {"tornado", "severe_storm"}:
            is_critical_trigger = True
            score = max(score, 0.90)
            drivers.append(RiskDriver(factor="official_alert", value=f"{official_alert_type}:{official_alert_level}", contribution=0.45))
        elif official_alert_level in {"orange", "warning"}:
            is_warning_trigger = True
            score = max(score, 0.65)
            drivers.append(RiskDriver(factor="official_alert", value=f"{official_alert_type}:{official_alert_level}", contribution=0.30))

        # Gust speeds
        if max_gust >= self.gust_critical_kmh:
            is_critical_trigger = True
            score = max(score, 0.88)
            drivers.append(RiskDriver(factor="gust_kmh", value=round(max_gust, 1), contribution=0.40))
            evidence.append(f"Destructive gusts recorded: {max_gust:.1f} km/h >= {self.gust_critical_kmh} km/h.")
        elif max_gust >= self.gust_warning_kmh:
            is_warning_trigger = True
            score = max(score, 0.60)
            drivers.append(RiskDriver(factor="gust_kmh", value=round(max_gust, 1), contribution=0.30))
            evidence.append(f"Gale/storm gusts recorded: {max_gust:.1f} km/h >= {self.gust_warning_kmh} km/h.")

        # Pressure drop
        if max_pressure_drop >= self.pressure_drop_thresh:
            is_warning_trigger = True
            score = max(score, 0.55)
            drivers.append(RiskDriver(factor="pressure_drop_hpa_3h", value=round(max_pressure_drop, 1), contribution=0.25))
            evidence.append(f"Rapid barometric pressure drop: {max_pressure_drop:.1f} hPa/3h (cyclonic deepening).")

        # Rain rate
        if max_rain_rate >= self.rain_rate_critical:
            score = min(1.0, score + 0.15)
            drivers.append(RiskDriver(factor="rain_rate_mm_h", value=round(max_rain_rate, 1), contribution=0.20))
            evidence.append(f"Violent rain rate: {max_rain_rate:.1f} mm/h.")
        elif max_rain_rate >= self.rain_rate_warning:
            score = min(1.0, score + 0.10)
            drivers.append(RiskDriver(factor="rain_rate_mm_h", value=round(max_rain_rate, 1), contribution=0.10))

        # Hail & Lightning
        if hail_detected:
            score = min(1.0, score + 0.10)
            drivers.append(RiskDriver(factor="hail", value=True, contribution=0.10))
            evidence.append("Severe hail detected by surface sensors.")
        if max_lightning_rate >= self.lightning_rate_crit:
            score = min(1.0, score + 0.10)
            drivers.append(RiskDriver(factor="lightning_rate", value=round(max_lightning_rate, 1), contribution=0.10))
            evidence.append(f"Intense electrical storm activity: {max_lightning_rate:.1f} strikes/min.")

        risk_score = round(min(1.0, max(0.05, score)), 2)

        if is_critical_trigger or risk_score >= 0.75:
            risk_level = "critical"
        elif is_warning_trigger or risk_score >= 0.50:
            risk_level = "warning"
        elif risk_score >= 0.30:
            risk_level = "watch"
        else:
            risk_level = "low"

        affected_area = None
        if center_lat is not None and center_lon is not None:
            affected_area = AffectedArea(
                center=GeoPoint(lat=center_lat, lon=center_lon),
                radius_km=25.0,
            )

        assessment = HazardAssessment(
            hazard="storm",
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=0.88,
            drivers=drivers,
            affected_area=affected_area,
            eta_minutes=60 if risk_level in {"warning", "critical"} else None,
            independent_source_count=len(sources),
            evidence=evidence,
            metadata={
                "max_gust_kmh": max_gust,
                "radar_rotation": radar_rotation,
                "hail": hail_detected,
                "pressure_drop_hpa_3h": max_pressure_drop,
            },
        )

        return apply_critical_source_cap(assessment, self.min_sources_critical)


if __name__ == "__main__":
    print("Running StormEngine self-test...")
    engine = StormEngine()

    test_storm = [
        {
            "kind": "weather_observation",
            "source": "station_north",
            "data": {"gust_kmh": 112.0, "pressure_drop_hpa_3h": 6.2, "radar_rotation": True},
        },
        {
            "kind": "official_weather_alert",
            "source": "national_met_service",
            "data": {"alert_type": "severe_storm", "alert_level": "red"},
        },
    ]

    res = engine.evaluate(test_storm)
    print(f"Storm Risk Score: {res.risk_score} | Level: {res.risk_level} | Sources: {res.independent_source_count}")
    assert res.risk_level == "critical", "Expected critical storm"
    print("StormEngine self-test passed successfully.")

