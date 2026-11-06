"""
Deterministic Tsunami Hazard Assessment Engine.
Evaluates tsunami potential conditional on offshore earthquake catalog parameters
and verifies sea-level gauge anomaly telemetry.
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
    haversine_distance_km,
    load_hazard_config,
)


class TsunamiEngine:
    """
    Assesses tsunami propagation risk following an offshore undersea earthquake.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_hazard_config()
        self.tsunami_cfg = self.config.get("tsunami", {})
        self.city_center = self.config.get("city_center", {"lat": 36.8065, "lon": 10.1815})
        self.min_magnitude = self.tsunami_cfg.get("min_magnitude", 6.5)
        self.max_depth_km = self.tsunami_cfg.get("max_depth_km", 70.0)
        self.wave_speed_kmh = self.tsunami_cfg.get("wave_speed_kmh", 700.0)
        self.sea_level_thresh_m = self.tsunami_cfg.get("sea_level_change_threshold_m", 0.30)
        self.sea_level_window_min = self.tsunami_cfg.get("sea_level_window_min", 30)
        self.region_bbox = self.tsunami_cfg.get("region_bbox", {
            "lat_min": 30.0,
            "lat_max": 46.0,
            "lon_min": -6.0,
            "lon_max": 37.0,
        })
        self.min_sources_critical = self.config.get("source_agreement", {}).get(
            "min_sources_for_critical", 2
        )

    def evaluate(
        self,
        quake_assessment: HazardAssessment,
        sea_level_events: Optional[Sequence[Union[Dict[str, Any], Any]]] = None,
        now_ts: Optional[float] = None
    ) -> HazardAssessment:
        """
        Evaluates tsunami risk given an earthquake assessment and optional sea-level gauge events.
        """
        now = now_ts or time.time()
        evidence: List[str] = []
        drivers: List[RiskDriver] = []
        sources = set()

        meta = quake_assessment.metadata or {}
        mag = meta.get("magnitude")
        depth_km = meta.get("depth_km")
        epi_lat = meta.get("epicenter_lat")
        epi_lon = meta.get("epicenter_lon")
        offshore = meta.get("offshore", False)
        catalog_src = meta.get("catalog_source", "seismic_catalog")

        if mag is None or epi_lat is None or epi_lon is None:
            return HazardAssessment(
                hazard="tsunami",
                risk_level="low",
                risk_score=0.0,
                confidence=0.5,
                drivers=[],
                evidence=["Tsunami evaluation skipped: Incomplete earthquake metadata."],
            )

        sources.add(catalog_src)

        # 1. Geographic region bounding check
        in_region = (
            self.region_bbox["lat_min"] <= epi_lat <= self.region_bbox["lat_max"]
            and self.region_bbox["lon_min"] <= epi_lon <= self.region_bbox["lon_max"]
        )

        # 2. Candidate criteria check
        is_candidate = (
            offshore
            and mag >= self.min_magnitude
            and depth_km <= self.max_depth_km
            and in_region
        )

        city_lat = float(self.city_center.get("lat", 36.8065))
        city_lon = float(self.city_center.get("lon", 10.1815))
        dist_to_coast_km = haversine_distance_km(city_lat, city_lon, epi_lat, epi_lon)

        # ETA calculation in minutes
        speed = max(100.0, float(self.wave_speed_kmh))
        eta_minutes = int(round((dist_to_coast_km / speed) * 60.0))

        drivers.append(RiskDriver(factor="earthquake_magnitude", value=mag, contribution=0.4 if is_candidate else 0.05))
        drivers.append(RiskDriver(factor="depth_km", value=depth_km, contribution=0.2 if is_candidate else 0.0))
        drivers.append(RiskDriver(factor="offshore", value=bool(offshore), contribution=0.2 if offshore else 0.0))

        if not is_candidate:
            reasons = []
            if not offshore:
                reasons.append("epicenter is onshore")
            if mag < self.min_magnitude:
                reasons.append(f"magnitude M{mag} < {self.min_magnitude}")
            if depth_km > self.max_depth_km:
                reasons.append(f"hypocenter depth {depth_km}km > {self.max_depth_km}km")
            if not in_region:
                reasons.append("epicenter outside monitored marine region")

            evidence.append(f"Tsunami criteria not met: {', '.join(reasons)}.")
            return HazardAssessment(
                hazard="tsunami",
                risk_level="low",
                risk_score=0.10,
                confidence=0.90,
                drivers=drivers,
                evidence=evidence,
                metadata={"candidate": False, "dist_to_coast_km": round(dist_to_coast_km, 1)},
            )

        evidence.append(
            f"Tsunami candidate confirmed: Offshore M{mag:.1f} shallow ({depth_km:.1f}km) earthquake "
            f"in Mediterranean basin. Dist={dist_to_coast_km:.1f}km -> ETA approx {eta_minutes} min."
        )

        # 3. Check sea-level observation telemetry
        sea_level_change_detected = False
        max_delta_m = 0.0

        if sea_level_events:
            for raw in sea_level_events:
                ev = raw if isinstance(raw, dict) else (raw.model_dump() if hasattr(raw, "model_dump") else raw.__dict__)
                if ev.get("kind") != "sea_level_observation":
                    continue
                ts = float(ev.get("ts", now))
                if (now - ts) / 60.0 > self.sea_level_window_min:
                    continue

                d = ev.get("data", {})
                chg = d.get("change_m_per_min")
                sl = d.get("sea_level_m")
                base = d.get("baseline_m", 0.0)

                delta = abs(chg) if chg is not None else (abs(sl - base) if sl is not None else 0.0)
                if delta > max_delta_m:
                    max_delta_m = delta

                if delta >= self.sea_level_thresh_m:
                    sea_level_change_detected = True
                    sources.add(ev.get("source", "tide_gauge"))
                    evidence.append(
                        f"Tide gauge '{d.get('station_id', 'unknown')}' recorded sea-level anomaly: "
                        f"{delta:.2f}m >= threshold {self.sea_level_thresh_m:.2f}m."
                    )

        if sea_level_change_detected:
            drivers.append(RiskDriver(factor="sea_level_anomaly_m", value=round(max_delta_m, 2), contribution=0.35))
            risk_score = 0.95
            risk_level = "critical"
            confidence = 0.95
        else:
            risk_score = 0.65
            risk_level = "warning"
            confidence = 0.80
            evidence.append(
                f"Candidate status active without sea-level gauge confirmation: "
                f"maximum recorded anomaly {max_delta_m:.2f}m (below {self.sea_level_thresh_m:.2f}m)."
            )

        assessment = HazardAssessment(
            hazard="tsunami",
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=confidence,
            drivers=drivers,
            affected_area=AffectedArea(
                center=GeoPoint(lat=city_lat, lon=city_lon),
                radius_km=30.0,
            ),
            eta_minutes=eta_minutes,
            independent_source_count=len(sources),
            needs_human_confirmation=True,  # Tsunami warning/critical always requires human review
            evidence=evidence,
            metadata={
                "candidate": True,
                "dist_to_coast_km": round(dist_to_coast_km, 1),
                "sea_level_confirmed": sea_level_change_detected,
                "max_sea_level_anomaly_m": round(max_delta_m, 2),
            },
        )

        return apply_critical_source_cap(assessment, self.min_sources_critical)


if __name__ == "__main__":
    print("Running TsunamiEngine self-test...")
    from .hazard_common import GeoPoint

    engine = TsunamiEngine()
    fake_quake_assessment = HazardAssessment(
        hazard="earthquake",
        risk_level="critical",
        risk_score=0.88,
        confidence=0.90,
        metadata={
            "magnitude": 7.1,
            "depth_km": 15.0,
            "epicenter_lat": 37.80,
            "epicenter_lon": 10.50,
            "offshore": True,
            "catalog_source": "emsc",
        },
    )

    # 1. Candidate warning without tide gauge
    res_warn = engine.evaluate(fake_quake_assessment)
    print(f"Candidate Level: {res_warn.risk_level} (Score: {res_warn.risk_score}, ETA: {res_warn.eta_minutes}m)")
    assert res_warn.risk_level == "warning"

    # 2. Critical with tide gauge
    tide_event = {
        "kind": "sea_level_observation",
        "source": "la_goulette_tide_gauge",
        "ts": time.time(),
        "data": {
            "station_id": "TUN-TG-01",
            "sea_level_m": 1.45,
            "baseline_m": 0.80,
            "change_m_per_min": 0.40,
        },
    }
    res_crit = engine.evaluate(fake_quake_assessment, [tide_event])
    print(f"Confirmed Level: {res_crit.risk_level} (Score: {res_crit.risk_score}, Sources: {res_crit.independent_source_count})")
    assert res_crit.risk_level == "critical"
    assert res_crit.needs_human_confirmation is True
    print("TsunamiEngine self-test passed successfully.")

