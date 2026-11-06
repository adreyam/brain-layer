"""
Deterministic Catalog Earthquake Assessment Engine.
Processes catalog earthquake telemetry (USGS/EMSC) with magnitude, depth, and epicenter,
evaluates impact on the metropolitan area, and feeds parameters to the TsunamiEngine.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from .hazard_common import (
    AffectedArea,
    GeoPoint,
    HazardAssessment,
    RiskDriver,
    apply_critical_source_cap,
    haversine_distance_km,
    load_hazard_config,
)


def is_mediterranean_offshore(lat: float, lon: float) -> bool:
    """
    Heuristic check to determine if epicenter is offshore in the Mediterranean/sea basin.
    Focuses on central Mediterranean waters off the North African / southern European coasts.
    """
    # General Mediterranean marine bounding box
    if not (30.0 <= lat <= 46.0 and -6.0 <= lon <= 37.0):
        return False

    # Water basin bounding approximation (e.g. Strait of Sicily, Ionian, Tyrrhenian, Aegean, Gulf of Tunis)
    # Marine zone off northern Tunisia / central Med: Lat 36.9 - 38.5, Lon 8.5 - 12.0
    if 36.85 <= lat <= 38.8 and 9.5 <= lon <= 12.5:
        return True
    # Wider Mediterranean marine zones
    if 33.0 <= lat <= 44.0 and 10.0 <= lon <= 25.0:
        return True
    return False


class CatalogQuakeEngine:
    """
    Evaluates seismic catalog events from USGS, EMSC, or regional seismic networks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_hazard_config()
        self.eq_cfg = self.config.get("earthquake", {})
        self.city_center = self.config.get("city_center", {"lat": 36.8065, "lon": 10.1815})
        self.min_magnitude = self.eq_cfg.get("min_magnitude", 4.5)
        self.max_radius_km = self.eq_cfg.get("radius_km", 350.0)
        self.shallow_depth_km = self.eq_cfg.get("shallow_depth_km", 30.0)
        self.mag_thresholds = self.eq_cfg.get(
            "magnitude_thresholds", {"watch": 4.5, "warning": 5.5, "critical": 6.5}
        )
        self.min_sources_critical = self.config.get("source_agreement", {}).get(
            "min_sources_for_critical", 2
        )

    def evaluate(
        self,
        event: Union[Dict[str, Any], Any],
        plum_active: bool = False,
        plum_evidence: Optional[List[str]] = None,
        now_ts: Optional[float] = None
    ) -> HazardAssessment:
        """
        Evaluates a single seismic catalog event.
        `plum_active` indicates whether local seismic stations registered concurrent ground motion.
        """
        now = now_ts or time.time()
        ev = event if isinstance(event, dict) else (event.model_dump() if hasattr(event, "model_dump") else event.__dict__)
        data = ev.get("data", {})
        evidence: List[str] = []
        drivers: List[RiskDriver] = []

        # Extract core catalog parameters
        mag = data.get("magnitude")
        if mag is None:
            mag = ev.get("magnitude")

        depth_km = data.get("depth_km")
        if depth_km is None:
            depth_km = ev.get("depth_km", 10.0)

        epi_lat = data.get("epicenter_lat")
        if epi_lat is None:
            epi_lat = ev.get("lat")

        epi_lon = data.get("epicenter_lon")
        if epi_lon is None:
            epi_lon = ev.get("lon")

        catalog_source = data.get("catalog_source") or ev.get("source", "seismic_catalog")

        if mag is None or epi_lat is None or epi_lon is None:
            return HazardAssessment(
                hazard="earthquake",
                risk_level="low",
                risk_score=0.0,
                confidence=0.3,
                drivers=[],
                evidence=["Rejected: Missing essential seismic catalog fields (magnitude, lat, or lon)."],
            )

        mag = float(mag)
        depth_km = float(depth_km)
        epi_lat = float(epi_lat)
        epi_lon = float(epi_lon)

        # Distance to city center
        city_lat = float(self.city_center.get("lat", 36.8065))
        city_lon = float(self.city_center.get("lon", 10.1815))
        dist_km = haversine_distance_km(city_lat, city_lon, epi_lat, epi_lon)

        evidence.append(
            f"Catalog quake: M{mag:.1f}, depth={depth_km:.1f}km, "
            f"epicenter=({epi_lat:.3f}, {epi_lon:.3f}), distance={dist_km:.1f}km from city center."
        )

        # Check offshore status
        offshore = data.get("offshore")
        if offshore is None:
            offshore = is_mediterranean_offshore(epi_lat, epi_lon)
            evidence.append(f"Derived offshore flag via Mediterranean marine boundaries: offshore={offshore}.")

        # If magnitude is below cutoff or distance too far
        if mag < self.min_magnitude:
            return HazardAssessment(
                hazard="earthquake",
                risk_level="low",
                risk_score=round(max(0.0, (mag / 10.0)), 2),
                confidence=0.8,
                drivers=[RiskDriver(factor="magnitude", value=mag, contribution=round(mag / 10.0, 2))],
                evidence=evidence + [f"Magnitude M{mag:.1f} below minimum monitoring threshold ({self.min_magnitude})."],
                metadata={
                    "magnitude": mag,
                    "depth_km": depth_km,
                    "epicenter_lat": epi_lat,
                    "epicenter_lon": epi_lon,
                    "distance_km": round(dist_km, 1),
                    "offshore": bool(offshore),
                },
            )

        if dist_km > self.max_radius_km:
            return HazardAssessment(
                hazard="earthquake",
                risk_level="low",
                risk_score=0.1,
                confidence=0.85,
                drivers=[RiskDriver(factor="distance_km", value=round(dist_km, 1), contribution=0.1)],
                evidence=evidence + [f"Epicenter distance ({dist_km:.1f}km) exceeds monitoring radius ({self.max_radius_km}km)."],
                metadata={
                    "magnitude": mag,
                    "depth_km": depth_km,
                    "epicenter_lat": epi_lat,
                    "epicenter_lon": epi_lon,
                    "distance_km": round(dist_km, 1),
                    "offshore": bool(offshore),
                },
            )

        # Base scoring from magnitude
        # Cutoffs: watch: 4.5, warning: 5.5, critical: 6.5
        crit_mag = self.mag_thresholds.get("critical", 6.5)
        warn_mag = self.mag_thresholds.get("warning", 5.5)
        watch_mag = self.mag_thresholds.get("watch", 4.5)

        if mag >= crit_mag:
            base_score = 0.85
            base_level = "critical"
        elif mag >= warn_mag:
            base_score = 0.65
            base_level = "warning"
        elif mag >= watch_mag:
            base_score = 0.40
            base_level = "watch"
        else:
            base_score = 0.20
            base_level = "low"

        # Distance attenuation adjustment: closer earthquakes exert significantly higher PGA
        dist_factor = max(0.0, 1.0 - (dist_km / self.max_radius_km))
        depth_factor = 0.15 if depth_km <= self.shallow_depth_km else -0.10

        composite_score = base_score + (0.15 * dist_factor) + depth_factor
        risk_score = min(1.0, max(0.1, round(composite_score, 2)))

        drivers.append(RiskDriver(factor="magnitude", value=mag, contribution=round(base_score, 2)))
        drivers.append(RiskDriver(factor="distance_km", value=round(dist_km, 1), contribution=round(0.15 * dist_factor, 2)))
        drivers.append(RiskDriver(factor="depth_km", value=round(depth_km, 1), contribution=round(depth_factor, 2)))

        # Determine level based on adjusted score
        if risk_score >= 0.70 and mag >= warn_mag:
            level = "critical"
        elif risk_score >= 0.50:
            level = "warning"
        elif risk_score >= 0.30:
            level = "watch"
        else:
            level = "low"

        # Independent sources
        sources = {catalog_source}
        confidence = 0.85
        if plum_active:
            sources.add("plum_sensor_network")
            confidence = 0.95
            evidence.append("Concurrent PLUM sensor ground-motion confirmation detected (+1 independent source).")
            if plum_evidence:
                evidence.extend(plum_evidence)

        assessment = HazardAssessment(
            hazard="earthquake",
            risk_level=level,
            risk_score=risk_score,
            confidence=confidence,
            drivers=drivers,
            affected_area=AffectedArea(
                center=GeoPoint(lat=city_lat, lon=city_lon),
                radius_km=max(20.0, min(150.0, (mag - 4.0) * 40.0)),
            ),
            eta_minutes=None,  # Earthquakes cannot be given ETA
            independent_source_count=len(sources),
            evidence=evidence,
            metadata={
                "magnitude": mag,
                "depth_km": depth_km,
                "epicenter_lat": epi_lat,
                "epicenter_lon": epi_lon,
                "distance_km": round(dist_km, 1),
                "offshore": bool(offshore),
                "catalog_source": catalog_source,
            },
        )

        return apply_critical_source_cap(assessment, self.min_sources_critical)


if __name__ == "__main__":
    print("Running CatalogQuakeEngine self-test...")
    engine = CatalogQuakeEngine()

    test_quake = {
        "kind": "seismic_catalog",
        "source": "usgs_feed",
        "data": {
            "magnitude": 6.8,
            "depth_km": 12.0,
            "epicenter_lat": 37.10,
            "epicenter_lon": 10.30,
            "offshore": True,
            "catalog_source": "usgs",
        },
    }

    res = engine.evaluate(test_quake, plum_active=True)
    print(f"Quake Risk Score: {res.risk_score} | Level: {res.risk_level} | Sources: {res.independent_source_count}")
    assert res.risk_level == "critical", "Expected critical earthquake"
    print("CatalogQuakeEngine self-test passed successfully.")

