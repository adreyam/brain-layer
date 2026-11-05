"""
Emergency Axis Agent.

Serves as the operational dispatch layer for the Emergency axis. It integrates
standalone physics/analytic modules (PLUM Engine & Crime Hotspot Engine) to construct
standardized dispatch orders and CAP alerts.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from .plum_engine import PLUMEngine, SensorReading, Station, StationData, WaveformTelemetry, station_intensity
from .crime_hotspot import CrimeHotspotEngine, RiskTerrainContext, RawCrimeEvent, IncidentContext


class ActionItem(BaseModel):
    type: str          # "dispatch_units", "alert", "corridor_clear", "structural_check"
    resource: str      # "units:nearest", "channel:cap", "road:corridor"
    detail: str
    priority: int = 1  # 0 = Critical / Immediate
    # Units requested by a dispatch_units order, e.g. ["fire_engine", "fire_engine", "ambulance"].
    # Known types: fire_engine, ambulance, police, rescue. The dashboard simulation reads this list.
    units: List[str] = Field(default_factory=list)


class EmergencyAgentResponse(BaseModel):
    axis: str = "emergency"
    severity: int
    confidence: float
    status: str = "ok"
    plumAlertTriggered: bool = False
    peakMMI: float = 0.0
    plumMode: Optional[str] = None                 # "network" (full PLUM) or "single_station"
    plumDetections: List[Dict[str, Any]] = Field(default_factory=list)
    plumSuppressed: List[Dict[str, Any]] = Field(default_factory=list)
    stationPeakMMI: Dict[str, float] = Field(default_factory=dict)
    actions: List[ActionItem]
    evidence: List[str]


class EmergencyAgent:
    """Operational agent coordinating response logic and underlying domain engines."""

    def __init__(self, plum_engine: Optional[PLUMEngine] = None, crime_engine: Optional[CrimeHotspotEngine] = None):
        self.plum = plum_engine or PLUMEngine()
        self.crime = crime_engine or CrimeHotspotEngine()

    def run_plum_network(self, network: dict) -> dict:
        """Run the full PLUM engine on a station network.

        network = {"samplingRateHz": 40, "stations": [{"sensor_id", "lat", "lon",
                   "accel": {"vertical_z": [...], "east_west_e": [...], "north_south_n": [...]}}]}
        Accelerations in m/s^2, all stations aligned on the same start time.
        """
        fs = float(network.get("samplingRateHz", 100))
        data = [
            StationData(
                station=Station(s["sensor_id"], float(s["lat"]), float(s["lon"])),
                sample_rate_hz=fs,
                accel={k: np.asarray(v, dtype=float) for k, v in s["accel"].items()},
            )
            for s in network["stations"]
        ]
        result = self.plum.run(data)
        peaks = {d.station.sensor_id: round(float(np.nanmax(station_intensity(d))), 2) for d in data}
        peak_sid = max(peaks, key=peaks.get)
        evidence = [f"PLUM network: {len(data)} stations, {fs:.0f} Hz, peak MMI {peaks[peak_sid]} at {peak_sid}"]
        for det in result.detections:
            evidence.extend(det.evidence)
            evidence.append(f"Event declared at t={det.detection_time_s:.0f} s")
        for sup in result.suppressed:
            evidence.append(f"Suppressed trigger at {sup.station} (MMI {sup.trigger_mmi:.2f}, t={sup.trigger_second} s): {sup.reason}")
        if not result.detections and not result.suppressed:
            evidence.append("No station reached the primary threshold (MMI 4.0)")
        return {
            "alert": bool(result.detections),
            "peak_mmi": peaks[peak_sid],
            "evidence": evidence,
            "report": {
                "plumMode": "network",
                "plumDetections": [asdict(d) for d in result.detections],
                "plumSuppressed": [asdict(s) for s in result.suppressed],
                "stationPeakMMI": peaks,
            },
        }

    def process_event(self, event_dict: dict) -> dict:
        """Main entry point compatible with LangGraph Orchestrator payload schema."""
        actions: List[ActionItem] = []
        evidence: List[str] = []
        severity = 1
        confidence = 0.8
        plum_alert = False
        peak_mmi = 0.0
        plum_extra: Dict[str, Any] = {}

        kind = event_dict.get("kind", "")
        data = event_dict.get("data", {})
        source = event_dict.get("source", "unknown")

        # 1. Seismic Telemetry Path (PLUM Engine)
        if "stationNetwork" in data or "rawWaveformTelemetry" in data or kind == "seismic_sensor":
            try:
                if "stationNetwork" in data:
                    # Full PLUM (Kilb et al. 2021): per-station 1-s MMI + two-station confirmation
                    net = self.run_plum_network(data["stationNetwork"])
                    peak_mmi, plum_alert = net["peak_mmi"], net["alert"]
                    evidence.extend(net["evidence"])
                    plum_extra = net["report"]
                else:
                    # Simplified single-sensor path (no neighbour confirmation)
                    telemetry_data = data.get("rawWaveformTelemetry", data)
                    raw_telemetry = WaveformTelemetry(**telemetry_data)
                    pga, peak_mmi, plum_alert, plum_evidence = self.plum.process_telemetry(raw_telemetry)
                    evidence.extend(plum_evidence)
                    plum_extra = {"plumMode": "single_station"}

                if plum_alert:
                    severity = 5 if peak_mmi >= 6.0 else 4
                    confidence = 0.95
                    actions.extend([
                        ActionItem(
                            type="alert",
                            resource="channel:cap",
                            detail=f"CRITICAL: PLUM Early Warning MMI {peak_mmi} detected. Initiate shelter-in-place.",
                            priority=0
                        ),
                        ActionItem(
                            type="dispatch_units",
                            resource="units:nearest",
                            detail=f"Deploy first responders to grid zone ({event_dict.get('lat')}, {event_dict.get('lon')}).",
                            priority=0,
                            units=["rescue", "fire_engine", "ambulance", "police"] if peak_mmi >= 6.0 else ["rescue", "ambulance"]
                        ),
                        ActionItem(
                            type="corridor_clear",
                            resource="road:corridor",
                            detail="Request Mobility Axis to clear primary emergency transit corridors.",
                            priority=1
                        )
                    ])
                elif plum_extra.get("plumSuppressed"):
                    severity = 1
                    sup = plum_extra["plumSuppressed"][0]
                    actions.append(ActionItem(
                        type="sensor_check",
                        resource=f"sensor:{sup['station']}",
                        detail=(f"PLUM suppressed an isolated trigger at {sup['station']} (MMI {sup['trigger_mmi']}): "
                                f"no neighbour confirmation. No public alert; schedule a sensor inspection."),
                        priority=3
                    ))
                else:
                    severity = 2
                    actions.append(ActionItem(
                        type="structural_check",
                        resource="sensor:seismic",
                        detail=f"Minor ground motion detected (MMI {peak_mmi}). Continue background monitoring.",
                        priority=3
                    ))
            except Exception as e:
                evidence.append(f"PLUM processing error: {str(e)}")
                severity = 3

        # 2. Acoustic / Crime Incident Path (Crime Hotspot Engine)
        elif kind in {"acoustic_gunshot_triangulation", "crime_hotspot"}:
            try:
                crime_event = RawCrimeEvent(
                    precursorEventId=event_dict.get("id", "EVT-RAW"),
                    category=kind,
                    sensorSource=source,
                    decibelLevel=data.get("decibelLevel"),
                    associatedContext=IncidentContext(**data.get("associatedContext", {}))
                )
                risk_score, classification, crime_evidence = self.crime.analyze_event(crime_event)
                evidence.extend(crime_evidence)

                severity = 5 if risk_score >= 0.75 else (3 if risk_score >= 0.5 else 2)
                confidence = 0.85

                actions.extend([
                    ActionItem(
                        type="dispatch_units",
                        resource="units:nearest",
                        detail=f"Dispatch Tactical/Police units for {classification} (Score: {risk_score}).",
                        priority=0 if severity >= 4 else 2,
                        units=(["police", "police", "ambulance"] if severity >= 5 else ["police"])
                    ),
                    ActionItem(
                        type="alert",
                        resource="channel:cap",
                        detail=f"Broadcast active incident notification for {kind}.",
                        priority=1
                    )
                ])
            except Exception as e:
                evidence.append(f"Crime hotspot processing error: {str(e)}")
                severity = 3

        # 3. Vision Smoke & Fire Detection (Edge CV Camera)
        elif kind in {"vision_fire_smoke", "cv_fire"}:
            conf = data.get("confidence", 0.8)
            fire_detected = data.get("fire_detected", False)
            smoke_detected = data.get("smoke_detected", False)
            flame_pct = data.get("flame_area_pct", 0.0)

            if fire_detected:
                severity = 5 if conf >= 0.85 else 4
                evidence.append(f"CV camera confirmed active fire (confidence={conf:.2f}, flame_area={flame_pct}%).")
                actions.extend([
                    ActionItem(
                        type="dispatch_units",
                        resource="units:nearest",
                        detail=f"Dispatch Fire Brigade & EMS to fire zone ({event_dict.get('lat')}, {event_dict.get('lon')}).",
                        priority=0,
                        units=["fire_engine", "fire_engine", "ambulance"] if severity >= 5 else ["fire_engine", "ambulance"]
                    ),
                    ActionItem(
                        type="alert",
                        resource="channel:cap",
                        detail="CRITICAL: Structural/Active fire detected via computer vision. Evacuate immediate sector.",
                        priority=0
                    ),
                    ActionItem(
                        type="corridor_clear",
                        resource="road:corridor",
                        detail="Request Mobility Axis to clear emergency fire transit corridor.",
                        priority=1
                    )
                ])
            elif smoke_detected:
                severity = 3
                evidence.append(f"CV camera detected dense smoke plume (confidence={conf:.2f}).")
                actions.extend([
                    ActionItem(
                        type="dispatch_units",
                        resource="units:nearest",
                        detail="Dispatch reconnaissance fire unit for smoke inspection.",
                        priority=2,
                        units=["fire_engine"]
                    ),
                    ActionItem(
                        type="alert",
                        resource="channel:cap",
                        detail="Advisory: Smoke plume detected in district.",
                        priority=2
                    )
                ])

        # 4. Hydrology & Flood Telemetry
        elif kind in {"flood_sensor", "flood"}:
            water_lvl = data.get("water_level_cm", 0.0)
            critical_thresh = data.get("critical_threshold_cm", 50.0)
            submerged_lanes = data.get("submerged_lane_count", 0)

            if water_lvl >= critical_thresh:
                severity = 5
                evidence.append(f"Critical flood stage breached: water level {water_lvl}cm >= threshold {critical_thresh}cm.")
                actions.extend([
                    ActionItem(
                        type="dispatch_units",
                        resource="units:nearest",
                        detail="Deploy Civil Protection & Water Rescue teams to flooded zone.",
                        priority=0,
                        units=["rescue", "rescue"]
                    ),
                    ActionItem(
                        type="alert",
                        resource="channel:cap",
                        detail=f"URGENT: Flash flood warning. Water depth at {water_lvl}cm. Avoid underpasses.",
                        priority=0
                    ),
                    ActionItem(
                        type="corridor_clear",
                        resource="road:corridor",
                        detail=f"Block flooded roadway ({submerged_lanes} lanes submerged) and divert all traffic.",
                        priority=0
                    )
                ])
            else:
                severity = 2
                evidence.append(f"Water level elevated ({water_lvl}cm) but below critical threshold ({critical_thresh}cm).")
                actions.append(ActionItem(
                    type="monitor",
                    resource="sensor:hydrology",
                    detail="Continue automated hydrological monitoring of water drainage rate.",
                    priority=3
                ))

        # 5. Direct Incident Triggers (Accident, Fire Alarm, Collapse, Gas Leak)
        elif kind in {"accident", "fire_alarm", "collapse", "gas_leak"}:
            severity = data.get("severity", 4)
            evidence.append(f"Received direct emergency trigger: kind={kind} from source={source}.")

            actions.extend([
                ActionItem(
                    type="dispatch_units",
                    resource="units:nearest",
                    detail=f"Dispatch Police & EMS to reported {kind} event.",
                    priority=0,
                    units={
                        "accident": ["police", "ambulance"],
                        "fire_alarm": ["fire_engine", "ambulance"],
                        "collapse": ["rescue", "fire_engine", "ambulance"],
                        "gas_leak": ["fire_engine", "police"],
                    }[kind] + (["ambulance"] if kind == "accident" and data.get("injuries_reported", 0) >= 3 else [])
                ),
                ActionItem(
                    type="alert",
                    resource="channel:cap",
                    detail=f"Broadcast active incident notification for {kind}.",
                    priority=1
                )
            ])

        # 4. Fallback Path
        else:
            evidence.append(f"Received generic telemetry event ({kind}).")
            severity = 1
            actions.append(ActionItem(
                type="monitor",
                resource="channel:internal",
                detail="Logging raw metric; no emergency dispatch required.",
                priority=4
            ))

        return EmergencyAgentResponse(
            severity=severity,
            confidence=confidence,
            plumAlertTriggered=plum_alert,
            peakMMI=peak_mmi,
            **plum_extra,
            actions=actions,
            evidence=evidence
        ).model_dump()