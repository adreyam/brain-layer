"""
Emergency Axis Agent.

Serves as the operational dispatch layer for the Emergency axis. It integrates
standalone physics/analytic modules (PLUM Engine & Crime Hotspot Engine) to construct
standardized dispatch orders and CAP alerts.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .plum_engine import PLUMEngine, SensorReading, WaveformTelemetry
from .crime_hotspot import CrimeHotspotEngine, RiskTerrainContext, RawCrimeEvent, IncidentContext


class ActionItem(BaseModel):
    type: str          # "dispatch_units", "alert", "corridor_clear", "structural_check"
    resource: str      # "units:nearest", "channel:cap", "road:corridor"
    detail: str
    priority: int = 1  # 0 = Critical / Immediate


class EmergencyAgentResponse(BaseModel):
    axis: str = "emergency"
    severity: int
    confidence: float
    status: str = "ok"
    plumAlertTriggered: bool = False
    peakMMI: float = 0.0
    actions: List[ActionItem]
    evidence: List[str]


class EmergencyAgent:
    """Operational agent coordinating response logic and underlying domain engines."""

    def __init__(self, plum_engine: Optional[PLUMEngine] = None, crime_engine: Optional[CrimeHotspotEngine] = None):
        self.plum = plum_engine or PLUMEngine()
        self.crime = crime_engine or CrimeHotspotEngine()

    def process_event(self, event_dict: dict) -> dict:
        """Main entry point compatible with LangGraph Orchestrator payload schema."""
        actions: List[ActionItem] = []
        evidence: List[str] = []
        severity = 1
        confidence = 0.8
        plum_alert = False
        peak_mmi = 0.0

        kind = event_dict.get("kind", "")
        data = event_dict.get("data", {})
        source = event_dict.get("source", "unknown")

        # 1. Seismic Telemetry Path (PLUM Engine)
        if "rawWaveformTelemetry" in data or kind == "seismic_sensor":
            try:
                telemetry_data = data.get("rawWaveformTelemetry", data)
                raw_telemetry = WaveformTelemetry(**telemetry_data)
                pga, peak_mmi, plum_alert, plum_evidence = self.plum.process_telemetry(raw_telemetry)
                evidence.extend(plum_evidence)

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
                            priority=0
                        ),
                        ActionItem(
                            type="corridor_clear",
                            resource="road:corridor",
                            detail="Request Mobility Axis to clear primary emergency transit corridors.",
                            priority=1
                        )
                    ])
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
                        priority=0 if severity >= 4 else 2
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

        # 3. Direct Incident Triggers (Accident, Fire, Gas Leak, Collapse)
        elif kind in {"accident", "fire_alarm", "collapse", "gas_leak"}:
            severity = data.get("severity", 4)
            evidence.append(f"Received direct emergency trigger: kind={kind} from source={source}.")

            actions.extend([
                ActionItem(
                    type="dispatch_units",
                    resource="units:nearest",
                    detail=f"Dispatch Police & EMS to reported {kind} event.",
                    priority=0
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
            actions=actions,
            evidence=evidence
        ).model_dump()