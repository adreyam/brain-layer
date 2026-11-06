"""
Emergency Axis Agent.

Serves as the operational dispatch layer for the Emergency axis. It integrates
standalone physics/analytic modules (PLUM Engine & Crime Hotspot Engine, Flood Engine,
Catalog Quake Engine, Tsunami Engine, and Storm Engine) to construct standardized dispatch
orders and CAP alerts.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .plum_engine import PLUMEngine, WaveformTelemetry
from .crime_hotspot import CrimeHotspotEngine, RawCrimeEvent, IncidentContext
from .hazard_common import (
    HazardAssessment,
    load_hazard_config,
    map_risk_level_to_severity,
)
from .flood_engine import FloodEngine
from .catalog_quake_engine import CatalogQuakeEngine
from .tsunami_engine import TsunamiEngine
from .storm_engine import StormEngine
from .hazard_playbooks import (
    PlaybookAction,
    generate_cap_alert,
    get_playbook_actions,
    run_llm_planner_if_enabled,
)


class ActionItem(BaseModel):
    type: str          # "dispatch_units", "alert", "corridor_clear", "structural_check", "evacuate", "monitor"
    resource: str      # "units:nearest", "channel:cap", "road:corridor"
    detail: str
    priority: int = 1  # 0 = Critical / Immediate, 1 = High, 2 = Medium, 3 = Low
    requires_human_approval: bool = False


class EmergencyAgentResponse(BaseModel):
    axis: str = "emergency"
    severity: int
    confidence: float
    status: str = "ok"
    plumAlertTriggered: bool = False
    peakMMI: float = 0.0
    hazard: Optional[str] = None
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    requires_human_approval: bool = False
    hazard_assessment: Optional[HazardAssessment] = None
    actions: List[ActionItem]
    evidence: List[str]


class EmergencyAgent:
    """Operational agent coordinating response logic and underlying domain engines."""

    def __init__(
        self,
        plum_engine: Optional[PLUMEngine] = None,
        crime_engine: Optional[CrimeHotspotEngine] = None,
        flood_engine: Optional[FloodEngine] = None,
        catalog_quake_engine: Optional[CatalogQuakeEngine] = None,
        tsunami_engine: Optional[TsunamiEngine] = None,
        storm_engine: Optional[StormEngine] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or load_hazard_config()
        self.plum = plum_engine or PLUMEngine()
        self.crime = crime_engine or CrimeHotspotEngine()
        self.flood = flood_engine or FloodEngine(self.config)
        self.catalog_quake = catalog_quake_engine or CatalogQuakeEngine(self.config)
        self.tsunami = tsunami_engine or TsunamiEngine(self.config)
        self.storm = storm_engine or StormEngine(self.config)

        # In-memory short-term buffer for multi-event correlation
        self._event_buffer: List[Dict[str, Any]] = []
        self._last_plum_active_ts: float = 0.0
        self._last_catalog_assessment: Optional[HazardAssessment] = None

    def _buffer_event(self, event_dict: Dict[str, Any]) -> None:
        """Stores event in memory and purges records older than 6 hours."""
        now = time.time()
        self._event_buffer.append(dict(event_dict))
        # Keep only events within last 6 hours (21600 seconds)
        cutoff = now - 21600
        self._event_buffer = [
            e for e in self._event_buffer if float(e.get("ts", now)) >= cutoff
        ]

    def _get_buffered_by_kinds(self, kinds: set[str]) -> List[Dict[str, Any]]:
        return [e for e in self._event_buffer if e.get("kind") in kinds]

    def _hazard_assessment_to_response(
        self,
        assessment: HazardAssessment,
        raw_event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Converts a HazardAssessment into a standardized EmergencyAgentResponse dictionary."""
        playbook_actions = get_playbook_actions(assessment)
        refined_actions = run_llm_planner_if_enabled(assessment, playbook_actions)

        action_items: List[ActionItem] = []
        for act in refined_actions:
            action_items.append(
                ActionItem(
                    type=act.type,
                    resource=act.resource,
                    detail=act.detail,
                    priority=act.priority,
                    requires_human_approval=act.requires_human_approval,
                )
            )

        # If warning or critical, attach CAP alert action
        if assessment.risk_level in {"warning", "critical"}:
            trust_id = raw_event.get("data", {}).get("trust_verdict_id") or raw_event.get("trust_verdict_id")
            cap_xml = generate_cap_alert(
                assessment,
                msg_type="Alert",
                trust_verdict_id=trust_id,
            )
            action_items.append(
                ActionItem(
                    type="alert_cap",
                    resource="channel:cap",
                    detail=f"CAP 1.2 payload generated ({assessment.hazard.upper()} {assessment.risk_level.upper()}).",
                    priority=0 if assessment.risk_level == "critical" else 1,
                    requires_human_approval=assessment.needs_human_confirmation,
                )
            )

        severity = map_risk_level_to_severity(assessment.risk_level, self.config)
        requires_approval = (
            assessment.needs_human_confirmation
            or any(a.requires_human_approval for a in action_items)
            or assessment.risk_level == "critical"
        )

        return EmergencyAgentResponse(
            axis="emergency",
            severity=severity,
            confidence=assessment.confidence,
            status="ok",
            plumAlertTriggered=False,
            peakMMI=0.0,
            hazard=assessment.hazard,
            risk_level=assessment.risk_level,
            risk_score=assessment.risk_score,
            requires_human_approval=requires_approval,
            hazard_assessment=assessment,
            actions=action_items,
            evidence=assessment.evidence,
        ).model_dump()

    def process_event(self, event_dict: dict) -> dict:
        """Main entry point compatible with LangGraph Orchestrator payload schema."""
        self._buffer_event(event_dict)

        actions: List[ActionItem] = []
        evidence: List[str] = []
        severity = 1
        confidence = 0.8
        plum_alert = False
        peak_mmi = 0.0

        kind = event_dict.get("kind", "")
        data = event_dict.get("data", {})
        source = event_dict.get("source", "unknown")

        # ------------------------------------------------------------------- #
        # 1. Seismic Telemetry Path (PLUM Engine) - Existing Path
        # ------------------------------------------------------------------- #
        if "rawWaveformTelemetry" in data or kind == "seismic_sensor":
            try:
                telemetry_data = data.get("rawWaveformTelemetry", data)
                raw_telemetry = WaveformTelemetry(**telemetry_data)
                pga, peak_mmi, plum_alert, plum_evidence = self.plum.process_telemetry(raw_telemetry)
                evidence.extend(plum_evidence)

                if plum_alert:
                    self._last_plum_active_ts = time.time()
                    severity = 5 if peak_mmi >= 6.0 else 4
                    confidence = 0.95
                    actions.extend([
                        ActionItem(
                            type="alert",
                            resource="channel:cap",
                            detail=f"CRITICAL: PLUM Early Warning MMI {peak_mmi} detected. Initiate shelter-in-place.",
                            priority=0,
                            requires_human_approval=True,
                        ),
                        ActionItem(
                            type="dispatch_units",
                            resource="units:nearest",
                            detail=f"Deploy first responders to grid zone ({event_dict.get('lat')}, {event_dict.get('lon')}).",
                            priority=0,
                            requires_human_approval=True,
                        ),
                        ActionItem(
                            type="corridor_clear",
                            resource="road:corridor",
                            detail="Request Mobility Axis to clear primary emergency transit corridors.",
                            priority=1,
                            requires_human_approval=False,
                        ),
                    ])
                else:
                    severity = 2
                    actions.append(ActionItem(
                        type="structural_check",
                        resource="sensor:seismic",
                        detail=f"Minor ground motion detected (MMI {peak_mmi}). Continue background monitoring.",
                        priority=3,
                        requires_human_approval=False,
                    ))
            except Exception as e:
                evidence.append(f"PLUM processing error: {str(e)}")
                severity = 3

            return EmergencyAgentResponse(
                severity=severity,
                confidence=confidence,
                plumAlertTriggered=plum_alert,
                peakMMI=peak_mmi,
                requires_human_approval=(severity >= 5),
                actions=actions,
                evidence=evidence,
            ).model_dump()

        # ------------------------------------------------------------------- #
        # 2. Acoustic / Crime Incident Path (Crime Hotspot Engine) - Existing
        # ------------------------------------------------------------------- #
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
                        requires_human_approval=(severity >= 5),
                    ),
                    ActionItem(
                        type="alert",
                        resource="channel:cap",
                        detail=f"Broadcast active incident notification for {kind}.",
                        priority=1,
                        requires_human_approval=False,
                    ),
                ])
            except Exception as e:
                evidence.append(f"Crime hotspot processing error: {str(e)}")
                severity = 3

            return EmergencyAgentResponse(
                severity=severity,
                confidence=confidence,
                plumAlertTriggered=plum_alert,
                peakMMI=peak_mmi,
                requires_human_approval=(severity >= 5),
                actions=actions,
                evidence=evidence,
            ).model_dump()

        # ------------------------------------------------------------------- #
        # 3. Hydrological & Flood Hazard Path (NEW)
        # ------------------------------------------------------------------- #
        elif kind in {"hydro_observation", "weather_forecast"}:
            recent_events = self._get_buffered_by_kinds({
                "hydro_observation", "weather_forecast", "weather_observation"
            })
            assessment = self.flood.evaluate(recent_events)
            return self._hazard_assessment_to_response(assessment, event_dict)

        # ------------------------------------------------------------------- #
        # 4. Seismic Catalog & Tsunami Path (NEW)
        # ------------------------------------------------------------------- #
        elif kind == "seismic_catalog":
            plum_active = (time.time() - self._last_plum_active_ts) < 120.0
            quake_assessment = self.catalog_quake.evaluate(
                event_dict,
                plum_active=plum_active,
                plum_evidence=["Recent local PLUM ground motion recorded."] if plum_active else None,
            )
            self._last_catalog_assessment = quake_assessment

            # If offshore candidate, evaluate tsunami potential
            if quake_assessment.metadata.get("offshore") and quake_assessment.metadata.get("magnitude", 0) >= 6.5:
                recent_sea_level = self._get_buffered_by_kinds({"sea_level_observation"})
                tsunami_assessment = self.tsunami.evaluate(quake_assessment, recent_sea_level)
                if tsunami_assessment.risk_level in {"warning", "critical"}:
                    return self._hazard_assessment_to_response(tsunami_assessment, event_dict)

            return self._hazard_assessment_to_response(quake_assessment, event_dict)

        # ------------------------------------------------------------------- #
        # 5. Sea Level Gauge Observation Path (NEW)
        # ------------------------------------------------------------------- #
        elif kind == "sea_level_observation":
            if self._last_catalog_assessment is not None:
                # Correlate with recent catalog earthquake
                recent_sea_level = self._get_buffered_by_kinds({"sea_level_observation"})
                tsunami_assessment = self.tsunami.evaluate(self._last_catalog_assessment, recent_sea_level)
                if tsunami_assessment.metadata.get("candidate"):
                    return self._hazard_assessment_to_response(tsunami_assessment, event_dict)

            # Normal baseline sea-level telemetry
            evidence.append(f"Recorded sea-level telemetry from station {data.get('station_id', 'unknown')}.")
            return EmergencyAgentResponse(
                severity=1,
                confidence=0.8,
                status="ok",
                hazard="tsunami",
                risk_level="low",
                risk_score=0.05,
                actions=[ActionItem(type="monitor", resource="sensor:tide_gauge", detail="Sea-level within baseline limits.", priority=3)],
                evidence=evidence,
            ).model_dump()

        # ------------------------------------------------------------------- #
        # 6. Meteorological & Storm Hazard Path (NEW)
        # ------------------------------------------------------------------- #
        elif kind in {"weather_observation", "official_weather_alert"}:
            recent_weather = self._get_buffered_by_kinds({
                "weather_observation", "weather_forecast", "official_weather_alert"
            })
            storm_assessment = self.storm.evaluate(recent_weather)

            # Also check if heavy rain triggers flood assessment
            recent_hydro = self._get_buffered_by_kinds({
                "hydro_observation", "weather_forecast", "weather_observation"
            })
            flood_assessment = self.flood.evaluate(recent_hydro)

            # Select dominant hazard
            if flood_assessment.risk_score > storm_assessment.risk_score and flood_assessment.risk_level in {"warning", "critical"}:
                return self._hazard_assessment_to_response(flood_assessment, event_dict)

            return self._hazard_assessment_to_response(storm_assessment, event_dict)

        # ------------------------------------------------------------------- #
        # 7. Direct Incident Triggers (Accident, Fire, Gas Leak, Collapse) - Existing
        # ------------------------------------------------------------------- #
        elif kind in {"accident", "fire_alarm", "collapse", "gas_leak"}:
            severity = data.get("severity", 4)
            evidence.append(f"Received direct emergency trigger: kind={kind} from source={source}.")

            actions.extend([
                ActionItem(
                    type="dispatch_units",
                    resource="units:nearest",
                    detail=f"Dispatch Police & EMS to reported {kind} event.",
                    priority=0,
                    requires_human_approval=(severity >= 5),
                ),
                ActionItem(
                    type="alert",
                    resource="channel:cap",
                    detail=f"Broadcast active incident notification for {kind}.",
                    priority=1,
                    requires_human_approval=False,
                ),
            ])

        # ------------------------------------------------------------------- #
        # 8. Fallback Path - Existing
        # ------------------------------------------------------------------- #
        else:
            evidence.append(f"Received generic telemetry event ({kind}).")
            severity = 1
            actions.append(ActionItem(
                type="monitor",
                resource="channel:internal",
                detail="Logging raw metric; no emergency dispatch required.",
                priority=4,
                requires_human_approval=False,
            ))

        return EmergencyAgentResponse(
            severity=severity,
            confidence=confidence,
            plumAlertTriggered=plum_alert,
            peakMMI=peak_mmi,
            requires_human_approval=(severity >= 5),
            actions=actions,
            evidence=evidence,
        ).model_dump()
