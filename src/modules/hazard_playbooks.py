"""
Deterministic Response Playbooks for Catastrophic Hazards.
Generates structured ActionItems and CAP-compatible alerts based on hazard type and risk level.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .hazard_common import HazardAssessment, map_risk_level_to_severity


class PlaybookAction(BaseModel):
    type: str          # "alert", "dispatch_units", "corridor_clear", "structural_check", "evacuate", "monitor"
    resource: str      # "channel:cap", "units:nearest", "road:corridor", etc.
    detail: str
    priority: int = 1  # 0 = Critical/Immediate, 1 = High, 2 = Medium, 3 = Low
    requires_human_approval: bool = False


# Base Deterministic Playbook Catalog
PLAYBOOK_CATALOG: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "flood": {
        "watch": [
            {
                "type": "monitor",
                "resource": "sensor:hydro",
                "detail": "Increase polling frequency of river gauges and urban drainage sensors.",
                "priority": 2,
                "requires_human_approval": False,
            },
            {
                "type": "preposition_equipment",
                "resource": "units:public_works",
                "detail": "Pre-position mobile water extraction pumps and flood barriers in low-lying sectors.",
                "priority": 2,
                "requires_human_approval": False,
            },
            {
                "type": "advisory",
                "resource": "axis:mobility",
                "detail": "Advise Mobility Axis of potential flash flood conditions on arterial routes.",
                "priority": 2,
                "requires_human_approval": False,
            },
        ],
        "warning": [
            {
                "type": "alert",
                "resource": "channel:cap",
                "detail": "WARNING: Significant flood risk detected. Avoid flood-prone zones and underpasses.",
                "priority": 1,
                "requires_human_approval": False,
            },
            {
                "type": "corridor_clear",
                "resource": "road:corridor",
                "detail": "Close vulnerable underpasses and divert traffic away from river basin corridors.",
                "priority": 1,
                "requires_human_approval": False,
            },
            {
                "type": "advisory",
                "resource": "axis:energy",
                "detail": "Alert Energy Axis to verify substation perimeter flood defenses in inundated sectors.",
                "priority": 1,
                "requires_human_approval": False,
            },
        ],
        "critical": [
            {
                "type": "evacuate",
                "resource": "channel:cap",
                "detail": "CRITICAL: Immediate evacuation advisory for inundated low-lying flood sectors.",
                "priority": 0,
                "requires_human_approval": True,
            },
            {
                "type": "dispatch_units",
                "resource": "units:nearest",
                "detail": "Deploy swift-water rescue teams and emergency first responders to designated flood zones.",
                "priority": 0,
                "requires_human_approval": True,
            },
            {
                "type": "corridor_clear",
                "resource": "road:corridor",
                "detail": "Establish full emergency clearance on primary flood evacuation corridors.",
                "priority": 0,
                "requires_human_approval": True,
            },
        ],
    },
    "tsunami": {
        "warning": [
            {
                "type": "alert",
                "resource": "channel:cap",
                "detail": "TSUNAMI WARNING: Offshore earthquake detected. Clear beaches, marinas, and port waterfronts immediately.",
                "priority": 1,
                "requires_human_approval": False,
            },
            {
                "type": "restrict",
                "resource": "road:coastal",
                "detail": "Halt inbound traffic to coastal highways and maritime ports.",
                "priority": 1,
                "requires_human_approval": False,
            },
            {
                "type": "advisory",
                "resource": "axis:mobility",
                "detail": "Coordinate coastal traffic signal priority for outbound evacuation flow.",
                "priority": 1,
                "requires_human_approval": False,
            },
        ],
        "critical": [
            {
                "type": "evacuate",
                "resource": "channel:cap",
                "detail": "CRITICAL TSUNAMI EVACUATION: Destructive tsunami wave incoming. Move inland and to high ground immediately.",
                "priority": 0,
                "requires_human_approval": True,
            },
            {
                "type": "activate_shelters",
                "resource": "facilities:shelters",
                "detail": "Unlock and activate inland elevated emergency reception centers and shelters.",
                "priority": 0,
                "requires_human_approval": True,
            },
            {
                "type": "dispatch_units",
                "resource": "units:nearest",
                "detail": "Deploy emergency sirens and tactical police for mandatory coastal evacuation.",
                "priority": 0,
                "requires_human_approval": True,
            },
        ],
    },
    "storm": {
        "warning": [
            {
                "type": "alert",
                "resource": "channel:cap",
                "detail": "STORM WARNING: Damaging wind gusts and intense precipitation imminent. Secure loose outdoor objects.",
                "priority": 1,
                "requires_human_approval": False,
            },
            {
                "type": "restrict",
                "resource": "sites:construction",
                "detail": "Mandate temporary halt on tower crane operations and high-altitude construction works.",
                "priority": 1,
                "requires_human_approval": False,
            },
            {
                "type": "advisory",
                "resource": "axis:energy",
                "detail": "Notify Energy Axis of heightened transmission line and tree-fall outage risk.",
                "priority": 2,
                "requires_human_approval": False,
            },
        ],
        "critical": [
            {
                "type": "alert",
                "resource": "channel:cap",
                "detail": "CRITICAL STORM / TORNADO WARNING: Take immediate shelter in an interior room on the lowest floor.",
                "priority": 0,
                "requires_human_approval": True,
            },
            {
                "type": "corridor_clear",
                "resource": "road:corridor",
                "detail": "Suspend above-ground mass transit lines and clear high-risk debris corridors.",
                "priority": 0,
                "requires_human_approval": True,
            },
            {
                "type": "dispatch_units",
                "resource": "units:nearest",
                "detail": "Mobilize emergency tree-clearing crews and downed-wire containment teams.",
                "priority": 0,
                "requires_human_approval": True,
            },
        ],
    },
    "earthquake": {
        "watch": [
            {
                "type": "structural_check",
                "resource": "sensor:seismic",
                "detail": "Moderate regional earthquake recorded in catalog. Maintain automated sensor monitoring.",
                "priority": 2,
                "requires_human_approval": False,
            },
        ],
        "warning": [
            {
                "type": "alert",
                "resource": "channel:cap",
                "detail": "EARTHQUAKE ALERT: Significant seismic event recorded. Inspect gas and electric lines for damage.",
                "priority": 1,
                "requires_human_approval": False,
            },
            {
                "type": "structural_check",
                "resource": "infrastructure:critical",
                "detail": "Initiate priority structural inspections on bridges, overpasses, and reservoir dams.",
                "priority": 1,
                "requires_human_approval": False,
            },
            {
                "type": "advisory",
                "resource": "axis:mobility",
                "detail": "Notify Mobility Axis to enforce speed restrictions across elevated highway viaducts.",
                "priority": 1,
                "requires_human_approval": False,
            },
        ],
        "critical": [
            {
                "type": "alert",
                "resource": "channel:cap",
                "detail": "CRITICAL EARTHQUAKE: Major ground shaking. Expect aftershocks; evacuate compromised structures.",
                "priority": 0,
                "requires_human_approval": True,
            },
            {
                "type": "dispatch_units",
                "resource": "units:nearest",
                "detail": "Deploy Search and Rescue (SAR) taskforces and urban heavy rescue units.",
                "priority": 0,
                "requires_human_approval": True,
            },
            {
                "type": "corridor_clear",
                "resource": "road:corridor",
                "detail": "Lock down compromised highway structures and clear priority emergency triage corridors.",
                "priority": 0,
                "requires_human_approval": True,
            },
        ],
    },
}


def get_playbook_actions(assessment: HazardAssessment) -> List[PlaybookAction]:
    """
    Retrieves deterministic actions for the given hazard and risk level.
    Enforces that all actions at 'critical' or involving 'evacuate' require human approval.
    """
    hazard_key = assessment.hazard.lower()
    level_key = assessment.risk_level.lower()

    hazard_actions = PLAYBOOK_CATALOG.get(hazard_key, {})
    raw_actions = hazard_actions.get(level_key, [])

    if not raw_actions and level_key != "low":
        # Fallback to general monitoring
        raw_actions = [{
            "type": "monitor",
            "resource": "channel:internal",
            "detail": f"Active {assessment.hazard} condition at {assessment.risk_level} level; continue monitoring.",
            "priority": 2,
            "requires_human_approval": False,
        }]

    actions: List[PlaybookAction] = []
    for item in raw_actions:
        action = PlaybookAction(**item)
        # Enforce human-approval invariant for evacuation or critical level
        if action.type == "evacuate" or assessment.risk_level == "critical":
            action.requires_human_approval = True
        actions.append(action)

    return actions


def generate_cap_alert(
    assessment: HazardAssessment,
    msg_type: str = "Alert",
    trust_verdict_id: Optional[str] = None,
    references: Optional[str] = None
) -> str:
    """
    Generates a CAP 1.2 compliant XML string for catastrophic hazard notifications.
    """
    import datetime
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S-00:00")
    exp_iso = (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30)
    ).strftime("%Y-%m-%dT%H:%M:%S-00:00")

    identifier = f"ALR-HAZ-{assessment.hazard.upper()}-{int(datetime.datetime.now().timestamp())}"
    severity_cap = "Extreme" if assessment.risk_level == "critical" else (
        "Severe" if assessment.risk_level == "warning" else "Moderate"
    )
    urgency_cap = "Immediate" if assessment.risk_level in {"critical", "warning"} else "Expected"

    lead_time_s = (assessment.eta_minutes * 60) if assessment.eta_minutes else 0
    ref_element = f"  <references>{references}</references>\n" if references else ""
    trust_element = f"""    <parameter>
      <valueName>trust_verdict_id</valueName>
      <value>{trust_verdict_id or 'auto-system-verified'}</value>
    </parameter>
"""

    return f"""<?xml version='1.0' encoding='utf-8'?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>{identifier}</identifier>
  <sender>citybrain@triplet-07.local</sender>
  <sent>{now_iso}</sent>
  <status>Actual</status>
  <msgType>{msg_type}</msgType>
  <source>citybrain/detector/{assessment.hazard}</source>
  <scope>Public</scope>
{ref_element}  <info>
    <category>Safety</category>
    <event>{assessment.hazard.capitalize()} Hazard</event>
    <urgency>{urgency_cap}</urgency>
    <severity>{severity_cap}</severity>
    <certainty>Observed</certainty>
    <expires>{exp_iso}</expires>
    <senderName>City Brain - Emergency Response</senderName>
    <headline>{assessment.hazard.upper()} {assessment.risk_level.upper()} Directive</headline>
    <instruction>{assessment.evidence[0] if assessment.evidence else 'Follow civil protection instructions.'}</instruction>
{trust_element}    <parameter>
      <valueName>lead_time_s</valueName>
      <value>{lead_time_s}</value>
    </parameter>
  </info>
</alert>
""".strip()


def run_llm_planner_if_enabled(
    assessment: HazardAssessment,
    playbook_actions: List[PlaybookAction]
) -> List[PlaybookAction]:
    """
    Optional LLM planner behind CITY_BRAIN_LLM and GEMINI_API_KEY.
    Refines alert phrasing or priority while strictly preserving hazard, risk_level,
    and human-approval requirements.
    Falls back silently to playbook_actions on any error.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    llm_enabled = os.getenv("CITY_BRAIN_LLM", "on").lower() != "off"

    if not api_key or not llm_enabled:
        return playbook_actions

    try:
        from google import genai
        from google.genai import types

        client = genai.Client()
        prompt = (
            f"You are an emergency civil defense planner. Refine these actions for a "
            f"{assessment.hazard} at {assessment.risk_level} level.\n"
            f"Current actions:\n{[a.model_dump() for a in playbook_actions]}\n"
            f"Provide concise tailored instructions for public safety. Do not remove any evacuation order."
        )

        model_name = os.getenv("CITY_BRAIN_MODEL", "gemini-3.5-flash-lite")
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        # Note: If LLM produces text, we can refine the detail of the alert action
        if resp.text:
            refined_actions = [a.model_copy() for a in playbook_actions]
            for a in refined_actions:
                if a.type == "alert":
                    a.detail = f"{a.detail} (Advisor: {resp.text.strip()[:150]})"
            return refined_actions

    except Exception:
        # Silent fallback to deterministic playbook as specified
        pass

    return playbook_actions

