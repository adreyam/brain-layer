"""
City Brain orchestrator (LangGraph + Laya System 1 Decision Engine + Google GenAI).

Dual-Process Flow:
  event (pure JSON sensor telemetry)
    -> classify (System 1: Laya non-autoregressive ~33ms router + rule safety floor;
                 System 2 fallback: Gemini deliberative multi-agent analysis)
    -> dispatch (parallel fan-out)
    -> run_agent x N (emergency | mobility | environment | energy)
    -> synthesize (priority conflict resolution)
    -> audit (SQLite) & publish to urban actuators
"""
from __future__ import annotations
from modules.emergency_agent import EmergencyAgent

import json
import logging
import operator
import os
import sqlite3
import time
import uuid
from typing import Annotated, Callable, Literal, TypedDict

from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Laya System 1 Decision Engine Integration
# --------------------------------------------------------------------------- #
LAYA_AVAILABLE = False
_LAYA_ROUTER = None
try:
    from laya.integrations.langchain import LayaRouter
    LAYA_CRITERIA = {
        "emergency": (
            "seismic earthquake waveform motion, computer vision camera fire and dense smoke detection, "
            "hydrology flood sensors, rising water levels, gas leaks, gunshot acoustic triangulation, structural collapse, life safety"
        ),
        "mobility": (
            "traffic congestion index, road closures, smart parking full occupancy and street queue spillover, "
            "smart street lights outages on roads, transit delays, emergency responder route clearance"
        ),
        "environment": (
            "air pollution, PM2.5, PM10, ground-level ozone O3 spikes, hazardous AQI, smart garbage bin full and collection overflow, "
            "odor VOC spikes, toxic gas leaks, excessive noise"
        ),
        "energy": (
            "power grid load percentage, substation outages, blackouts, feeder overloads, "
            "smart street light electrical luminaire circuit faults"
        ),
    }
    _LAYA_ROUTER = LayaRouter(
        criteria=LAYA_CRITERIA,
        instructions="Which urban specialist axis must handle this sensor event?",
        confidence_threshold=0.60,
        fallback="deliberate",
    )
    LAYA_AVAILABLE = True
except Exception as _laya_err:
    logging.getLogger("city_brain").warning("Laya System 1 engine initialization skipped: %s", _laya_err)

Axis = Literal["emergency", "mobility", "environment", "energy"]

# Domain agent modules
_EMERGENCY_AGENT_INSTANCE = EmergencyAgent()

# Lower number = wins when two axes compete for the same resource
PRIORITY: dict[str, int] = {"emergency": 0, "environment": 1, "mobility": 2, "energy": 3}
ROUTE_THRESHOLD = 0.50  # score needed for an axis to be activated
SAFETY_FLOOR = 0.90     # a rule scoring >= this can never be downgraded by any model

# Axes whose agent is really integrated. Others are still routed and scored, but their
# agent is not run: the decision lists them under "unavailable_axes" and no action is sent.
# Example: set CITY_BRAIN_AXES=emergency,mobility once the mobility axis is ready.
INTEGRATED_AXES: set[str] = {
    a.strip() for a in os.getenv("CITY_BRAIN_AXES", "emergency").split(",") if a.strip()
}

LLM_MODEL = os.getenv("CITY_BRAIN_MODEL", "gemini-3.5-flash-lite")
log = logging.getLogger("city_brain")


# --------------------------------------------------------------------------- #
# Ingestion model: strictly pure JSON sensor telemetry
# --------------------------------------------------------------------------- #
class CityEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    source: str                      # "twin_iot_sensor" | "cv_camera_edge" | "hydrology_sensor" | "smart_bin_iot" | etc.
    kind: str                        # "vision_fire_smoke", "flood_sensor", "smart_waste_bin", "air_pollution", "smart_parking", "smart_street_light"...
    ts: float = Field(default_factory=time.time)
    lat: float | None = None
    lon: float | None = None
    data: dict = Field(default_factory=dict)   # raw sensor measurements and telemetry


# --------------------------------------------------------------------------- #
# 1) Deterministic Heuristics & Safety Floor Rules
# --------------------------------------------------------------------------- #
Score = tuple[str, float, str]

def rule_emergency(e: CityEvent) -> list[Score]:
    d, out = e.data, []
    
    # 1. Computer Vision Camera Smoke & Fire
    if e.kind in {"vision_fire_smoke", "cv_fire"}:
        conf = d.get("confidence", 0.0)
        if d.get("fire_detected") and conf >= 0.80:
            out.append(("emergency", 1.0, f"CV confirmed fire (conf={conf:.2f}, flame={d.get('flame_area_pct', 0)}%)"))
        elif d.get("smoke_detected") and conf >= 0.70:
            out.append(("emergency", 0.85, f"CV dense smoke plume detected (conf={conf:.2f})"))

    # 2. Hydrology / Flood Telemetry
    if e.kind in {"flood_sensor", "flood"}:
        lvl = d.get("water_level_cm", 0.0)
        crit = d.get("critical_threshold_cm", 50.0)
        if lvl >= crit:
            out.append(("emergency", 0.95, f"Flood critical breach: {lvl}cm >= {crit}cm"))
        elif lvl >= crit * 0.7:
            out.append(("emergency", 0.75, f"Flood warning: water at {lvl}cm"))

    # 3. Seismic Sensors (PLUM Early Warning)
    if e.kind == "seismic_sensor" or "rawWaveformTelemetry" in d:
        out.append(("emergency", 1.0, "seismic waveform telemetry received"))

    # 4. Acoustic Gunshot & Tactical Hotspots
    if e.kind == "acoustic_gunshot_triangulation":
        out.append(("emergency", 0.95, f"acoustic gunshot decibels={d.get('decibelLevel')}"))

    # 5. Direct Catastrophic Incident Telemetry
    if e.kind in {"accident", "fire_alarm", "collapse", "gas_leak"}:
        out.append(("emergency", 1.0, f"critical event kind={e.kind}"))

    if d.get("severity", 0) >= 4:
        out.append(("emergency", 0.85, f"high severity indicator={d['severity']}"))

    return out


def rule_environment(e: CityEvent) -> list[Score]:
    d, out = e.data, []
    
    # 1. Smart Waste Bins Collection
    if e.kind in {"smart_waste_bin", "waste_fill"}:
        fill = d.get("fill_pct", d.get("waste_fill_pct", 0))
        if fill >= 90:
            out.append(("environment", 0.90, f"bin overflow critical: fill={fill}%"))
        elif fill >= 75:
            out.append(("environment", 0.65, f"bin collection scheduled: fill={fill}%"))
        if d.get("odor_voc_ppm", 0) > 80:
            out.append(("environment", 0.75, f"bin hazardous VOC odor spike={d['odor_voc_ppm']}ppm"))

    # 2. Air Pollution & Quality (PM2.5, AQI, O3 Ozone)
    if e.kind in {"air_pollution", "air_quality"}:
        pm25 = d.get("pm25", 0)
        aqi = d.get("aqi", 0)
        o3 = d.get("o3", d.get("o3_ppb", d.get("o3_ugm3", 0)))

        if pm25 > 150 or aqi > 200 or o3 > 120:
            out.append(("environment", 0.95, f"Hazardous air pollution: PM2.5={pm25}, AQI={aqi}, O3={o3}"))
        elif pm25 > 75 or aqi > 100 or o3 > 70:
            out.append(("environment", 0.80, f"Unhealthy air pollution: PM2.5={pm25}, AQI={aqi}, O3={o3}"))
        elif pm25 > 35 or o3 > 50:
            out.append(("environment", 0.55, f"Moderate air pollution: PM2.5={pm25}, O3={o3}"))

    if d.get("noise_db", 0) > 85:
        out.append(("environment", 0.60, f"noise_db={d['noise_db']}"))

    return out


def rule_mobility(e: CityEvent) -> list[Score]:
    d, out = e.data, []

    # 1. Smart Parking
    if e.kind == "smart_parking":
        occ = d.get("occupancy_pct", 0)
        if occ >= 95 or d.get("spillover_into_street"):
            out.append(("mobility", 0.85, f"Parking facility saturated ({occ}%), road queue spillover"))
        elif occ >= 85:
            out.append(("mobility", 0.65, f"Parking congestion elevated ({occ}%)"))

    # 2. Smart Street Lights on transit corridors
    if e.kind == "smart_street_light":
        if d.get("status") == "fault" and d.get("lamps_offline_count", 0) >= 3:
            out.append(("mobility", 0.65, f"Street light corridor failure ({d.get('lamps_offline_count')} lamps offline)"))

    # 3. Traffic Sensors & Inrix / Loop Telemetry
    if d.get("congestion_index", 0) > 0.7:
        out.append(("mobility", 0.75, f"traffic congestion_index={d['congestion_index']}"))
    if 0 < d.get("avg_speed_kmh", 999) < 15:
        out.append(("mobility", 0.60, f"slow arterial avg_speed={d['avg_speed_kmh']}km/h"))
    if e.kind in {"road_closure", "transit_delay"}:
        out.append(("mobility", 0.70, f"kind={e.kind}"))

    return out


def rule_energy(e: CityEvent) -> list[Score]:
    d, out = e.data, []

    # 1. Smart Street Lighting electrical faults
    if e.kind == "smart_street_light" and (d.get("power_fault") or d.get("status") == "fault"):
        out.append(("energy", 0.60, "luminaire circuit electrical fault detected"))

    # 2. Grid Power Load & Substations
    if d.get("load_pct", 0) > 85:
        out.append(("energy", 0.75, f"high substation load={d['load_pct']}%"))
    if e.kind in {"outage", "grid_fault", "blackout"}:
        out.append(("energy", 0.90, f"grid failure kind={e.kind}"))

    return out


def rule_cascades(e: CityEvent) -> list[Score]:
    d, out = e.data, []
    
    # Flood submerging roadway -> immediate mobility closure & detours
    if e.kind in {"flood_sensor", "flood"} and d.get("water_level_cm", 0) >= d.get("critical_threshold_cm", 50):
        out.append(("mobility", 0.85, "flood overtopping roadway -> emergency detours"))

    # Confirmed fire -> mobility emergency green corridor & smoke air pollution
    if e.kind in {"vision_fire_smoke", "cv_fire", "fire_alarm"}:
        if d.get("fire_detected") or e.kind == "fire_alarm":
            out.append(("mobility", 0.75, "fire responders require green wave and corridor clearing"))
            out.append(("environment", 0.65, "fire smoke plume air quality hazard"))

    if e.kind in {"accident", "collapse"}:
        out.append(("mobility", 0.65, "emergency incident requires arterial traffic diversion"))

    if e.kind == "gas_leak":
        out.append(("environment", 0.75, "gas leak poses toxic airborne hazard"))

    if e.kind in {"outage", "blackout"} and d.get("hospital_affected"):
        out.append(("emergency", 0.95, "critical hospital infrastructure affected by blackout"))

    return out


RULES: list[Callable[[CityEvent], list[Score]]] = [
    rule_emergency, rule_environment, rule_mobility, rule_energy, rule_cascades,
]


# --------------------------------------------------------------------------- #
# System 2: Gemini LLM Deliberative Router (Fallback for ambiguous events)
# --------------------------------------------------------------------------- #
ROUTE_DECLARATION = types.FunctionDeclaration(
    name="route_event",
    description="Decide which city axes must handle this sensor event when System 1 requests deliberation.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "summary": types.Schema(
                type=types.Type.STRING,
                description="One-sentence analysis of the sensor event."
            ),
            "routes": types.Schema(
                type=types.Type.ARRAY,
                description="List of axes that need to act.",
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "axis": types.Schema(
                            type=types.Type.STRING,
                            enum=list(PRIORITY)
                        ),
                        "score": types.Schema(
                            type=types.Type.NUMBER,
                            description="How strongly this axis must act (0 = not at all, 1 = absolute priority)."
                        ),
                        "reason": types.Schema(type=types.Type.STRING),
                    },
                    required=["axis", "score", "reason"],
                ),
            ),
        },
        required=["summary", "routes"],
    ),
)

ROUTER_SYSTEM = """You are the System 2 Deliberative Brain of a smart city. You receive raw JSON sensor telemetry:
Specialist urban axes:
- emergency: Seismic telemetry (PLUM MMI), CV camera smoke/fire alerts, flood sensor thresholds, structural collapse, crime/gunshot acoustic triangulation.
- mobility: Traffic congestion, smart parking saturation & queue spillover, road closures from floods/fires, street light corridor blackouts.
- environment: Air pollution (PM2.5, PM10, O3 ozone, AQI, NO2), smart garbage bin fill levels & VOC odors.
- energy: Power grid load, luminaire circuit faults, substation outages.
Rules:
- Route to EVERY axis that must act.
- Score 0..1 how strongly each axis must act. Omit irrelevant axes.
- If data is nominal and nothing needs action, return an empty routes list."""


def gemini_router(e: CityEvent) -> dict:
    client = genai.Client()
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents="Incoming sensor telemetry:\n" + e.model_dump_json(),
        config=types.GenerateContentConfig(
            system_instruction=ROUTER_SYSTEM,
            temperature=0.1,
            tools=[types.Tool(function_declarations=[ROUTE_DECLARATION])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["route_event"]
                )
            )
        ),
    )

    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.function_call and part.function_call.name == "route_event":
                return dict(part.function_call.args)

    raise ValueError("Gemini returned no routing function call decision")


LLM_ROUTER: Callable[[CityEvent], dict] | None = (
    gemini_router
    if os.getenv("GEMINI_API_KEY") and os.getenv("CITY_BRAIN_LLM", "on") != "off"
    else None
)


# --------------------------------------------------------------------------- #
# Domain Specialist Agents
# --------------------------------------------------------------------------- #
def emergency_agent(e: CityEvent) -> dict:
    return _EMERGENCY_AGENT_INSTANCE.process_event(e.model_dump())


def mobility_agent(e: CityEvent) -> dict:
    d = e.data
    actions = []
    evidence = []

    if e.kind == "smart_parking":
        actions.append({
            "type": "parking_guidance",
            "resource": "vms:parking_signs",
            "detail": f"Divert incoming drivers to adjacent garages. Lot {d.get('lot_id')} full ({d.get('occupancy_pct')}%)."
        })
        evidence.append(f"smart_parking occupancy={d.get('occupancy_pct')}%")
    elif e.kind == "smart_street_light":
        actions.append({
            "type": "traffic_advisory",
            "resource": "vms:corridor_signs",
            "detail": f"Caution: Reduced visibility. Street light failure on sector {d.get('segment_id')}."
        })
        evidence.append(f"street_light lamps_offline={d.get('lamps_offline_count')}")
    else:
        actions.extend([
            {"type": "reroute", "resource": "road:corridor", "detail": "divert traffic around the incident zone"},
            {"type": "signal_priority", "resource": "signals:corridor", "detail": "green wave for emergency responders"},
        ])
        evidence.append(f"congestion={d.get('congestion_index', 'elevated')}")

    return {
        "severity": 3,
        "confidence": 0.80,
        "actions": actions,
        "evidence": evidence,
    }


def environment_agent(e: CityEvent) -> dict:
    d = e.data
    actions = []
    evidence = []

    if e.kind in {"smart_waste_bin", "waste_fill"}:
        actions.append({
            "type": "waste_collection_dispatch",
            "resource": "trucks:waste_fleet",
            "detail": f"Route automated collection truck to bin {d.get('bin_id', 'zone')} (fill={d.get('fill_pct', d.get('waste_fill_pct'))}%)."
        })
        evidence.append(f"smart_bin fill={d.get('fill_pct', d.get('waste_fill_pct'))}%")
    else:
        pm25 = d.get("pm25")
        aqi = d.get("aqi")
        o3 = d.get("o3", d.get("o3_ppb", d.get("o3_ugm3")))
        details = []
        if pm25 is not None:
            details.append(f"PM2.5={pm25}")
        if aqi is not None:
            details.append(f"AQI={aqi}")
        if o3 is not None:
            details.append(f"O3={o3}")

        metrics_str = ", ".join(details) if details else "elevated pollutant levels"

        actions.extend([
            {
                "type": "advisory",
                "resource": "channel:public",
                "detail": f"Air quality health advisory: {metrics_str}. Vulnerable individuals limit outdoor exertion."
            },
            {
                "type": "restrict",
                "resource": "road:corridor",
                "detail": "Limit heavy diesel vehicles and high-emission transport in the affected district."
            },
        ])
        evidence.append(f"air_pollution telemetry: {metrics_str}")

    return {
        "severity": 3,
        "confidence": 0.85,
        "actions": actions,
        "evidence": evidence,
    }


def energy_agent(e: CityEvent) -> dict:
    d = e.data
    actions = []
    evidence = []

    if e.kind == "smart_street_light":
        actions.append({
            "type": "maintenance_dispatch",
            "resource": "crews:lighting",
            "detail": f"Dispatch electrical technician to repair luminaire fault on {d.get('segment_id', 'sector')}."
        })
        evidence.append(f"luminaire power_fault={d.get('power_fault')}")
    else:
        actions.append({
            "type": "load_shed",
            "resource": "feeder:F3",
            "detail": f"shed non-critical loads to protect grid balance (load={d.get('load_pct', 0)}%)"
        })
        evidence.append(f"load={d.get('load_pct')}%")

    return {
        "severity": 2,
        "confidence": 0.75,
        "actions": actions,
        "evidence": evidence,
    }


AGENTS: dict[str, Callable[[CityEvent], dict]] = {
    "emergency": emergency_agent,
    "mobility": mobility_agent,
    "environment": environment_agent,
    "energy": energy_agent,
}


# --------------------------------------------------------------------------- #
# LangGraph Workflow & State
# --------------------------------------------------------------------------- #
class State(TypedDict):
    event: dict
    scores: list[dict]
    routes: list[str]
    analysis: str
    results: Annotated[list[dict], operator.add]
    decision: dict


def _rule_scores(e: CityEvent) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for rule in RULES:
        for axis, score, reason in rule(e):
            b = best.setdefault(axis, {"score": 0.0, "reasons": []})
            b["score"] = max(b["score"], score)
            b["reasons"].append(reason)
    return best


def _format_event_for_laya(e: CityEvent) -> str:
    """Serializes structured sensor event into a rich context string for Laya's encoder."""
    return f"Sensor telemetry [{e.source}]: kind={e.kind}, coordinates=({e.lat}, {e.lon}), data={json.dumps(e.data)}"


def classify(state: State) -> dict:
    e = CityEvent(**state["event"])
    rules = _rule_scores(e)
    
    # Check if any hard safety rule fired
    hard_safety_triggered = any(r["score"] >= SAFETY_FLOOR for r in rules.values())

    # 1. System 1: Laya Sub-35ms Non-Autoregressive Decision Engine
    laya_scores: dict[str, float] = {}
    laya_route = None
    laya_conf = 0.0
    laya_latency_ms = 0.0

    if _LAYA_ROUTER:
        try:
            t0 = time.time()
            laya_input = _format_event_for_laya(e)
            laya_route = _LAYA_ROUTER.invoke(laya_input)
            laya_latency_ms = round((time.time() - t0) * 1000, 2)
            
            if _LAYA_ROUTER.last_decision:
                ans = _LAYA_ROUTER.last_decision.get("answers", {}).get("route", {})
                probs = ans.get("probabilities", {})
                laya_conf = ans.get("answer_confidence", ans.get("confidence", 0.0))
                for axis in PRIORITY:
                    if axis in probs:
                        laya_scores[axis] = float(probs[axis])
        except Exception as exc:
            log.warning("Laya System 1 evaluation failed: %s", exc)

    # 2. Determine whether to execute System 1 Fast-Path or consult System 2 (Gemini)
    llm: dict[str, dict] | None = None
    
    # We stay in System 1 (Fast-Path) if:
    # - A hard safety rule triggered (>= 0.90), OR
    # - Laya reached a confident route (!= "deliberate"), OR
    # - LLM router is unavailable / disabled.
    is_system_1_confident = (laya_route is not None and laya_route != "deliberate")
    
    if hard_safety_triggered or is_system_1_confident or not LLM_ROUTER:
        analysis = (
            f"[System 1 Reflex] Non-autoregressive decision via Laya "
            f"(route='{laya_route}', conf={laya_conf:.2f}, latency={laya_latency_ms}ms) + safety rules."
        )
    else:
        # Ambiguous event: Consult System 2 (Gemini Deliberation)
        try:
            raw = LLM_ROUTER(e)
            llm = {}
            for r in raw.get("routes", []):
                axis = r.get("axis")
                if axis in PRIORITY:
                    sc = min(max(float(r.get("score", 0)), 0.0), 1.0)
                    b = llm.setdefault(axis, {"score": 0.0, "reasons": []})
                    b["score"] = max(b["score"], sc)
                    b["reasons"].append(str(r.get("reason", "")))
            analysis = f"[System 2 Deliberative] Gemini: {raw.get('summary', '')}"
        except Exception as exc:
            log.warning("Gemini System 2 deliberation failed, falling back to System 1 rules: %s", exc)
            analysis = f"[System 1 Fallback] Gemini failed: {type(exc).__name__}"

    # 3. Fuse Scores across all 4 axes
    merged: list[dict] = []
    for axis in PRIORITY:
        r = rules.get(axis)
        l_prob = laya_scores.get(axis, 0.0)
        s2 = (llm or {}).get(axis)

        score = 0.0
        reasons = []
        srcs = []

        if s2:
            score = s2["score"]
            srcs.append("system2_gemini")
            reasons += [f"gemini: {x}" for x in s2["reasons"]]
        elif l_prob > 0:
            score = l_prob
            srcs.append("system1_laya")
            reasons.append(f"laya_prob: {l_prob:.2f}")

        # Deterministic rules & safety floor
        if r:
            if r["score"] >= SAFETY_FLOOR or not srcs:
                score = max(score, r["score"])
                srcs.append("safety_rules")
                reasons += [f"rule: {x}" for x in r["reasons"]]
            elif r["score"] > 0:
                score = max(score, r["score"])
                srcs.append("rules")
                reasons += [f"rule: {x}" for x in r["reasons"]]

        if score > 0:
            merged.append({
                "axis": axis,
                "score": round(score, 2),
                "source": "+".join(srcs),
                "reasons": reasons
            })

    ranked = sorted(merged, key=lambda x: (-x["score"], PRIORITY[x["axis"]]))
    routes = [b["axis"] for b in ranked if b["score"] >= ROUTE_THRESHOLD]
    return {"scores": ranked, "routes": routes, "analysis": analysis}


def dispatch(state: State):
    if not state["routes"]:
        return ["synthesize"]
    return [Send("run_agent", {"event": state["event"], "axis": a}) for a in state["routes"]]


def run_agent(payload: dict) -> dict:
    axis, event = payload["axis"], CityEvent(**payload["event"])
    if axis not in INTEGRATED_AXES:
        return {"results": [{
            "axis": axis, "status": "unavailable", "severity": 0, "confidence": 0.0,
            "actions": [], "evidence": [], "latency_ms": 0.0,
            "note": f"{axis} axis not integrated yet: event routed, no action executed",
        }]}
    started = time.time()
    try:
        res = AGENTS[axis](event)
        res.update(axis=axis, status="ok")
    except Exception as exc:
        res = {
            "axis": axis,
            "status": "failed",
            "error": str(exc),
            "severity": 0,
            "confidence": 0.0,
            "actions": [],
            "evidence": []
        }
    res["latency_ms"] = round((time.time() - started) * 1000, 2)
    return {"results": [res]}


def synthesize(state: State) -> dict:
    results = sorted(state["results"], key=lambda r: PRIORITY[r["axis"]])
    claimed: dict[str, str] = {}
    final, conflicts = [], []
    for r in results:
        for a in r["actions"]:
            owner = claimed.get(a["resource"])
            if owner is None:
                claimed[a["resource"]] = r["axis"]
                final.append({**a, "axis": r["axis"]})
            elif owner != r["axis"]:
                conflicts.append({
                    "resource": a["resource"],
                    "kept": owner,
                    "dropped": r["axis"],
                    "action": a["type"]
                })
    ok = [r for r in results if r["status"] == "ok"]
    decision = {
        "primary_axis": ok[0]["axis"] if ok else None,
        "activated_axes": [r["axis"] for r in results],
        "failed_axes": [r["axis"] for r in results if r["status"] == "failed"],
        "unavailable_axes": [r["axis"] for r in results if r["status"] == "unavailable"],
        "actions": final,
        "conflicts": conflicts,
        "evidence": [ev for r in ok for ev in r["evidence"]],
        # full report of each agent that ran (severity, confidence, evidence, engine outputs such as PLUM MMI)
        "agent_reports": {r["axis"]: {k: v for k, v in r.items() if k != "actions"} for r in ok},
        "analysis": state["analysis"],
        "why": state["scores"],
    }
    return {"decision": decision}


class AuditLog:
    # One database at the project root, whichever folder the script is started from
    DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "city_brain_audit.db")

    def __init__(self, path: str | None = None):
        self.db = sqlite3.connect(path or self.DEFAULT_PATH, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS decisions "
            "(event_id TEXT, ts REAL, event TEXT, decision TEXT)"
        )

    def write(self, event: dict, decision: dict) -> None:
        self.db.execute(
            "INSERT INTO decisions VALUES (?,?,?,?)",
            (event["id"], time.time(), json.dumps(event), json.dumps(decision)),
        )
        self.db.commit()


def publish_to_axes(decision: dict) -> None:
    for action in decision.get("actions", []):
        target_axis = action.get("axis")
        log.info("DISPATCHING TO AXIS [%s]: %s -> %s", target_axis.upper(), action["type"], action["detail"])


def build_graph(audit: AuditLog | None = None):
    audit = audit or AuditLog()

    def audit_node(state: State) -> dict:
        audit.write(state["event"], state["decision"])
        publish_to_axes(state["decision"])
        return {}

    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("run_agent", run_agent)
    g.add_node("synthesize", synthesize)
    g.add_node("audit", audit_node)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", dispatch, ["run_agent", "synthesize"])
    g.add_edge("run_agent", "synthesize")
    g.add_edge("synthesize", "audit")
    g.add_edge("audit", END)
    return g.compile()


_GRAPH = None


def decide(event: dict | CityEvent) -> dict:
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    ev = event if isinstance(event, CityEvent) else CityEvent(**event)
    out = _GRAPH.invoke({"event": ev.model_dump(), "results": []})
    return out["decision"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demos = [
        # 1. CV Smoke & Fire Detection
        {
            "source": "cv_camera_edge",
            "kind": "vision_fire_smoke",
            "lat": 36.8045, "lon": 10.1820,
            "data": {"fire_detected": True, "confidence": 0.94, "flame_area_pct": 14.2}
        },
        # 2. Ultrasonic Flood Sensor
        {
            "source": "hydrology_sensor",
            "kind": "flood_sensor",
            "lat": 36.7980, "lon": 10.1710,
            "data": {"water_level_cm": 88.0, "critical_threshold_cm": 50.0, "submerged_lane_count": 2}
        },
        # 3. Smart Garbage Bin Fill Level
        {
            "source": "smart_bin_iot",
            "kind": "smart_waste_bin",
            "data": {"bin_id": "BIN-TUNIS-42", "fill_pct": 95.0, "odor_voc_ppm": 110.0}
        },
        # 4. Multi-Pollutant Air Quality Station (including Ozone O3)
        {
            "source": "air_quality_station",
            "kind": "air_pollution",
            "lat": 36.8060, "lon": 10.1835,
            "data": {
                "station_id": "AQ-CENTRE-01",
                "aqi": 185,
                "pm25": 110.4,
                "pm10": 155.0,
                "o3_ppb": 125.0,
                "no2_ppb": 52.0
            }
        },
        # 5. Smart Parking Saturated
        {
            "source": "parking_management",
            "kind": "smart_parking",
            "data": {"lot_id": "LOT-CENTRE-VILLE", "occupancy_pct": 98.0, "spillover_into_street": True}
        },
        # 5. Smart Street Light Power Fault
        {
            "source": "lighting_controller",
            "kind": "smart_street_light",
            "data": {"segment_id": "SEG-BOURGUIBA-03", "status": "fault", "lamps_offline_count": 7, "power_fault": True}
        },
        # 6. Grid Blackout Affecting Hospital
        {
            "source": "sensor",
            "kind": "outage",
            "data": {"load_pct": 30, "hospital_affected": True}
        }
    ]

    for d in demos:
        print(f"\n=================== DEMO: {d['kind']} ===================")
        res = decide(d)
        print("Analysis :", res.get("analysis"))
        print("Primary  :", res.get("primary_axis"))
        print("Axes     :", res.get("activated_axes"))
        print("Actions  :", json.dumps(res.get("actions"), indent=2))
