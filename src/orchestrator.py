"""
City Brain orchestrator (LangGraph + Google GenAI).

Flow:
  event -> classify (Gemini LLM analysis + rule safety floor) -> dispatch (parallel fan-out)
        -> run_agent x N (emergency | mobility | environment | energy)
        -> synthesize (conflict resolution by priority) -> audit (SQLite) & publish

pip install langgraph pydantic google-genai
"""
from __future__ import annotations
try:
    from modules.emergency_agent import EmergencyAgent
except ImportError:
    from src.modules.emergency_agent import EmergencyAgent

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

Axis = Literal["emergency", "mobility", "environment", "energy"]

#axe modules & calls
_EMERGENCY_AGENT_INSTANCE = EmergencyAgent()

def emergency_agent(e: CityEvent) -> dict:
    """Delegates incoming event dictionary to the Emergency Agent module."""
    return _EMERGENCY_AGENT_INSTANCE.process_event(e.model_dump())

# Lower number = wins when two axes want the same resource.
PRIORITY: dict[str, int] = {"emergency": 0, "environment": 1, "mobility": 2, "energy": 3}
ROUTE_THRESHOLD = 0.5  # score needed for an axis to be activated
SAFETY_FLOOR = 0.9     # a rule scoring >= this can never be downgraded by the LLM

LLM_MODEL = os.getenv("CITY_BRAIN_MODEL", "gemini-3.5-flash-lite")
log = logging.getLogger("city_brain")


# --------------------------------------------------------------------------- #
# Input model: everything entering the Brain is normalized to a CityEvent
# --------------------------------------------------------------------------- #
class CityEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    source: str                      # "sensor" | "operator" | "twin" | "guardian"
    kind: str                        # "air_quality", "traffic", "power_load", "fire_alarm", "text"...
    ts: float = Field(default_factory=time.time)
    lat: float | None = None
    lon: float | None = None
    data: dict = Field(default_factory=dict)   # raw measurements
    text: str | None = None                    # free text from an operator


# --------------------------------------------------------------------------- #
# 1) Classification: each rule returns (axis, score 0..1, reason)
# --------------------------------------------------------------------------- #
Score = tuple[str, float, str]

EMERGENCY_KINDS = {"fire_alarm", "accident", "medical", "flood", "gas_leak", "collapse"}
EMERGENCY_WORDS = ("fire", "explosion", "injured", "ambulance", "collapse", "flood", "gas leak")


def rule_emergency(e: CityEvent) -> list[Score]:
    out: list[Score] = []
    if e.kind in EMERGENCY_KINDS:
        out.append(("emergency", 1.0, f"kind={e.kind}"))
    if e.text and any(w in e.text.lower() for w in EMERGENCY_WORDS):
        out.append(("emergency", 0.9, "emergency keyword in text"))
    if e.data.get("severity", 0) >= 4:
        out.append(("emergency", 0.8, f"severity={e.data['severity']}"))

    # Catastrophic Hazard extensions
    if e.kind == "seismic_catalog":
        mag = float(e.data.get("magnitude", 0))
        if mag >= 6.0 or (mag >= 6.5 and e.data.get("offshore")):
            out.append(("emergency", 0.95, f"major seismic catalog mag={mag}"))
        elif mag >= 5.0:
            out.append(("emergency", 0.75, f"moderate seismic catalog mag={mag}"))
        elif mag >= 4.5:
            out.append(("emergency", 0.60, f"minor seismic catalog mag={mag}"))

    elif e.kind == "hydro_observation":
        river = e.data.get("river_level_m")
        stage = e.data.get("flood_stage_m")
        if river is not None and stage is not None and river >= stage:
            out.append(("emergency", 0.85, f"river level {river}m >= flood stage {stage}m"))
        elif e.data.get("drainage_load", 0) > 0.8:
            out.append(("emergency", 0.60, f"high drainage load={e.data['drainage_load']}"))

    elif e.kind == "weather_observation":
        if e.data.get("radar_rotation"):
            out.append(("emergency", 0.95, "radar rotation detected (tornado/severe convective vortex)"))
        elif e.data.get("gust_kmh", 0) >= 105 or e.data.get("wind_speed_kmh", 0) >= 90:
            out.append(("emergency", 0.90, f"critical wind gust={e.data.get('gust_kmh', 0)} km/h"))
        elif e.data.get("gust_kmh", 0) >= 75:
            out.append(("emergency", 0.65, f"warning wind gust={e.data.get('gust_kmh', 0)} km/h"))
        elif e.data.get("rain_rate_mm_h", 0) >= 25:
            out.append(("emergency", 0.60, f"heavy rain rate={e.data.get('rain_rate_mm_h', 0)} mm/h"))

    elif e.kind == "weather_forecast":
        r24 = e.data.get("rain_24h_forecast_mm", 0)
        if r24 >= 80:
            out.append(("emergency", 0.70, f"heavy rain forecast 24h={r24} mm"))
        elif r24 >= 40:
            out.append(("emergency", 0.55, f"moderate rain forecast 24h={r24} mm"))

    elif e.kind == "official_weather_alert":
        lvl = str(e.data.get("alert_level", "")).lower()
        if lvl in {"red", "critical", "extreme"} or "tornado" in str(e.data.get("alert_type", "")).lower():
            out.append(("emergency", 0.90, f"official weather alert level={lvl}"))
        else:
            out.append(("emergency", 0.70, f"official weather alert level={lvl}"))

    elif e.kind == "sea_level_observation":
        chg = abs(e.data.get("change_m_per_min", 0))
        sl = abs(e.data.get("sea_level_m", 0) - e.data.get("baseline_m", 0))
        if chg >= 0.3 or sl >= 0.3:
            out.append(("emergency", 0.80, "sea level anomaly detected"))
        else:
            out.append(("emergency", 0.50, "routine sea level observation"))

    return out


def rule_environment(e: CityEvent) -> list[Score]:
    d, out = e.data, []
    pm25 = d.get("pm25")
    if pm25 is not None:
        if pm25 > 150:
            out.append(("environment", 0.95, f"pm25={pm25}"))
        elif pm25 > 75:
            out.append(("environment", 0.8, f"pm25={pm25}"))
        elif pm25 > 35:
            out.append(("environment", 0.55, f"pm25={pm25}"))
    if d.get("noise_db", 0) > 85:
        out.append(("environment", 0.6, f"noise_db={d['noise_db']}"))
    if d.get("waste_fill_pct", 0) > 90:
        out.append(("environment", 0.5, f"waste_fill={d['waste_fill_pct']}%"))
    return out


def rule_mobility(e: CityEvent) -> list[Score]:
    d, out = e.data, []
    if d.get("congestion_index", 0) > 0.7:
        out.append(("mobility", 0.7, f"congestion={d['congestion_index']}"))
    if 0 < d.get("avg_speed_kmh", 999) < 15:
        out.append(("mobility", 0.6, f"avg_speed={d['avg_speed_kmh']}"))
    if e.kind in {"road_closure", "transit_delay"}:
        out.append(("mobility", 0.7, f"kind={e.kind}"))
    return out


def rule_energy(e: CityEvent) -> list[Score]:
    d, out = e.data, []
    if d.get("load_pct", 0) > 85:
        out.append(("energy", 0.7, f"load={d['load_pct']}%"))
    if e.kind in {"outage", "grid_fault", "blackout"}:
        out.append(("energy", 0.9, f"kind={e.kind}"))
    return out


def rule_cascades(e: CityEvent) -> list[Score]:
    out: list[Score] = []
    if e.kind in {"accident", "fire_alarm", "flood"}:
        out.append(("mobility", 0.6, "emergency needs route clearing / rerouting"))
    if e.kind == "gas_leak":
        out.append(("environment", 0.7, "gas leak -> air quality risk"))
    if e.kind in {"outage", "blackout"} and e.data.get("hospital_affected"):
        out.append(("emergency", 0.9, "outage affects a hospital"))

    # Catastrophic Hazard Cascades
    if e.kind in {"seismic_catalog", "seismic_sensor", "tsunami"}:
        out.append(("mobility", 0.8, "seismic/tsunami hazard -> bridge and coastal transit restriction"))
        out.append(("energy", 0.7, "seismic event -> grid protection and substation check"))
        out.append(("environment", 0.6, "seismic impact -> industrial/chemical release risk"))

    if e.kind in {"hydro_observation", "flood"} or (e.kind == "weather_forecast" and e.data.get("rain_24h_forecast_mm", 0) >= 60):
        out.append(("mobility", 0.7, "flood risk -> underpass and arterial road closures"))
        out.append(("energy", 0.65, "flood risk -> substation inundation protection"))
        out.append(("environment", 0.6, "flood inundation -> water contamination risk"))

    if e.kind in {"weather_observation", "official_weather_alert"} and (e.data.get("gust_kmh", 0) >= 75 or e.data.get("radar_rotation")):
        out.append(("energy", 0.8, "severe wind/storm -> power line outage & tree fall risk"))
        out.append(("mobility", 0.7, "severe storm -> transit suspension & road clearing"))

    return out


RULES: list[Callable[[CityEvent], list[Score]]] = [
    rule_emergency, rule_environment, rule_mobility, rule_energy, rule_cascades,
]


# --------------------------------------------------------------------------- #
# 1b) Gemini LLM router
# --------------------------------------------------------------------------- #
ROUTE_DECLARATION = types.FunctionDeclaration(
    name="route_event",
    description="Decide which city axes must handle this event.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "summary": types.Schema(
                type=types.Type.STRING,
                description="One-sentence analysis of the event."
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

ROUTER_SYSTEM = """You are the routing brain of a smart city. You receive ONE incoming event
and decide which specialist agents must handle it:
- emergency: fires, accidents, injuries, floods, gas leaks, life/safety threats
- mobility: traffic, congestion, road closures, public transport
- environment: air/water quality, pollution, noise, waste
- energy: power load, outages, grid faults
Rules:
- Route to EVERY axis that must act.
- Score 0..1 how strongly each axis must act. Omit irrelevant axes.
- If data is normal and nothing needs action, return an empty routes list."""


def gemini_router(e: CityEvent) -> dict:
    client = genai.Client()
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents="Incoming event:\n" + e.model_dump_json(),
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
# 2) Domain Agents
# --------------------------------------------------------------------------- #
def emergency_agent(e: CityEvent) -> dict:
    return _EMERGENCY_AGENT_INSTANCE.process_event(e.model_dump())


def mobility_agent(e: CityEvent) -> dict:
    return {
        "severity": 3, "confidence": 0.75,
        "actions": [
            {"type": "reroute", "resource": "road:corridor", "detail": "divert traffic around the zone"},
            {"type": "signal_priority", "resource": "signals:corridor", "detail": "green wave for responders"},
        ],
        "evidence": [f"congestion={e.data.get('congestion_index')}"],
    }


def environment_agent(e: CityEvent) -> dict:
    return {
        "severity": 3, "confidence": 0.8,
        "actions": [
            {"type": "advisory", "resource": "channel:public", "detail": "air quality advisory for the district"},
            {"type": "restrict", "resource": "road:corridor", "detail": "limit heavy vehicles in the area"},
        ],
        "evidence": [f"pm25={e.data.get('pm25')}"],
    }


def energy_agent(e: CityEvent) -> dict:
    return {
        "severity": 2, "confidence": 0.7,
        "actions": [
            {"type": "load_shed", "resource": "feeder:F3", "detail": "shed non-critical load"},
        ],
        "evidence": [f"load={e.data.get('load_pct')}%"],
    }


AGENTS: dict[str, Callable[[CityEvent], dict]] = {
    "emergency": emergency_agent,
    "mobility": mobility_agent,
    "environment": environment_agent,
    "energy": energy_agent,
}


# --------------------------------------------------------------------------- #
# 3) LangGraph Structure
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


def _llm_scores(e: CityEvent) -> tuple[dict[str, dict], str]:
    raw = LLM_ROUTER(e)
    out: dict[str, dict] = {}
    for r in raw.get("routes", []):
        axis = r.get("axis")
        if axis not in PRIORITY:
            continue
        score = min(max(float(r.get("score", 0)), 0.0), 1.0)
        b = out.setdefault(axis, {"score": 0.0, "reasons": []})
        b["score"] = max(b["score"], score)
        b["reasons"].append(str(r.get("reason", "")))
    return out, str(raw.get("summary", ""))


def classify(state: State) -> dict:
    e = CityEvent(**state["event"])
    rules = _rule_scores(e)
    llm: dict[str, dict] | None = None
    analysis = "rule-based routing (LLM unavailable)"
    if LLM_ROUTER:
        try:
            llm, analysis = _llm_scores(e)
        except Exception as exc:
            log.warning("Gemini LLM routing failed, using rules: %s", exc)
            analysis = f"rule-based routing (Gemini LLM failed: {type(exc).__name__})"

    merged: list[dict] = []
    for axis in PRIORITY:
        r, l = rules.get(axis), (llm or {}).get(axis)
        score, reasons, srcs = 0.0, [], []
        if l:
            score, srcs = l["score"], ["llm"]
            reasons += [f"llm: {x}" for x in l["reasons"]]
        if r and (llm is None or r["score"] >= SAFETY_FLOOR):
            score = max(score, r["score"])
            srcs.append("rules")
            reasons += [f"rule: {x}" for x in r["reasons"]]
        if score > 0:
            merged.append({"axis": axis, "score": round(score, 2),
                           "source": "+".join(srcs), "reasons": reasons})
    ranked = sorted(merged, key=lambda x: (-x["score"], PRIORITY[x["axis"]]))
    routes = [b["axis"] for b in ranked if b["score"] >= ROUTE_THRESHOLD]
    return {"scores": ranked, "routes": routes, "analysis": analysis}


def dispatch(state: State):
    if not state["routes"]:
        return ["synthesize"]
    return [Send("run_agent", {"event": state["event"], "axis": a}) for a in state["routes"]]


def run_agent(payload: dict) -> dict:
    axis, event = payload["axis"], CityEvent(**payload["event"])
    started = time.time()
    try:
        res = AGENTS[axis](event)
        res.update(axis=axis, status="ok")
    except Exception as exc:
        res = {"axis": axis, "status": "failed", "error": str(exc),
               "severity": 0, "confidence": 0.0, "actions": [], "evidence": []}
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
                conflicts.append({"resource": a["resource"], "kept": owner,
                                  "dropped": r["axis"], "action": a["type"]})
    ok = [r for r in results if r["status"] == "ok"]
    decision = {
        "primary_axis": ok[0]["axis"] if ok else None,
        "activated_axes": [r["axis"] for r in results],
        "failed_axes": [r["axis"] for r in results if r["status"] == "failed"],
        "actions": final,
        "conflicts": conflicts,
        "evidence": [ev for r in ok for ev in r["evidence"]],
        "analysis": state["analysis"],
        "why": state["scores"],
    }
    return {"decision": decision}


class AuditLog:
    def __init__(self, path: str = "city_brain_audit.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
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
        log.info("DISPATCHING TO AXIS [%s]: %s -> %s", target_axis.upper(), action['type'], action['detail'])


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
        {"source": "sensor", "kind": "accident", "lat": 36.80, "lon": 10.18,
         "data": {"severity": 4, "congestion_index": 0.8}},
        {"source": "sensor", "kind": "air_quality", "data": {"pm25": 120}},
        {"source": "sensor", "kind": "outage", "data": {"load_pct": 40, "hospital_affected": True}},
        {"source": "sensor", "kind": "traffic", "data": {"congestion_index": 0.3}},
    ]
    for d in demos:
        print(d["kind"], "->", json.dumps(decide(d), indent=2)[:600], "\n")