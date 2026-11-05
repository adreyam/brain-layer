"""
City Brain Control Room server.

Serves the dashboard and runs every scenario through the real orchestrator
(rules + Laya System 1 + Gemini System 2 fallback), logging each decision to SQLite.

API
- GET  /                 -> dashboard/city_brain_dashboard.html
- GET  /api/health       -> engine status (Laya, Gemini) and which axes are integrated
- GET  /api/city         -> city model: blocks, facilities, tunnels, bins, truck routes
- GET  /api/blocks       -> blocks only (kept for older clients)
- GET  /api/scenarios    -> scenario catalogue for the top bar
- GET  /api/stats        -> aggregate statistics from city_brain_audit.db
- GET  /api/history      -> last 30 decisions from city_brain_audit.db
- POST /api/scenario     -> {"scenario": "...", "block": "D5", "bin_id": "..."} -> builds the
                            sensor event, runs decide(), returns event + decision + map target
- POST /api/decide       -> raw CityEvent JSON -> decide()

Usage:
    python src/city_brain_dashboard.py
"""
from __future__ import annotations

import http.server
import json
import logging
import sqlite3
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DASHBOARD_FILE = PROJECT_ROOT / "dashboard" / "city_brain_dashboard.html"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import orchestrator  # noqa: E402
from orchestrator import AuditLog, decide, PRIORITY, SAFETY_FLOOR  # noqa: E402
import city_model  # noqa: E402

PORT = 8085
DB_PATH = AuditLog.DEFAULT_PATH
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("city_brain_server")
_DECIDE_LOCK = threading.Lock()  # one decision at a time (shared SQLite + model state)


def _db() -> sqlite3.Connection | None:
    return sqlite3.connect(DB_PATH) if Path(DB_PATH).exists() else None


def health() -> dict:
    return {
        "status": "online",
        "laya_loaded": orchestrator.LAYA_AVAILABLE,
        "gemini_configured": orchestrator.LLM_ROUTER is not None,
        "gemini_model": orchestrator.LLM_MODEL,
        "axes": {a: ("integrated" if a in orchestrator.INTEGRATED_AXES else "not_integrated") for a in PRIORITY},
        "blocks": len(city_model.BLOCKS),
        "database": Path(DB_PATH).name,
    }


def stats() -> dict:
    out = {"total_events": 0, "safety_floor_triggers": 0, "conflicts_resolved": 0,
           "awaiting_axes": 0, "primary_axis_counts": {a: 0 for a in PRIORITY}}
    conn = _db()
    if not conn:
        return out
    try:
        rows = conn.execute("SELECT decision FROM decisions").fetchall()
        out["total_events"] = len(rows)
        for (raw,) in rows:
            try:
                dec = json.loads(raw)
            except Exception:
                continue
            if dec.get("primary_axis") in out["primary_axis_counts"]:
                out["primary_axis_counts"][dec["primary_axis"]] += 1
            out["conflicts_resolved"] += len(dec.get("conflicts") or [])
            if dec.get("unavailable_axes"):
                out["awaiting_axes"] += 1
            # a deterministic rule scored at or above the safety floor
            if any(w.get("score", 0) >= SAFETY_FLOOR and "rules" in w.get("source", "") for w in dec.get("why", [])):
                out["safety_floor_triggers"] += 1
    finally:
        conn.close()
    return out


def slim_event(event: dict) -> dict:
    """Drop raw waveforms (hundreds of KB) before sending an event to the browser.
    The audit DB keeps the full event."""
    net = (event.get("data") or {}).get("stationNetwork")
    if not net:
        return event
    data = dict(event["data"])
    data["stationNetwork"] = {k: v for k, v in net.items() if k != "stations"}
    data["stationNetwork"]["stations"] = [{k: v for k, v in st.items() if k != "accel"} for st in net.get("stations", [])]
    data["stationNetwork"]["waveforms_omitted"] = True
    return {**event, "data": data}


def history(limit: int = 30) -> list[dict]:
    conn = _db()
    if not conn:
        return []
    try:
        rows = conn.execute(
            "SELECT event_id, ts, event, decision FROM decisions ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"event_id": r[0], "ts": r[1], "event": slim_event(json.loads(r[2])), "decision": json.loads(r[3])} for r in rows]
    except Exception as exc:
        log.warning("Could not read audit db: %s", exc)
        return []
    finally:
        conn.close()


def run_decide(event: dict) -> tuple[dict, dict, float]:
    with _DECIDE_LOCK:
        ev = orchestrator.CityEvent(**event)
        t0 = time.time()
        decision = decide(ev)
        return ev.model_dump(), decision, round((time.time() - t0) * 1000, 2)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        log.debug(fmt, *args)

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/city_brain_dashboard.html"):
            if not DASHBOARD_FILE.exists():
                return self._json({"error": "dashboard file not found"}, 404)
            body = DASHBOARD_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/health":
            self._json(health())
        elif path == "/api/city":
            self._json(city_model.city_payload())
        elif path == "/api/blocks":
            self._json(city_model.BLOCKS)
        elif path == "/api/scenarios":
            self._json(city_model.SCENARIOS)
        elif path == "/api/stats":
            self._json(stats())
        elif path == "/api/history":
            self._json(history())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/scenario":
                event, target = city_model.build_scenario(body.get("scenario"), body.get("block"), body.get("bin_id"),
                                                       body.get("magnitude"))
                event, decision, ms = run_decide(event)
                return self._json({"scenario": body.get("scenario"), "event": slim_event(event), "target": target,
                                   "decision": decision, "latency_ms": ms})
            if self.path == "/api/decide":
                event, decision, ms = run_decide(body)
                return self._json({"event": slim_event(event), "decision": decision, "latency_ms": ms})
            self._json({"error": "not found"}, 404)
        except Exception as exc:
            log.exception("Request failed")
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def start_server() -> None:
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer(("", PORT), Handler) as httpd:
        h = health()
        print("=" * 64)
        print(" CITY BRAIN CONTROL ROOM")
        print(f" Open        : http://localhost:{PORT}")
        print(f" Laya (S1)   : {'loaded' if h['laya_loaded'] else 'not loaded (rules only)'}")
        print(f" Gemini (S2) : {'configured' if h['gemini_configured'] else 'off (no GEMINI_API_KEY)'}")
        print(f" Axes        : {', '.join(f'{a}={s}' for a, s in h['axes'].items())}")
        print(f" Audit DB    : {DB_PATH}")
        print("=" * 64)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    start_server()
