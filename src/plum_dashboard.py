"""
PLUM Algorithm Testing Dashboard Server.

Launches a lightweight local server that hosts the interactive PLUM test dashboard
and provides live API hooks into PLUMEngine and EmergencyAgent.

Usage:
    python src/plum_dashboard.py
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import webbrowser
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DASHBOARD_FILE = PROJECT_ROOT / "dashboard" / "plum_dashboard.html"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.plum_engine import PLUMEngine, WaveformTelemetry
from modules.emergency_agent import EmergencyAgent

PORT = 8080


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html", "/plum_dashboard.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if DASHBOARD_FILE.exists():
                with open(DASHBOARD_FILE, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<h1>Dashboard file not found</h1>")
            return
        elif self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "engine": "PLUMEngine"}).encode())
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/evaluate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                telemetry = WaveformTelemetry(**data.get("rawWaveformTelemetry", data))
                pga, mmi, alert, evidence = PLUMEngine.process_telemetry(telemetry)

                # Also run through EmergencyAgent
                event = {
                    "kind": "seismic_sensor",
                    "source": "dashboard_test",
                    "lat": 36.8021,
                    "lon": 10.1815,
                    "data": {"rawWaveformTelemetry": telemetry.model_dump()},
                }
                agent = EmergencyAgent()
                agent_resp = agent.process_event(event)

                resp = {
                    "status": "ok",
                    "pga": pga,
                    "mmi": mmi,
                    "alert_triggered": alert,
                    "evidence": evidence,
                    "agent_response": agent_resp,
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode())
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())
            return
        self.send_response(404)
        self.end_headers()


def start_server():
    os.chdir(str(PROJECT_ROOT))
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print("=" * 60)
        print("PLUM ALGORITHM TESTING DASHBOARD")
        print(f"Server running at: {url}")
        print("Open this URL in your browser to test the PLUM algorithm interactively.")
        print("Press Ctrl+C to stop.")
        print("=" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    start_server()
