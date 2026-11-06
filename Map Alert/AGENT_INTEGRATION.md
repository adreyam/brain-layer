# Agent integration

Agents post JSON to POST /api/v1/incidents. The frontend is not part of the integration contract.

Required fields: title, domain, severity, latitude, longitude. source is recommended. external_id is an optional unique idempotency key; duplicate values return HTTP 409. Optional description, status, occurred_at and metadata carry context. Use WGS84 coordinates and timezone-aware ISO 8601 timestamps. Unknown top-level fields and invalid coordinate ranges are rejected.

Example:

    {
      "title":"Elevated PM2.5",
      "description":"Reading exceeded the threshold",
      "domain":"environment",
      "severity":"high",
      "latitude":36.812,
      "longitude":10.1655,
      "source":"environment-agent",
      "external_id":"sensor-42-event-19",
      "metadata":{"pollutant":"PM2.5","value":118,"unit":"ug/m3"}
    }

Python agents can use requests.post("http://localhost:8000/api/v1/incidents", json=payload, timeout=5) and call raise_for_status() on the response.

Allowed domains: security, mobility, energy, environment, emergency, disaster. Severity: low, medium, high, critical. Status: active, investigating, resolved.

WebSocket clients connect to ws://localhost:8000/api/v1/ws. Server events are JSON envelopes with type incident.created or incident.updated and an incident object. Update status or severity using PATCH /api/v1/incidents/{id}.

Authentication is not part of this MVP. Add per-agent credentials and authorization before exposing ingestion outside a trusted network.
