# City Brain Layer: Interface Specification

This document details the interface contracts for the City Brain Layer, including normalized `CityEvent` schemas, the catastrophic hazard extensions, the `HazardAssessment` contract, and downstream `EmergencyAgentResponse` schemas.

---

## 1. Input Contract: Normalized `CityEvent`

All incoming events to the City Brain orchestrator conform to the `CityEvent` Pydantic model:

```json
{
  "id": "evt-unique-id",
  "source": "sensor | operator | twin | guardian | feed",
  "kind": "string",
  "ts": 1730826000.0,
  "lat": 36.8065,
  "lon": 10.1815,
  "data": {},
  "text": "Optional natural language text"
}
```

### Standard Event Payloads (`data`)

#### A. Hydrological Observation (`hydro_observation`)
Fluvial and urban drainage telemetry.
```json
{
  "river_level_m": 4.80,
  "flood_stage_m": 4.50,
  "rate_of_rise_m_h": 0.25,
  "reservoir_level_ratio": 0.85,
  "drainage_load": 0.90
}
```

#### B. Weather Observation (`weather_observation`)
Surface meteorological station and Doppler radar telemetry.
```json
{
  "rain_rate_mm_h": 28.5,
  "wind_speed_kmh": 65.0,
  "gust_kmh": 95.0,
  "pressure_hpa": 998.2,
  "rain_24h_mm": 55.0,
  "pressure_drop_hpa_3h": 5.8,
  "soil_saturation": 0.92,
  "radar_rotation": false,
  "hail": false,
  "lightning_rate": 22.0
}
```

#### C. Weather Forecast (`weather_forecast`)
Numerical weather prediction model outputs (e.g., Open-Meteo).
```json
{
  "rain_24h_forecast_mm": 95.0,
  "horizon_h": 24,
  "rain_6h_forecast_mm": 35.0,
  "gust_forecast_kmh": 85.0
}
```

#### D. Official Weather Alert (`official_weather_alert`)
Civil protection and national meteorological institute bulletins (e.g., INM).
```json
{
  "alert_type": "severe_storm",
  "alert_level": "red",
  "valid_until": 1730833200.0,
  "issuer": "INM Civil Protection",
  "description": "Destructive wind squalls and heavy rainfall expected."
}
```

#### E. Seismic Catalog (`seismic_catalog`)
Regional and global earthquake catalog solutions (USGS, EMSC).
```json
{
  "magnitude": 6.8,
  "depth_km": 12.0,
  "epicenter_lat": 37.50,
  "epicenter_lon": 10.40,
  "event_time": 1730825900.0,
  "offshore": true,
  "catalog_source": "emsc"
}
```

#### F. Sea Level Observation (`sea_level_observation`)
Tide gauge and coastal sea-level radar stations.
```json
{
  "station_id": "TUN-TG-01",
  "sea_level_m": 1.65,
  "baseline_m": 0.85,
  "change_m_per_min": 0.45
}
```

#### G. Seismic Waveform Sensor (`seismic_sensor`)
High-frequency ground motion telemetry processed by the PLUM engine.
```json
{
  "rawWaveformTelemetry": {
    "samplingRateHz": 100,
    "components": {
      "vertical_z": [0.05, 0.45, 1.25],
      "east_west_e": [0.08, -0.35, 0.95],
      "north_south_n": [-0.03, 0.25, 0.85]
    },
    "units": "m/s^2",
    "clippingDetected": false,
    "noiseFloorExceeded": false
  }
}
```

#### H. Acoustic Triangulation & Hotspot (`acoustic_gunshot_triangulation`, `crime_hotspot`)
Acoustic sensors and historical risk-terrain context.
```json
{
  "decibelLevel": 118.2,
  "associatedContext": {
    "dist_to_police_station_m": 2500.0,
    "dist_to_hospital_m": 1800.0,
    "historical_crime_count": 28,
    "is_weekend": true,
    "is_night": true,
    "street_lighting_status": "malfunctioning"
  }
}
```

---

## 2. Hazard Assessment Contract (`HazardAssessment`)

Every hazard engine (`FloodEngine`, `CatalogQuakeEngine`, `TsunamiEngine`, `StormEngine`) produces a standardized `HazardAssessment` object:

```json
{
  "hazard": "flood | earthquake | tsunami | storm",
  "risk_level": "low | watch | warning | critical",
  "risk_score": 0.75,
  "confidence": 0.92,
  "drivers": [
    {"factor": "rain_24h_forecast_mm", "value": 95.0, "contribution": 0.35},
    {"factor": "soil_saturation", "value": 0.92, "contribution": 0.20}
  ],
  "affected_area": {
    "center": {"lat": 36.8065, "lon": 10.1815},
    "radius_km": 15.0
  },
  "eta_minutes": 120,
  "independent_source_count": 3,
  "needs_human_confirmation": true,
  "evidence": [
    "Flood composite risk index: 0.75 -> CRITICAL (active factors: 5/5, confidence: 0.92)"
  ],
  "metadata": {}
}
```

> **Note on Risk Score**: The `risk_score` is a normalized 0.0 - 1.0 composite risk index; it is strictly **not a probability**.

---

## 3. Emergency Agent Output Contract (`EmergencyAgentResponse`)

```json
{
  "axis": "emergency",
  "severity": 5,
  "confidence": 0.92,
  "status": "ok",
  "plumAlertTriggered": false,
  "peakMMI": 0.0,
  "hazard": "flood",
  "risk_level": "critical",
  "risk_score": 0.75,
  "requires_human_approval": true,
  "actions": [
    {
      "type": "evacuate",
      "resource": "channel:cap",
      "detail": "CRITICAL: Immediate evacuation advisory for inundated low-lying flood sectors.",
      "priority": 0,
      "requires_human_approval": true
    },
    {
      "type": "dispatch_units",
      "resource": "units:nearest",
      "detail": "Deploy swift-water rescue teams and emergency first responders to designated flood zones.",
      "priority": 0,
      "requires_human_approval": true
    },
    {
      "type": "alert_cap",
      "resource": "channel:cap",
      "detail": "CAP 1.2 payload generated (FLOOD CRITICAL).",
      "priority": 0,
      "requires_human_approval": true
    }
  ],
  "evidence": [
    "Flood composite risk index: 0.75 -> CRITICAL"
  ]
}
```

---

## 4. Safety & Invariant Rules

1. **Deterministic Execution**: All hazard detection logic is pure Python with zero LLM in the detection path.
2. **Safety Floor**: Deterministic classification rules scoring ≥ 0.90 can never be downgraded by the Gemini LLM router.
3. **Critical Cap Rule (Section 7.6)**: An assessment cannot reach `critical` unless confirmed by at least `min_sources_for_critical` (default 2) independent sources. Otherwise, it is automatically capped at `warning`.
4. **Human Approval Invariant**: All actions at `critical` level and all actions involving `evacuate` have `requires_human_approval = True`.

