# City Brain Layer

> An AI-powered smart-city orchestration platform that ingests urban events (sensor telemetry, operator reports, digital-twin feeds, external observation networks) and coordinates multi-axis response across **Emergency**, **Mobility**, **Environment**, and **Energy** domains.

---

## Architecture Overview

```
                                      ┌───────────────┐
                         CityEvent    │  Orchestrator │
       sensors ─────────────────────► │  (LangGraph)  │
       operators ───────────────────► │               │
       digital twin ────────────────► └───────┬───────┘
       external feeds (Meteo/USGS/..)         │
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
               ┌────────────┐          ┌─────────────┐         ┌─────────────┐
               │  classify  │          │    ...      │         │    ...      │
               │  (rules +  │          └─────────────┘         └─────────────┘
               │   Gemini)  │
               └──────┬─────┘
                      │ parallel fan-out
        ┌─────────────┼──────────────┬──────────────┐
        ▼             ▼              ▼              ▼
    Emergency      Mobility     Environment      Energy
      Agent         Agent          Agent          Agent
   ┌────┴─────┐
   │ Engines: │
   │ - PLUM   │
   │ - Crime  │
   │ - Flood  │
   │ - Quake  │
   │ - Tsunami│
   │ - Storm  │
   └──────────┘
        │
        ▼
   ┌──────────┐    ┌──────────┐
   │synthesize│───►│  audit   │──► SQLite + publish
   │(conflict │    │(AuditLog)│
   │resolution)    └──────────┘
   └──────────┘
```

### Processing Pipeline

1. **Classify** — Every incoming `CityEvent` is scored by deterministic rules *and* optionally by a **Gemini LLM router** (function-calling). A safety-floor mechanism ensures critical rule scores (≥ 0.9) can never be downgraded by the LLM.
2. **Dispatch** — Axes whose score exceeds the route threshold (0.5) receive the event in parallel via LangGraph's `Send` fan-out.
3. **Run Agents** — Domain-specific agents produce severity, confidence, action lists, and evidence chains.
4. **Synthesize** — A priority-based conflict resolver (`emergency:0 > environment:1 > mobility:2 > energy:3`) deduplicates resource claims across axes.
5. **Audit & Publish** — The final decision is persisted in SQLite and dispatched to downstream axis controllers.

---

## Repository Structure

```
brain-layer/
├── README.md                          # Platform documentation
├── INTERFACE.md                       # Complete API and Event payload schemas
├── requirements.txt                   # Dependency manifest
├── pytest.ini                         # Pytest configuration
├── city_brain_audit.db                # SQLite audit log (runtime artifact)
│
├── config/
│   └── hazard_thresholds.yaml         # Configurable hazard thresholds and cutoffs
│
├── src/
│   ├── __init__.py
│   ├── orchestrator.py                # LangGraph state machine, Gemini router,
│   │                                  #   classification rules, cascade logic, audit
│   └── modules/
│       ├── __init__.py
│       ├── emergency_agent.py         # Emergency Axis Agent (dispatch, playbooks)
│       ├── hazard_common.py           # HazardAssessment model, config loader, helpers
│       ├── hazard_playbooks.py        # Deterministic playbooks, CAP 1.2 generator, optional LLM planner
│       ├── plum_engine.py             # PLUM seismic waveform early-warning engine
│       ├── crime_hotspot.py           # Crime hotspot risk scoring engine
│       ├── hotspot_model.py           # Violent-crime hotspot ML pipeline (250 m grid)
│       ├── flood_engine.py            # Hydrological and pluvial flood hazard engine
│       ├── catalog_quake_engine.py    # Catalog earthquake engine (USGS/EMSC)
│       ├── tsunami_engine.py          # Tsunami candidate & sea-level evaluation engine
│       ├── storm_engine.py            # Storm nowcasting and severe convective engine
│       └── feeds/                     # External data feed adapters (offline-capable)
│           ├── __init__.py
│           ├── open_meteo.py          # Open-Meteo forecast and live weather adapter
│           ├── usgs.py                # USGS earthquake catalog adapter
│           ├── emsc.py                # EMSC Mediterranean seismic adapter
│           ├── glofas.py              # Copernicus GloFAS river discharge stub
│           └── weather_alert.py       # Official meteorological alert stub
│
├── mocks/
│   ├── __init__.py                    # Re-exports all mock fixtures
│   ├── mock_plum.py / .json           # PLUM seismic waveform scenarios
│   ├── mock_crime_hotspot.py / .json  # Crime Hotspot scenarios
│   ├── mock_flood.py / .json          # Flood hazard scenarios
│   ├── mock_catalog_quake.py / .json  # Catalog earthquake scenarios
│   ├── mock_tsunami.py / .json        # Tsunami and sea-level anomaly scenarios
│   └── mock_storm.py / .json          # Severe storm & Doppler radar scenarios
│
└── tests/
    ├── cap_01_alert.xml               # CAP 1.2 exercise: initial earthquake alert
    ├── cap_02_escalation.xml          # CAP 1.2 exercise: severity escalation (Update)
    ├── cap_03_all_clear.xml           # CAP 1.2 exercise: all-clear (Update)
    ├── cap_04_cancel.xml              # CAP 1.2 exercise: Guardian revocation (Cancel)
    ├── cap_05_tsunami_warning.xml     # CAP 1.2 exercise: coastal tsunami warning
    ├── test_flood_engine.py           # Pytest unit tests for FloodEngine
    ├── test_catalog_quake_engine.py   # Pytest unit tests for CatalogQuakeEngine
    ├── test_tsunami_engine.py         # Pytest unit tests for TsunamiEngine
    ├── test_storm_engine.py           # Pytest unit tests for StormEngine
    ├── test_emergency_agent_hazards.py# Pytest integration tests for EmergencyAgent
    └── test_orchestrator_hazard_rules.py# Pytest tests for orchestrator routing
```

---

## Core Emergency Engines

| Engine | File | Method & Description |
|---|---|---|
| **PLUM Engine** | `src/modules/plum_engine.py` | Deterministic 2-station seismic ground-motion detection (Kilb et al., 2021; Worden et al., 2012) |
| **Crime Hotspot Engine** | `src/modules/crime_hotspot.py` | Multi-factor risk terrain scoring (Alhammadi, 2025) |
| **Flood Engine** | `src/modules/flood_engine.py` | Weighted composite risk index combining rainfall forecast, soil saturation, river stage ratio, and drainage load |
| **Catalog Quake Engine** | `src/modules/catalog_quake_engine.py` | Magnitude/depth/distance attenuation analysis from USGS/EMSC feeds; feeds into Tsunami Engine |
| **Tsunami Engine** | `src/modules/tsunami_engine.py` | Offshore marine candidate filtering, wave propagation ETA, and tide gauge sea-level anomaly confirmation |
| **Storm Engine** | `src/modules/storm_engine.py` | High-gust detection, rapid barometric drop, convective rotation signatures, and official weather alerts |

---

## Key Invariants

1. **Deterministic Hazard Detection**: No LLM is in the detection path; all hazard evaluations are pure, explainable, and reproducible logic.
2. **Risk Index, Not Probability**: All engine outputs represent normalized composite risk scores [0.0 - 1.0], never probabilities.
3. **Critical Source Agreement (Section 7.6)**: An assessment cannot reach `critical` without at least `min_sources_for_critical` (default 2) independent sources. Otherwise, it is safely capped at `warning`.
4. **Human Approval Safeguard**: Evacuation directives and all `critical` actions automatically flag `requires_human_approval = True`.
5. **Offline Operation**: System operates fully with `CITY_BRAIN_LLM=off` or without an API key; optional LLM planners fail silently to deterministic playbooks.

---

## Quick Start & Verification

### 1. Run the Pytest Test Suite
```bash
# Run all 25 unit and integration tests offline
pytest
```

### 2. Run Self-Contained Mock Verifications
```bash
python mocks/mock_flood.py
python mocks/mock_catalog_quake.py
python mocks/mock_tsunami.py
python mocks/mock_storm.py
python mocks/mock_plum.py
python mocks/mock_crime_hotspot.py
```

### 3. Run the Orchestrator Demo
```bash
python src/orchestrator.py
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `CITY_BRAIN_HAZARD_CONFIG` | `config/hazard_thresholds.yaml` | Path to hazard thresholds YAML file |
| `CITY_BRAIN_LLM` | `on` | Set to `off` to disable Gemini routing and LLM planning |
| `CITY_BRAIN_MODEL` | `gemini-3.5-flash-lite` | Gemini model name |
| `GEMINI_API_KEY` | None | Google GenAI API key for optional LLM router and planner |
