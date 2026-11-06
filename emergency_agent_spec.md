# City Brain Layer: Catastrophic Hazard Extension (Emergency Axis)

> Implementation spec for the IDE assistant. This **extends the existing City Brain Layer repository**; it does not replace it. Read the existing code first (Section 2), then build only what is described here. Where a value is marked `CONFIG`, put it in a config file, never hardcode it.

## 1. Context

City Brain Layer is a LangGraph orchestrator that ingests `CityEvent`s, classifies them with deterministic rules plus an optional Gemini router, fans out to the Emergency, Mobility, Environment and Energy agents, then synthesizes, audits and publishes the result.

The Emergency agent (`src/modules/emergency_agent.py`) currently handles:

- seismic sensor events through the PLUM engine (`seismic_sensor`)
- crime and acoustic events through the Crime Hotspot engine
- direct triggers: `accident`, `fire_alarm`, `collapse`, `gas_leak`

**This work adds the catastrophic hazards that are missing:** flood, tsunami and severe storm / tornado, plus a catalog-based earthquake path (needed for tsunami). Earthquake sensor detection already exists through PLUM and must be reused, not rebuilt.

### Design decisions (consistent with the existing architecture)

1. **Detection is deterministic.** New hazard engines follow the same pattern as `plum_engine.py`: pure, explainable logic with no LLM in the detection path.
2. **No ML for these hazards.** There is no reliable historical data for them. Output is a **risk index with levels**, never a probability. (`hotspot_model.py` is an ML pipeline because crime has open historical data; the catastrophic hazards do not.)
3. **The LLM is optional and never classifies the hazard.** The existing Gemini router stays a router. Response planning is deterministic playbooks first, with an optional LLM planner behind the existing `CITY_BRAIN_LLM` flag (Section 9).
4. **Log everything** (already done by `AuditLog`) so ML can be added later once real events accumulate.

## 2. Step 0: read the existing code before writing anything

The IDE assistant must read these files and **conform to what is actually there** instead of inventing schemas:

| File | What to extract |
|---|---|
| `src/orchestrator.py` | `CityEvent` fields (especially `kind` and payload structure), `rule_emergency`, `rule_cascades`, route threshold (0.5), safety-floor logic (rule score >= 0.9 cannot be downgraded), priority map, `synthesize`, `AuditLog` schema |
| `src/modules/emergency_agent.py` | `EmergencyAgent` dispatch by `kind`, `EmergencyAgentResponse`, `ActionItem` fields, how CAP-compatible actions are produced, the severity scale used |
| `src/modules/plum_engine.py` | Code style, class/function shape, how results are returned, how the self-test is written |
| `src/modules/crime_hotspot.py` | Pattern for scoring + tier classification with weights and thresholds |
| `mocks/mock_plum.py`, `mocks/mock_crime_hotspot.py`, `mocks/__init__.py` | Mock scenario and verification pattern, JSON payload twin files |
| `tests/cap_0*.xml` | CAP 1.2 structure, parameters (`trust_verdict_id`, `triggered_by`, `lead_time_s`) |

Anything this spec says about field names that conflicts with the real code: **the real code wins**. Report the conflict in a comment and adapt.

## 3. Ground rules

- Do not change the behavior of existing events (`seismic_sensor`, crime, accident, fire_alarm, collapse, gas_leak). Existing mocks and the PLUM self-test must still pass unchanged.
- Add new `kind` values; do not rename existing ones.
- Never present a score as a probability (code, docs, alert text).
- No threshold hardcoded in logic; load from config (Section 12).
- Engines must not call the network. All external data arrives as `CityEvent`s (Section 8 covers optional feed adapters that create events).
- Never crash on bad input: return a structured "rejected / insufficient data" result with a reason.
- Add `__init__.py` files (`src/`, `src/modules/`) and a dependency manifest if still missing.

## 4. Architecture (what changes)

```
sensors / operators / digital twin / external feeds
        |
        v   CityEvent (new kinds: weather_observation, hydro_observation,
        |                          seismic_catalog, sea_level_observation,
        |                          weather_forecast, official_weather_alert)
   Orchestrator.classify
        |   rule_emergency  <-- NEW rules for the kinds above
        |   rule_cascades   <-- NEW cascades (Section 7.3)
        |   Gemini router unchanged (safety floor unchanged)
        v
   Emergency Agent  (dispatch by kind)
        |-- seismic_sensor ---------> PLUM engine            (existing)
        |-- crime kinds ------------> Crime Hotspot engine   (existing)
        |-- hydro / forecast -------> FloodEngine            (NEW)
        |-- seismic_catalog --------> CatalogQuakeEngine     (NEW)
        |-- + sea_level_observation-> TsunamiEngine          (NEW)
        |-- weather / alert --------> StormEngine            (NEW)
        v
   HazardAssessment -> playbook (+ optional LLM planner) -> ActionItems / CAP
        v
   synthesize (existing priority resolver) -> audit -> publish
```

## 5. Files to add or modify

```
src/modules/
  hazard_common.py        # NEW: HazardAssessment model, risk level helpers, config loader
  flood_engine.py         # NEW
  catalog_quake_engine.py # NEW (magnitude/epicenter path, feeds tsunami)
  tsunami_engine.py       # NEW
  storm_engine.py         # NEW
  emergency_agent.py      # MODIFY: add dispatch branches + playbooks
  hazard_playbooks.py     # NEW: deterministic response per hazard and level
  feeds/                  # NEW (optional polling adapters that emit CityEvents)
    open_meteo.py
    usgs.py
    emsc.py
    glofas.py             # stub
src/orchestrator.py       # MODIFY: rule_emergency, rule_cascades only (minimal diff)
config/
  hazard_thresholds.yaml  # NEW
  hazard_playbooks.yaml   # NEW (or keep playbooks in code, see Section 9)
mocks/
  mock_flood.py / .json
  mock_catalog_quake.py / .json
  mock_tsunami.py / .json
  mock_storm.py / .json
  __init__.py             # MODIFY: re-export new fixtures
tests/
  test_flood_engine.py
  test_catalog_quake_engine.py
  test_tsunami_engine.py
  test_storm_engine.py
  test_emergency_agent_hazards.py
  test_orchestrator_hazard_rules.py
  cap_05_tsunami_warning.xml   # optional CAP exercise
```

## 6. Input contract: new `CityEvent` kinds

Use the existing `CityEvent` model. Put hazard data in the existing payload structure (check how `seismic_sensor` does it). Required for every new kind: timestamp (UTC), location (lat/lon), source, and a quality or confidence field if the model has one. If `CityEvent` has no quality field, put `quality` in the payload.

| Kind | Required payload | Optional payload |
|---|---|---|
| `hydro_observation` | `river_level_m`, `flood_stage_m` | `rate_of_rise_m_h`, `reservoir_level_ratio`, `drainage_load` (0..1) |
| `weather_observation` | `rain_rate_mm_h`, `wind_speed_kmh`, `gust_kmh`, `pressure_hpa` | `rain_24h_mm`, `pressure_drop_hpa_3h`, `soil_saturation` (0..1), `radar_rotation` (bool), `hail`, `lightning_rate` |
| `weather_forecast` | `rain_24h_forecast_mm`, `horizon_h` | `rain_6h_forecast_mm`, `gust_forecast_kmh` |
| `official_weather_alert` | `alert_type`, `alert_level`, `valid_until` | `issuer` (for example INM Tunisia) |
| `seismic_catalog` | `magnitude`, `depth_km`, `epicenter_lat`, `epicenter_lon`, `event_time` | `offshore` (bool, else derive from a coastline bounding check), `catalog_source` (`usgs`/`emsc`) |
| `sea_level_observation` | `sea_level_m`, `station_id` | `change_m_per_min`, `baseline_m` |

Canonical units: mm, mm/h, meters, km, km/h, hPa, g. If a payload arrives in other units, convert in a small normalizer function inside `hazard_common.py` and record the conversion in the evidence chain.

## 7. Hazard engines

### 7.1 Common output: `HazardAssessment` (`hazard_common.py`)

Pydantic model returned by every new engine. Keep it close to the fields `EmergencyAgentResponse` and `ActionItem` already use so mapping is trivial.

```json
{
  "hazard": "flood",
  "risk_level": "warning",
  "risk_score": 0.62,
  "confidence": 0.8,
  "drivers": [
    {"factor": "rain_24h_forecast_mm", "value": 85, "contribution": 0.30},
    {"factor": "soil_saturation", "value": 0.9, "contribution": 0.18}
  ],
  "affected_area": {"center": {"lat": 36.8, "lon": 10.18}, "radius_km": 15},
  "eta_minutes": 120,
  "independent_source_count": 2,
  "needs_human_confirmation": true,
  "evidence": ["..."]
}
```

- `risk_level` in `low | watch | warning | critical`.
- `risk_score` is a 0..1 **risk index, not a probability**. State this in the model docstring.
- `eta_minutes` is `null` when unknown (always for earthquakes).
- `drivers` makes every decision explainable and goes into the existing evidence chain.

**Severity mapping:** map `risk_level` to the existing severity scale used by `EmergencyAgentResponse` (the crime engine already maps high-risk to severity 5). Suggested starting point, to be confirmed against the real scale: `low` = 1, `watch` = 2, `warning` = 3 to 4, `critical` = 5. Put the mapping in config.

### 7.2 FloodEngine (`flood_engine.py`)

Weighted risk index. Normalize each factor to 0..1:

```python
score = (
    0.35 * min(rain_24h_forecast_mm / 100, 1) +
    0.20 * min(rain_now_mm_h / 30, 1) +
    0.20 * soil_saturation +
    0.15 * river_level_ratio +      # river_level_m / flood_stage_m
    0.10 * drainage_load
)
# levels: <0.3 low, <0.5 watch, <0.7 warning, else critical   (all CONFIG)
```

- Inputs are assembled from the latest `weather_forecast`, `weather_observation` and `hydro_observation` events near the same area. The engine takes a list of recent events (same pattern as PLUM taking a network of stations).
- Missing optional inputs: redistribute their weight over the available ones and lower `confidence`.
- Stale inputs (older than `CONFIG.max_age_minutes` per kind) are ignored and noted in `evidence`.
- Weights and cutoffs are expert-estimated starting points, not learned.

### 7.3 CatalogQuakeEngine (`catalog_quake_engine.py`)

PLUM remains the **sensor-based** earthquake detector. This engine handles **catalog** events (USGS/EMSC) that carry magnitude and epicenter, which PLUM does not produce and which tsunami detection needs. Earthquakes cannot be predicted; this only reacts.

- Trigger when magnitude >= `CONFIG.eq_min_magnitude` (default 4.5) within `CONFIG.eq_radius_km` of the city.
- Starting severity (CONFIG): `watch` M4.5 to 5.4, `warning` M5.5 to 6.4, `critical` M >= 6.5 nearby. Adjust by distance (attenuate) and depth (shallow is worse).
- If a PLUM event is active at the same time, include it as a second independent source (raises `independent_source_count` and `confidence`).
- Output includes the parsed quake parameters so `TsunamiEngine` can consume them.

### 7.4 TsunamiEngine (`tsunami_engine.py`)

Conditional on an earthquake (from `CatalogQuakeEngine`).

- Candidate if: offshore AND magnitude >= `CONFIG.tsunami_min_magnitude` (default 6.5) AND depth_km <= `CONFIG.tsunami_max_depth_km` (default 70) AND epicenter inside `CONFIG.tsunami_region` (Mediterranean bounding box).
- Level `warning` if candidate only. Level `critical` if candidate AND a `sea_level_observation` shows change above `CONFIG.sea_level_change_m` within `CONFIG.sea_level_window_min`.
- `eta_minutes` = distance to coast / `CONFIG.tsunami_wave_speed_kmh`.
- Always `needs_human_confirmation = true` at `critical`.

### 7.5 StormEngine (`storm_engine.py`)

Short-term (hours) only. Do not claim longer forecasting.

- `warning` if `gust_kmh >= CONFIG.gust_warning_kmh` or `pressure_drop_hpa_3h >= CONFIG.pressure_drop_hpa_3h`.
- `critical` if `radar_rotation` is true, or an `official_weather_alert` of severe-storm or tornado type is active, or gusts >= `CONFIG.gust_critical_kmh`.
- Also add hail and lightning as optional score contributors.

### 7.6 Source agreement and the "critical" cap

The orchestrator already has a Guardian trust verdict (CAP messages carry `trust_verdict_id`) and a conflict resolver. Inspect how the verdict is attached to events and reuse it.

- Count **independent** sources per assessment (two feeds relaying the same upstream data count as one). A positive Guardian trust verdict counts as a confirming source if that matches how it is used today.
- An assessment may not reach `critical` unless `independent_source_count >= CONFIG.min_sources_for_critical` (default 2). Otherwise cap at `warning` and note why in `evidence`.
- This cap is inside the hazard engines. It must not interfere with the orchestrator's safety floor, which protects rule scores from being downgraded by the LLM.

## 8. Orchestrator changes (minimal diff)

Only touch `rule_emergency` and `rule_cascades` in `src/orchestrator.py`.

**`rule_emergency`:** add deterministic scoring for the new kinds, using the same style as existing branches. Starting guidance (CONFIG where numeric):

- `seismic_catalog` with magnitude >= 6.0 nearby, or an offshore candidate tsunami: score >= 0.9
- `hydro_observation` with river level >= flood stage: score >= 0.7
- `weather_observation` with gust >= critical threshold or radar rotation: score >= 0.9
- `official_weather_alert` with severe type: score >= 0.7
- ambiguous inputs (single elevated reading): 0.5 to 0.6 so they still route (threshold 0.5) but are not treated as critical

**`rule_cascades`:** add cross-axis cascades so other agents are engaged:

- earthquake or tsunami -> Mobility (coastal and bridge closures), Energy (grid check), Environment (hazardous release risk)
- flood -> Mobility (road closures), Energy (substation flooding), Environment (water contamination)
- storm -> Energy (outage risk), Mobility (road and transit disruption)

Keep the priority order `emergency > environment > mobility > energy` unchanged.

## 9. Emergency agent changes and playbooks

In `emergency_agent.py`:

1. Add dispatch branches for the new kinds, following how `seismic_sensor` calls PLUM. Collect recent related events (the agent or orchestrator state may already keep a short window; reuse it, otherwise add a small in-memory buffer with a max age from config).
2. Convert each `HazardAssessment` into `EmergencyAgentResponse` with `ActionItem`s.
3. For `watch` and above, generate CAP-compatible alert actions exactly as the PLUM path does. Support the full lifecycle: `Alert`, `Update` (escalation), `Update` with all-clear, `Cancel`.
4. Preserve existing fallback behavior ("generic telemetry logging") for unknown kinds.

### Deterministic playbooks (`hazard_playbooks.py`)

Default response per hazard and level. Example shape (actual action types must match the existing `ActionItem` vocabulary):

| Hazard | Level | Actions |
|---|---|---|
| flood | watch | monitor, pre-position pumps, notify Mobility |
| flood | warning | public alert for flood-prone zones, close underpasses, notify Energy |
| flood | critical | evacuation advisory for affected zones (human approval required), dispatch rescue units |
| tsunami | warning | coastal public alert, clear beaches and ports, notify Mobility |
| tsunami | critical | evacuation of coastal zones (human approval required), activate shelters |
| storm | warning | public alert, secure construction sites, notify Energy |
| storm | critical | shelter-in-place alert, suspend transit, dispatch units |
| earthquake (catalog) | warning/critical | structural checks (same as PLUM path), notify Mobility and Energy |

Rule: any **evacuation** action, and any action at `critical`, sets a human-approval flag on the `ActionItem` / response (use whatever field exists, or add one).

### Optional LLM planner (off by default for hazards)

- Enabled only when `CITY_BRAIN_LLM` is not `off` **and** `GEMINI_API_KEY` is set (same switches as the router).
- Input: one `HazardAssessment` plus whatever city context the orchestrator state already holds. **Never** raw sensor streams. The LLM must not change `hazard` or `risk_level`.
- Output: a validated structured list of `ActionItem`s (pydantic) used to refine the playbook, for example by customizing public-alert wording for the affected area.
- On any failure (timeout, invalid output, no key) fall back silently to the deterministic playbook. The system must behave correctly with the LLM completely off.

## 10. External data feeds (optional adapters)

`src/modules/feeds/` adapters fetch external data and **emit `CityEvent`s**; engines themselves stay offline.

| Feed | Purpose | Emits |
|---|---|---|
| Open-Meteo (free, no key) | rainfall and wind forecast and current weather | `weather_forecast`, `weather_observation` |
| USGS | earthquake catalog | `seismic_catalog` |
| EMSC | earthquake catalog (strong for the Mediterranean) | `seismic_catalog` |
| Copernicus GloFAS | river discharge forecast | `hydro_observation` (stub for now) |
| National meteorological institute alerts | official warnings | `official_weather_alert` (stub, define the interface) |

- Polling interval, endpoints and area of interest come from config.
- Tag each event with its `catalog_source` or `source` so independent-source counting works.
- Use recorded sample responses as test fixtures; tests must not touch the network.

## 11. Mocks and tests

Follow the existing mock pattern: each mock file is self-executable and verifies end to end through `EmergencyAgent`, with a JSON twin.

**Mock scenarios to add**

- `mock_flood.py`: heavy forecast on saturated soil (critical), moderate rain (watch), dry conditions (low), missing optional inputs (confidence lowered), stale data ignored.
- `mock_catalog_quake.py`: M6.8 shallow nearby (critical with PLUM confirmation), M5.0 nearby (watch), M7 far away (below threshold).
- `mock_tsunami.py`: offshore M7.0 shallow without sea-level confirmation (warning), with confirmation (critical, human approval), onshore or deep quake (no tsunami).
- `mock_storm.py`: gust only (warning), radar rotation (critical), official alert (critical), calm (low).

**Pytest files** (the README notes none exist yet; add these as the first formal tests)

- engine unit tests: every level boundary, missing inputs, stale inputs, config overrides
- `critical` never produced with fewer than `min_sources_for_critical` independent sources
- agent tests: each new kind routes correctly, existing kinds unchanged, evacuation and `critical` set the human-approval flag
- orchestrator tests: new rule scores route to the Emergency axis, cascades notify the right axes, safety floor still holds
- regression: existing PLUM and crime mocks still pass

Tests must run offline, with the LLM disabled (`CITY_BRAIN_LLM=off`).

## 12. Configuration (`config/hazard_thresholds.yaml`)

One documented YAML with every `CONFIG` value from this file: default, unit and a comment each. Put this note at the top:

> Values are expert-estimated starting points and must be reviewed by a domain expert (civil protection, hydrologist, seismologist) before any real deployment.

Load it in `hazard_common.py` with an env var override for the path (for example `CITY_BRAIN_HAZARD_CONFIG`), and fall back to built-in defaults if the file is missing.

## 13. Acceptance criteria

1. Replaying the mock scenarios through the orchestrator produces the expected axis routing, `HazardAssessment` and `EmergencyAgentResponse` for flood, catalog earthquake, tsunami and storm.
2. Existing PLUM self-test, `mock_plum.py` and `mock_crime_hotspot.py` pass unchanged.
3. No score is labelled or described as a probability anywhere.
4. No threshold is hardcoded outside the config.
5. `critical` is never emitted with fewer than two independent sources.
6. Evacuation actions and all `critical` responses require human approval.
7. Everything works with the LLM fully off; the LLM planner failing never blocks a response.
8. Every decision appears in the existing `AuditLog`, including the drivers and evidence.
9. New pytest suite passes offline.

## 14. Future work (do not implement now)

- Once `AuditLog` has enough real events and outcomes, add a `record_outcome(event_id, label)` helper and train a gradient-boosting classifier (same approach as `hotspot_model.py`). Compare it against the rule-based index and keep the rules unless the model is clearly better.
- Synthetic flood scenarios from a hydrological model to supplement training data (validate against real events).
- STA/LTA waveform processing beyond PLUM, radar rotation detection, wildfire as an additional hazard.
- Fill in `INTERFACE.md` with the new event kinds and the `HazardAssessment` contract.

## 15. Suggested build order

1. Read the existing code (Section 2); summarize `CityEvent`, `ActionItem`, severity scale and the dispatch pattern in a short comment block.
2. `hazard_common.py` and `config/hazard_thresholds.yaml`
3. `flood_engine.py` + mock + tests
4. `catalog_quake_engine.py` + mock + tests
5. `tsunami_engine.py` + mock + tests
6. `storm_engine.py` + mock + tests
7. `hazard_playbooks.py` and `emergency_agent.py` dispatch branches + tests
8. `orchestrator.py`: `rule_emergency` and `rule_cascades` + tests
9. Feed adapters (`feeds/`) with recorded fixtures
10. Optional LLM planner behind the existing flags
11. Update `README.md` (new modules, event kinds, quick start) and `INTERFACE.md`
