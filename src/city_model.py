"""
City model for the City Brain dashboard.

Single source of truth for the simulated city the dashboard draws and the
scenario events it sends to the orchestrator:

  * 10x10 blocks (rows A-J, columns 1-10) with a zone or facility type
  * road graph: 11x11 intersections ("nodes"), node (i, j) is the corner
    north-west of block (row i, col j)
  * facilities: police, fire stations, hospitals, sanitation depot, substations...
  * underground layer: metro lines, road tunnel, sewer mains, power ducts
  * smart waste bins on block sides and garbage-truck collection routes
  * scenario builders that turn a dashboard click into a sensor event (CityEvent JSON)

Coordinates: block (r, c) centre is lat = BASE_LAT + (9 - r) * STEP, lon = BASE_LON + c * STEP.
Everything here is simulated until the Twin layer provides the real feeds.
"""
from __future__ import annotations

import datetime
import math
import random
import string
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
try:  # waveform samples from the team's PLUM mocks
    from mocks.mock_plum import get_mock_plum_event
except Exception:  # pragma: no cover
    get_mock_plum_event = None

ROWS = list(string.ascii_uppercase[:10])  # A..J
N = 10
BASE_LAT, BASE_LON, STEP = 36.8000, 10.1800, 0.001

# --------------------------------------------------------------------------- #
# Zoning (one letter per block, rows A..J, columns 1..10)
#   R residential  C commercial  I industrial  P park
#   H hospital     L police      F fire station
#   D sanitation depot   S substation   E school   K parking garage
# --------------------------------------------------------------------------- #
ZONING = [
    "RRRRRRRSII",  # A
    "RRPERFRRII",  # B
    "RLRRPRRHRR",  # C
    "RRRCCCCRRR",  # D
    "RRCCCCCKRP",  # E
    "RRCCCCCRFR",  # F
    "IIIRRERLRR",  # G
    "ISIRHRPRRR",  # H
    "IIFRRRRRRR",  # I
    "IIIDRRRRRR",  # J
]

TYPE_OF = {
    "R": "residential", "C": "commercial", "I": "industrial", "P": "park",
    "H": "hospital", "L": "police", "F": "fire_station", "D": "depot",
    "S": "substation", "E": "school", "K": "parking",
}

FACILITY_NAMES = {
    "A8": "North Substation (66/11 kV)",
    "H2": "South Substation (66/11 kV)",
    "C8": "Central General Hospital",
    "H5": "South Medical Center",
    "C2": "North Police Precinct",
    "G8": "South Police Precinct",
    "B6": "Fire Station 1 (North)",
    "F9": "Fire Station 2 (East)",
    "I3": "Fire Station 3 (Industrial / Hazmat)",
    "J4": "Municipal Sanitation Depot",
    "B4": "North Secondary School",
    "G6": "South Primary School",
    "E8": "Central Parking Garage",
    "B3": "North Gardens",
    "C5": "Central Park",
    "E10": "East Park",
    "H7": "South Park",
}

SHORT_NAMES = {
    "A8": "N. Substation", "H2": "S. Substation", "C8": "Central Hospital", "H5": "South Medical",
    "C2": "North Police", "G8": "South Police", "B6": "Fire St. 1", "F9": "Fire St. 2", "I3": "Fire St. 3",
    "J4": "Waste Depot", "B4": "N. School", "G6": "S. School", "E8": "Parking",
}

# Named roads (node index of the road line)
ROAD_NAMES = {
    "row": {3: "North Road", 5: "Central Avenue", 7: "South Road"},
    "col": {2: "West Street", 5: "Market Street", 8: "East Street"},
}

# District labels: (title, top-left block, bottom-right block)
DISTRICTS = [
    ("COMMERCIAL CORE", "D3", "F7"), ("INDUSTRIAL WEST", "G1", "J3"), ("INDUSTRIAL NORTH-EAST", "A9", "B10"),
    ("RESIDENTIAL NORTH", "A1", "C7"), ("RESIDENTIAL SOUTH", "G4", "J10"),
]

FACILITY_ASSETS = {
    "hospital": ["emergency_department", "ambulance_bay", "backup_generator", "helipad"],
    "police": ["patrol_cars", "acoustic_gunshot_array", "cctv_hub"],
    "fire_station": ["fire_engines", "rescue_unit", "hydrant_network"],
    "depot": ["garbage_trucks", "fleet_telematics"],
    "substation": ["transformers", "scada_meters", "feeder_breakers"],
    "school": ["school_zone_signals", "air_quality_station"],
    "parking": ["entry_exit_sensors", "vms_sign"],
    "industrial": ["air_quality_station", "gas_sensors"],
    "commercial": ["cctv", "street_lights"],
    "residential": ["street_lights"],
    "park": ["pedestrian_lights", "air_quality_station"],
}


def block_id(r: int, c: int) -> str:
    return f"{ROWS[r]}{c + 1}"


def parse_block(bid: str) -> tuple[int, int] | None:
    if not bid or bid[0] not in ROWS:
        return None
    try:
        c = int(bid[1:]) - 1
    except ValueError:
        return None
    return (ROWS.index(bid[0]), c) if 0 <= c < N else None


def block_latlon(r: int, c: int) -> tuple[float, float]:
    return round(BASE_LAT + (9 - r) * STEP, 4), round(BASE_LON + c * STEP, 4)


def node_latlon(i: float, j: float) -> tuple[float, float]:
    # node (i, j) is the north-west corner of block (i, j): half a step off the block centre
    return round(BASE_LAT + (9 - i + 0.5) * STEP, 5), round(BASE_LON + (j - 0.5) * STEP, 5)


def _build_blocks() -> list[dict]:
    blocks = []
    for r in range(N):
        for c in range(N):
            code = ZONING[r][c]
            btype = TYPE_OF[code]
            bid = block_id(r, c)
            lat, lon = block_latlon(r, c)
            name = FACILITY_NAMES.get(bid) or f"{btype.capitalize()} block {bid}"
            blocks.append({
                "id": bid, "row": r, "col": c, "type": btype, "name": name,
                "assets": FACILITY_ASSETS.get(btype, []), "lat": lat, "lon": lon, "short": SHORT_NAMES.get(bid),
            })
    return blocks


BLOCKS = _build_blocks()
BLOCKS_BY_ID = {b["id"]: b for b in BLOCKS}

# --------------------------------------------------------------------------- #
# Underground layer (polylines in node coordinates [i, j], fractional allowed)
# --------------------------------------------------------------------------- #
TUNNELS = [
    {
        "id": "M1", "kind": "metro", "name": "Metro Line 1 (East-West)", "depth_m": -14,
        "path": [[4, 0], [4, 10]],
        "stations": [{"at": [4, 1], "name": "West Gate"}, {"at": [4, 3], "name": "Medina"},
                     {"at": [4, 6], "name": "Central (L1/L2)"}, {"at": [4, 9], "name": "East Market"}],
    },
    {
        "id": "M2", "kind": "metro", "name": "Metro Line 2 (North-South)", "depth_m": -18,
        "path": [[0, 6], [4, 6], [6, 6], [7.5, 5], [10, 5]],
        "stations": [{"at": [1, 6], "name": "North Hills"}, {"at": [4, 6], "name": "Central (L1/L2)"},
                     {"at": [7.5, 5], "name": "Hospital South"}, {"at": [9.5, 5], "name": "Depot Road"}],
    },
    {
        "id": "T1", "kind": "road", "name": "Boulevard Road Tunnel", "depth_m": -8,
        "path": [[3, 2], [3.5, 3.2], [6.5, 6.8], [7, 8]],
        "portals": [[3, 2], [7, 8]],
        "flood_sensor": {"id": "HYD-T1-LOWPOINT", "at": [5, 5], "critical_cm": 30},
    },
    {"id": "SW1", "kind": "sewer", "name": "West Sewer Main", "depth_m": -5, "path": [[0, 2], [10, 2]]},
    {"id": "SW2", "kind": "sewer", "name": "Central Sewer Main", "depth_m": -5, "path": [[0, 5], [10, 5]]},
    {"id": "SW3", "kind": "sewer", "name": "East Sewer Main", "depth_m": -5, "path": [[0, 8], [10, 8]]},
    {"id": "SW4", "kind": "sewer", "name": "North Interceptor", "depth_m": -6, "path": [[3, 0], [3, 10]]},
    {"id": "SW5", "kind": "sewer", "name": "South Interceptor", "depth_m": -6, "path": [[7, 0], [7, 10], ]},
    {"id": "SW6", "kind": "sewer", "name": "Outfall to Treatment Plant", "depth_m": -7, "path": [[7, 0], [10, 0]]},
    {"id": "PD-N1", "kind": "power", "name": "Feeder N1 (North Substation)", "substation": "A8",
     "depth_m": -1.5, "path": [[1, 7.5], [1, 3], [2, 3]]},
    {"id": "PD-N2", "kind": "power", "name": "Feeder N2 (North Substation)", "substation": "A8",
     "depth_m": -1.5, "path": [[1, 7.5], [1, 8], [6, 8], [6, 9]]},
    {"id": "PD-S1", "kind": "power", "name": "Feeder S1 (South Substation)", "substation": "H2",
     "depth_m": -1.5, "path": [[7.5, 1.5], [6, 1.5], [6, 4], [3, 4]]},
    {"id": "PD-S2", "kind": "power", "name": "Feeder S2 (South Substation)", "substation": "H2",
     "depth_m": -1.5, "path": [[7.5, 1.5], [8, 1.5], [8, 6], [5, 6]]},
    {"id": "PD-S3", "kind": "power", "name": "Feeder S3 (South Substation)", "substation": "H2",
     "depth_m": -1.5, "path": [[7.5, 1.5], [9, 1.5], [9, 9]]},
]
TUNNELS_BY_ID = {t["id"]: t for t in TUNNELS}

# --------------------------------------------------------------------------- #
# Smart waste bins (placed on a side of a block) and truck routes
# --------------------------------------------------------------------------- #
def _build_bins() -> list[dict]:
    rng = random.Random(14)
    bins = []
    for b in BLOCKS:
        if b["type"] not in {"residential", "commercial", "school", "park", "parking"}:
            continue
        if rng.random() < 0.55:
            side = rng.choice("nsew")
            bins.append({
                "id": f"BIN-{b['id']}", "block": b["id"], "side": side,
                "fill_pct": round(rng.uniform(8, 82), 1), "capacity_l": 1100,
            })
    return bins


BINS = _build_bins()
BINS_BY_ID = {b["id"]: b for b in BINS}

DEPOT_NODE = [9, 3]  # north-west corner of block J4

# Fixed collection rounds (waypoints, expanded to Manhattan paths by the client).
# This is the "before" picture: fixed schedules, no fill-level awareness.
TRUCKS = [
    {"id": "GT-01", "name": "Garbage truck 01 (North round)",
     "route": [[9, 3], [5, 3], [5, 0], [0, 0], [0, 10], [2, 10], [2, 0], [3, 0], [3, 10], [5, 10], [5, 3], [9, 3]]},
    {"id": "GT-02", "name": "Garbage truck 02 (Centre round)",
     "route": [[9, 3], [6, 3], [6, 10], [4, 10], [4, 2], [5, 2], [5, 9], [6, 9], [6, 3], [9, 3]]},
    {"id": "GT-03", "name": "Garbage truck 03 (South round)",
     "route": [[9, 3], [9, 10], [7, 10], [7, 3], [8, 3], [8, 10], [10, 10], [10, 3], [9, 3]]},
]


# --------------------------------------------------------------------------- #
# Traffic signals: 3 neighbouring junctions on the avenue between rows E and F
# (node row 5, columns 3-5). Fixed-time plan with an eastbound green wave
# (100 m between junctions at ~36 km/h -> 10 s offset). The Mobility axis
# (sumo-rl adaptive control + emergency priority) will replace this plan.
# --------------------------------------------------------------------------- #
SIGNAL_PLAN = {"cycle_s": 50, "ns_green_s": 21, "ew_green_s": 21, "yellow_s": 3, "all_red_s": 1}


def _corner_blocks(i: int, j: int) -> list[str]:
    return [block_id(r, c) for r in (i - 1, i) for c in (j - 1, j) if 0 <= r < N and 0 <= c < N]


SIGNALS = [
    {"id": f"SIG-0{k + 1}", "node": [5, 3 + k], "offset_s": 10 * k, "control": "fixed_time",
     "plan": SIGNAL_PLAN, "junction_of": _corner_blocks(5, 3 + k)}
    for k in range(3)
]


# --------------------------------------------------------------------------- #
# Seismic station network (accelerometers) used by the full PLUM engine
# --------------------------------------------------------------------------- #
SEISMIC_STATIONS = []
for _k, _bid in enumerate(["B2", "B6", "B9", "E2", "E9", "H3", "H7", "J9"]):
    _r, _c = (ROWS.index(_bid[0]), int(_bid[1:]) - 1)
    _lat, _lon = block_latlon(_r, _c)
    SEISMIC_STATIONS.append({"sensor_id": f"SEIS-{_k + 1:02d}", "block": _bid, "lat": _lat, "lon": _lon})


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = (lat2 - lat1) * 111.19
    dlon = (lon2 - lon1) * 111.19 * math.cos(math.radians(lat1))
    return math.hypot(dlat, dlon)


def synth_station_network(epi_lat: float, epi_lon: float, magnitude: float = 6.0, depth_km: float = 8.0,
                          fs: int = 40, duration_s: int = 24, spike_station: str | None = None,
                          seed: int = 0) -> dict:
    """Synthetic 3-axis accelerograms for every station (same model as plum_engine._demo).

    Earthquake: S-wave arrives at 2 s + hypocentral distance / 3 km/s, amplitude decays with distance
    and scales with magnitude. spike_station: no earthquake, only an isolated 1-s spike on that
    station (e.g. a truck hitting the sensor housing) -> PLUM should suppress it.
    """
    import numpy as np  # numpy is already a dependency of plum_engine

    rng = np.random.default_rng(seed)
    t = np.arange(fs * duration_s) / fs
    f0 = 3.0
    stations = []
    for st in SEISMIC_STATIONS:
        if spike_station is None:
            rh = math.hypot(_km(epi_lat, epi_lon, st["lat"], st["lon"]), depth_km)
            t_arr = 2.0 + rh / 3.0
            amp = 3.0 * (5.0 / (5.0 + rh)) ** 1.5 * 10 ** (0.5 * (magnitude - 6.0))   # PGA, m/s^2
            env = np.clip((t - t_arr) / 1.0, 0, 1) * np.exp(-np.clip(t - t_arr, 0, None) / 6.0)
            v = amp / (2 * math.pi * f0) * env * np.sin(2 * math.pi * f0 * (t - t_arr))
            a = np.gradient(v, 1 / fs)
        else:
            a = np.zeros_like(t)
            if st["sensor_id"] == spike_station:
                i0 = int(3 * fs)
                a[i0:i0 + fs] = 3.0 * np.sin(2 * math.pi * 5 * t[:fs])
        noise = lambda: 1e-4 * rng.standard_normal(len(t))
        scale = (0.5, 0.8, 0.6) if spike_station is None else (1.0, 0.1, 0.1)
        stations.append({
            "sensor_id": st["sensor_id"], "lat": st["lat"], "lon": st["lon"],
            "accel": {ax: [round(float(x), 5) for x in (a * k + noise())]
                      for ax, k in zip(("vertical_z", "east_west_e", "north_south_n"), scale)},
        })
    return {"samplingRateHz": fs, "units": "m/s^2", "durationS": duration_s, "stations": stations}


def city_payload() -> dict:
    return {
        "grid": {"rows": ROWS, "cols": N, "base_lat": BASE_LAT, "base_lon": BASE_LON, "step_deg": STEP},
        "blocks": BLOCKS, "tunnels": TUNNELS, "bins": BINS, "trucks": TRUCKS, "depot_node": DEPOT_NODE,
        "signals": SIGNALS, "seismic_stations": SEISMIC_STATIONS,
        "road_names": ROAD_NAMES, "districts": DISTRICTS,
        "simulated": True,
    }


# --------------------------------------------------------------------------- #
# Scenario builders: dashboard click -> sensor event for the orchestrator
# --------------------------------------------------------------------------- #
SCENARIOS = {
    "accident": {"title": "Road accident", "icon": "🚗", "default_block": "D5",
                 "axes": ["emergency", "mobility"]},
    "fire": {"title": "Building fire", "icon": "🔥", "default_block": "F4",
             "axes": ["emergency", "mobility", "environment"]},
    "load_spike": {"title": "Spoofed load spike", "icon": "⚡", "default_block": "H2",
                   "axes": ["energy"]},
    "garbage": {"title": "Garbage collection", "icon": "🗑️", "default_block": None,
                "axes": ["environment"]},
    "tunnel_flood": {"title": "Tunnel flood", "icon": "🌊", "default_block": "E5",
                     "axes": ["emergency", "mobility"]},
    "earthquake": {"title": "Earthquake (PLUM)", "icon": "🌐", "default_block": "E6",
                   "axes": ["emergency"]},
    "seismic_spike": {"title": "Seismic false trigger", "icon": "📈", "default_block": "H7",
                      "axes": ["emergency"]},
    "gunshot": {"title": "Gunshot", "icon": "🎯", "default_block": "D7",
                "axes": ["emergency"]},
}

BLOCK_M = 100.0  # one block is about 100 m


def _dist_to_type_m(r: int, c: int, btype: str) -> float:
    """Street (Manhattan) distance in metres from block (r, c) to the nearest block of a type."""
    d = min(abs(b["row"] - r) + abs(b["col"] - c) for b in BLOCKS if b["type"] == btype)
    return d * BLOCK_M


def _historical_crime_count(bid: str) -> int:
    """Simulated incident history per block (stand-in until real records come from the Twin layer)."""
    base = {"commercial": 18, "industrial": 14, "residential": 9, "park": 12}.get(BLOCKS_BY_ID[bid]["type"], 6)
    return base + random.Random(bid).randint(0, 16)


_WAVEFORM_FALLBACK = {
    "samplingRateHz": 100, "units": "m/s^2", "clippingDetected": False, "noiseFloorExceeded": False,
    "components": {"vertical_z": [0.05, 0.45, 1.25, 2.10, 1.50], "east_west_e": [0.08, -0.35, 0.95, 1.80, 0.85],
                   "north_south_n": [-0.03, 0.25, 0.85, 1.95, 0.75]},
}


def _nearest(cands: list[str], r: int, c: int) -> str:
    return min(cands, key=lambda bid: abs(parse_block(bid)[0] - r) + abs(parse_block(bid)[1] - c))


def build_scenario(scenario: str, block: str | None = None, bin_id: str | None = None,
                   magnitude: float | None = None) -> tuple[dict, dict]:
    """Returns (event, target). `target` tells the dashboard where to draw the incident."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}'")
    spec = SCENARIOS[scenario]
    bid = block if block in BLOCKS_BY_ID else spec["default_block"]

    if scenario == "accident":
        r, c = parse_block(bid)
        edge = [[r, c], [r, c + 1]]  # road on the north side of the block
        lat, lon = node_latlon(r, c + 0.5)
        event = {
            "source": "traffic_cctv_edge", "kind": "accident", "lat": lat, "lon": lon,
            "data": {"block_id": bid, "road_segment": f"ROAD-{bid}-N", "severity": 4,
                     "vehicles_involved": 3, "injuries_reported": 2, "lanes_blocked": 2,
                     "confidence": 0.91},
        }
        return event, {"type": "accident", "block": bid, "edge": edge}

    if scenario == "fire":
        if BLOCKS_BY_ID[bid]["type"] in {"park"}:
            bid = spec["default_block"]
        b = BLOCKS_BY_ID[bid]
        event = {
            "source": "cv_camera_edge", "kind": "vision_fire_smoke", "lat": b["lat"], "lon": b["lon"],
            "data": {"block_id": bid, "camera_id": f"CAM-{bid}-02", "fire_detected": True,
                     "smoke_detected": True, "confidence": 0.94, "flame_area_pct": 14.2},
        }
        return event, {"type": "fire", "block": bid}

    if scenario == "load_spike":
        r, c = parse_block(bid)
        sub = bid if BLOCKS_BY_ID[bid]["type"] == "substation" else _nearest(["A8", "H2"], r, c)
        feeder = "PD-S2" if sub == "H2" else "PD-N2"
        b = BLOCKS_BY_ID[sub]
        event = {
            "source": "scada_feeder_meter", "kind": "grid_load", "lat": b["lat"], "lon": b["lon"],
            "data": {"block_id": sub, "substation_id": sub, "feeder_id": feeder,
                     "load_pct": 148.0, "previous_load_pct": 42.0, "rate_of_change_pct_per_s": 106.0,
                     "neighbour_feeders_load_pct": [41.0, 44.0], "current_a_measured": 212.0,
                     "telemetry_signature_valid": False},
        }
        return event, {"type": "load_spike", "block": sub, "feeder": feeder}

    if scenario == "garbage":
        cand = None
        if bin_id in BINS_BY_ID:
            cand = BINS_BY_ID[bin_id]
        elif block:
            in_block = [x for x in BINS if x["block"] == block]
            cand = in_block[0] if in_block else None
        cand = cand or max(BINS, key=lambda x: x["fill_pct"])
        b = BLOCKS_BY_ID[cand["block"]]
        event = {
            "source": "smart_bin_iot", "kind": "smart_waste_bin", "lat": b["lat"], "lon": b["lon"],
            "data": {"block_id": b["id"], "bin_id": cand["id"], "fill_pct": 96.0,
                     "odor_voc_ppm": 45.0, "capacity_l": cand["capacity_l"]},
        }
        return event, {"type": "garbage", "block": b["id"], "bin_id": cand["id"]}

    if scenario == "earthquake":
        b = BLOCKS_BY_ID[bid]
        mag = float(magnitude or 6.0)
        event = {
            "source": "seismic_network", "kind": "seismic_sensor", "lat": b["lat"], "lon": b["lon"],
            "data": {"block_id": bid, "magnitude_sim": mag, "depth_km_sim": 8.0,
                     "stationNetwork": synth_station_network(b["lat"], b["lon"], magnitude=mag)},
        }
        return event, {"type": "earthquake", "block": bid}

    if scenario == "seismic_spike":
        r, c = parse_block(bid)
        st = min(SEISMIC_STATIONS, key=lambda x: abs(parse_block(x["block"])[0] - r) + abs(parse_block(x["block"])[1] - c))
        b = BLOCKS_BY_ID[st["block"]]
        event = {
            "source": "seismic_network", "kind": "seismic_sensor", "lat": b["lat"], "lon": b["lon"],
            "data": {"block_id": st["block"],
                     "stationNetwork": synth_station_network(b["lat"], b["lon"], spike_station=st["sensor_id"])},
        }
        return event, {"type": "seismic_spike", "block": st["block"], "station": st["sensor_id"]}

    if scenario == "gunshot":
        r, c = parse_block(bid)
        b = BLOCKS_BY_ID[bid]
        now = datetime.datetime.now()
        event = {
            "source": "acoustic_array", "kind": "acoustic_gunshot_triangulation", "lat": b["lat"], "lon": b["lon"],
            "data": {
                "block_id": bid, "decibelLevel": 118.2, "severity": 5,
                "associatedContext": {
                    "dist_to_police_station_m": _dist_to_type_m(r, c, "police"),
                    "dist_to_hospital_m": _dist_to_type_m(r, c, "hospital"),
                    "historical_crime_count": _historical_crime_count(bid),
                    "is_weekend": now.weekday() >= 5,
                    "is_night": now.hour >= 20 or now.hour < 6,
                    "street_lighting_status": "ok",
                },
            },
        }
        return event, {"type": "gunshot", "block": bid}

    # tunnel_flood
    t = TUNNELS_BY_ID["T1"]
    s = t["flood_sensor"]
    lat, lon = node_latlon(*s["at"])
    event = {
        "source": "hydrology_sensor", "kind": "flood_sensor", "lat": lat, "lon": lon,
        "data": {"block_id": "E5", "sensor_id": s["id"], "tunnel_id": "T1", "water_level_cm": 46.0,
                 "critical_threshold_cm": float(s["critical_cm"]), "submerged_lane_count": 2},
    }
    return event, {"type": "tunnel_flood", "block": "E5", "tunnel": "T1", "at": s["at"]}
