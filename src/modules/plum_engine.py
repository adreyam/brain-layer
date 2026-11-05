"""
PLUM (Propagation of Local Undamped Motion) event-detection engine.

Deterministic re-implementation of the detection logic described in
Kilb et al. (2021), JGR Solid Earth, 126, e2020JB021053 (Instance #5):

  1. Per station, per 1-second block, compute peak 3-axis acceleration and
     velocity and convert them to Modified Mercalli Intensity (Worden et al., 2012).
  2. Co-located velocity + acceleration sensors are cross-checked
     (ratio R = I_A / I_V must be in 0.9-1.1, else one sensor is rejected).
  3. Two-station trigger logic: a station with MMI >= primary (4.0) is PENDING;
     it becomes an event DETECTION only if one of its six nearest neighbours
     reaches MMI >= secondary (3.0) within 15 s. Otherwise the trigger is suppressed.
  4. An active event ends when all its stations stay below MMI 2.0 for 10 s.

Determinism: no randomness, no wall-clock time, and all tie-breaks are by
(distance, station id) or (MMI, station id).

NOTE: PLUM detects shaking; it does not estimate magnitude or epicentre.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

try:  # optional: without scipy the band-pass step is skipped (demean only)
    from scipy.signal import butter, sosfilt
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

AXES = ("vertical_z", "east_west_e", "north_south_n")


# ------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class PlumConfig:
    primary_mmi: float = 4.0            # station #1 threshold (Instance #5)
    secondary_mmi: float = 3.0          # station #2 threshold (Instance #5)
    confirm_window_s: int = 15          # neighbour must confirm within this time
    n_neighbors: int = 6                # neighbours examined per trigger
    # 15 s x ~3 km/s S-wave = 45 km. The paper enforces this only through the
    # timeout; capping distance explicitly stops far-away unrelated stations
    # from "confirming" a spike by coincidence.
    max_neighbor_km: Optional[float] = 45.0
    end_mmi: float = 2.0                # event over when all members < this ...
    end_hold_s: int = 10                # ... for this many seconds
    ratio_lo: float = 0.9               # co-located sensor QC band
    ratio_hi: float = 1.1
    vel_trust_mmi: float = 3.0          # QC fail: I_V < 3 -> trust velocity, else accel
    bandpass_hz: Tuple[float, float] = (0.5, 25.0)


@dataclass(frozen=True)
class Station:
    sensor_id: str
    lat: float
    lon: float


@dataclass
class StationData:
    """Continuous 3-axis records for one station. All stations must share the same
    start time (sample 0 = second 0). Accelerometer in m/s^2, velocity in m/s."""
    station: Station
    sample_rate_hz: float
    accel: Optional[Dict[str, Sequence[float]]] = None
    velocity: Optional[Dict[str, Sequence[float]]] = None


@dataclass
class Detection:
    detection_time_s: float        # end of the 1-s packet in which confirmation occurred
    trigger_station: str
    trigger_second: int
    trigger_mmi: float
    confirming_station: str
    confirming_mmi: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class Suppressed:
    station: str
    trigger_second: int
    trigger_mmi: float
    reason: str


@dataclass
class PlumResult:
    detections: List[Detection]
    suppressed: List[Suppressed]


# ------------------------------------------------------------- intensity conversion
def pga_to_mmi(pga_cm_s2: np.ndarray) -> np.ndarray:
    """Worden et al. (2012) bilinear PGA -> MMI (PGA in cm/s^2)."""
    lg = np.log10(np.maximum(pga_cm_s2, 1e-6))
    return np.where(lg <= 1.57, 1.78 + 1.55 * lg, -1.60 + 3.70 * lg)


def pgv_to_mmi(pgv_cm_s: np.ndarray) -> np.ndarray:
    """Worden et al. (2012) bilinear PGV -> MMI (PGV in cm/s)."""
    lg = np.log10(np.maximum(pgv_cm_s, 1e-6))
    return np.where(lg <= 0.53, 3.78 + 1.47 * lg, 2.89 + 3.16 * lg)


def mmi_from_pga_pgv(pga_cm_s2: np.ndarray, pgv_cm_s: np.ndarray) -> np.ndarray:
    """PGA-based MMI at low intensity, PGV-based at high intensity, linear blend
    between MMI 5 and 7 (ShakeMap-style). ASSUMPTION: the paper cites Worden's
    equations without spelling out the combination; verify against the source."""
    m_a, m_v = pga_to_mmi(pga_cm_s2), pgv_to_mmi(pgv_cm_s)
    w = np.clip((m_a - 5.0) / 2.0, 0.0, 1.0)
    return np.clip((1.0 - w) * m_a + w * m_v, 1.0, 10.0)


# ------------------------------------------------------------------ signal handling
def _prep(x: Sequence[float], fs: float, cfg: PlumConfig) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if not _HAVE_SCIPY:
        return x
    nyq = fs / 2.0
    lo, hi = cfg.bandpass_hz[0], min(cfg.bandpass_hz[1], 0.95 * nyq)
    if lo >= hi:
        return x
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfilt(sos, x)  # causal, so it behaves the same in real time


def _integrate(x: np.ndarray, fs: float, cfg: PlumConfig) -> np.ndarray:
    y = np.concatenate([[0.0], np.cumsum((x[1:] + x[:-1]) / 2.0)]) / fs
    return _prep(y, fs, cfg)  # re-filter to suppress integration drift


def _differentiate(x: np.ndarray, fs: float) -> np.ndarray:
    return np.gradient(x, 1.0 / fs)


def _peak_per_second(axes: List[np.ndarray], fs: float) -> np.ndarray:
    """max over each 1-s block of sqrt(Z^2 + E^2 + N^2)  (Eqs. 1a/1b)."""
    spp = int(round(fs))
    n_sec = min(len(a) for a in axes) // spp
    mag = np.sqrt(sum(a[: n_sec * spp] ** 2 for a in axes))
    return mag.reshape(n_sec, spp).max(axis=1)


def _read_axes(d: Dict[str, Sequence[float]], fs: float, cfg: PlumConfig) -> List[np.ndarray]:
    missing = [k for k in AXES if k not in d]
    if missing:
        raise ValueError(f"missing component(s): {missing}")
    arrs = [_prep(d[k], fs, cfg) for k in AXES]
    if len({len(a) for a in arrs}) != 1:
        raise ValueError("components must have equal length")
    return arrs


def station_intensity(sd: StationData, cfg: PlumConfig = PlumConfig()) -> np.ndarray:
    """1-sample-per-second MMI series for one station, with co-located sensor QC."""
    fs = sd.sample_rate_hz
    if sd.accel is None and sd.velocity is None:
        raise ValueError(f"{sd.station.sensor_id}: no data")

    i_a = i_v = None
    if sd.accel is not None:
        a = _read_axes(sd.accel, fs, cfg)
        v_from_a = [_integrate(x, fs, cfg) for x in a]
        i_a = mmi_from_pga_pgv(100.0 * _peak_per_second(a, fs), 100.0 * _peak_per_second(v_from_a, fs))
    if sd.velocity is not None:
        v = _read_axes(sd.velocity, fs, cfg)
        a_from_v = [_differentiate(x, fs) for x in v]
        i_v = mmi_from_pga_pgv(100.0 * _peak_per_second(a_from_v, fs), 100.0 * _peak_per_second(v, fs))

    if i_a is None:
        return i_v
    if i_v is None:
        return i_a

    n = min(len(i_a), len(i_v))
    i_a, i_v = i_a[:n], i_v[:n]
    ratio = i_a / i_v
    ok = (ratio >= cfg.ratio_lo) & (ratio <= cfg.ratio_hi)
    fallback = np.where(i_v < cfg.vel_trust_mmi, i_v, i_a)  # velocity may clip at high shaking
    return np.where(ok, 0.5 * (i_a + i_v), fallback)


# ---------------------------------------------------------------------- geometry
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_neighbors(stations: Sequence[Station], cfg: PlumConfig) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for s in stations:
        cands = sorted(
            (haversine_km(s.lat, s.lon, o.lat, o.lon), o.sensor_id)
            for o in stations if o.sensor_id != s.sensor_id
        )[: cfg.n_neighbors]
        if cfg.max_neighbor_km is not None:
            cands = [c for c in cands if c[0] <= cfg.max_neighbor_km]
        out[s.sensor_id] = [sid for _, sid in cands]
    return out


# ------------------------------------------------------------------ event detection
@dataclass
class _Event:
    members: set
    quiet: int = 0


def detect(mmi: Dict[str, np.ndarray], neighbors: Dict[str, List[str]],
           cfg: PlumConfig = PlumConfig()) -> PlumResult:
    """Run the two-station PLUM state machine over aligned per-second MMI series."""
    ids = sorted(mmi)
    n_sec = min(len(mmi[i]) for i in ids)
    nbr_sets = {k: set(v) for k, v in neighbors.items()}
    pending: Dict[str, int] = {}
    active: List[_Event] = []
    detections: List[Detection] = []
    suppressed: List[Suppressed] = []

    def val(sid: str, t: int) -> float:
        return float(mmi[sid][t])  # NaN compares False everywhere below

    for t in range(n_sec):
        # 1) active events: absorb adjoining shaking stations, then test for the end
        for ev in active:
            for sid in ids:
                if sid not in ev.members and val(sid, t) >= cfg.secondary_mmi and ev.members & nbr_sets[sid]:
                    ev.members.add(sid)
            ev.quiet = ev.quiet + 1 if all(not (val(m, t) >= cfg.end_mmi) for m in ev.members) else 0
        active = [ev for ev in active if ev.quiet < cfg.end_hold_s]
        in_event = set().union(*(ev.members for ev in active)) if active else set()

        # 2) new primary triggers
        for sid in ids:
            if val(sid, t) >= cfg.primary_mmi and sid not in in_event and sid not in pending:
                pending[sid] = t

        # 3) confirm or expire pending triggers (sorted -> deterministic)
        for sid in sorted(pending):
            if sid not in pending:
                continue  # absorbed by an event created earlier in this loop
            t0 = pending[sid]
            if t - t0 > cfg.confirm_window_s:
                suppressed.append(Suppressed(sid, t0, val(sid, t0),
                                             f"no neighbour reached MMI {cfg.secondary_mmi} within {cfg.confirm_window_s} s"))
                del pending[sid]
                continue
            confirmers = sorted(((val(n, t), n) for n in neighbors[sid] if val(n, t) >= cfg.secondary_mmi),
                                key=lambda c: (-c[0], c[1]))
            if not confirmers:
                continue
            c_mmi, c_id = confirmers[0]
            ev = _Event(members={sid, c_id})
            for p in sorted(pending):  # other pending stations adjoining this event join it
                if p in ev.members or ev.members & nbr_sets[p]:
                    ev.members.add(p)
            for p in ev.members:
                pending.pop(p, None)
            active.append(ev)
            detections.append(Detection(
                detection_time_s=t + 1.0, trigger_station=sid, trigger_second=t0,
                trigger_mmi=round(val(sid, t0), 2), confirming_station=c_id, confirming_mmi=round(c_mmi, 2),
                evidence=[
                    f"{sid} reached MMI {val(sid, t0):.2f} (>= {cfg.primary_mmi}) at t={t0} s",
                    f"neighbour {c_id} reached MMI {c_mmi:.2f} (>= {cfg.secondary_mmi}) at t={t} s",
                    "PLUM two-station confirmation satisfied",
                ]))

    for sid, t0 in sorted(pending.items()):  # record still-open triggers at end of data
        suppressed.append(Suppressed(sid, t0, val(sid, t0), "data ended before confirmation"))
    return PlumResult(detections, suppressed)


class WaveformTelemetry(BaseModel):
    samplingRateHz: int = 100
    components: Dict[str, List[float]]  # vertical_z, east_west_e, north_south_n
    units: str = "m/s^2"
    clippingDetected: bool = False
    noiseFloorExceeded: bool = False


class SensorReading(BaseModel):
    sensor_id: str
    timestamp: float = Field(default_factory=lambda: 0.0)
    data: Dict[str, Any] = Field(default_factory=dict)


class PLUMEngine:
    @staticmethod
    def run(data: Sequence[StationData], cfg: PlumConfig = PlumConfig()) -> PlumResult:
        stations = [d.station for d in data]
        if len({s.sensor_id for s in stations}) != len(stations):
            raise ValueError("duplicate sensor_id")
        mmi = {d.station.sensor_id: station_intensity(d, cfg) for d in data}
        return detect(mmi, build_neighbors(stations, cfg), cfg)

    @classmethod
    def calculate_pga(cls, components: Dict[str, Sequence[float]]) -> float:
        z = components.get("vertical_z", [])
        e = components.get("east_west_e", [])
        n = components.get("north_south_n", [])
        min_len = min(len(z), len(e), len(n))
        if min_len == 0:
            return 0.0
        max_vector = 0.0
        for i in range(min_len):
            mag = math.sqrt(z[i]**2 + e[i]**2 + n[i]**2)
            if mag > max_vector:
                max_vector = mag
        return max_vector

    @classmethod
    def pga_to_mmi_scalar(cls, pga_m_s2: float) -> float:
        """Worden et al. (2012) bilinear conversion PGA (m/s^2) -> MMI."""
        if pga_m_s2 <= 0.0001:
            return 1.0
        pga_cm_s2 = pga_m_s2 * 100.0
        lg = math.log10(max(pga_cm_s2, 1e-6))
        if lg <= 1.57:
            mmi = 1.78 + 1.55 * lg
        else:
            mmi = -1.60 + 3.70 * lg
        return round(max(1.0, min(10.0, mmi)), 2)

    @classmethod
    def process_telemetry(cls, telemetry: WaveformTelemetry | dict) -> Tuple[float, float, bool, List[str]]:
        if isinstance(telemetry, dict):
            telemetry = WaveformTelemetry(**telemetry)
        evidence = []
        if telemetry.clippingDetected:
            evidence.append("Hardware Warning: Sensor clipping detected (signal maxed out).")
        if telemetry.noiseFloorExceeded:
            evidence.append("Hardware Warning: Sensor noise floor exceeded.")
        pga = cls.calculate_pga(telemetry.components)
        mmi = cls.pga_to_mmi_scalar(pga)
        evidence.append(f"PLUM Computed PGA: {pga:.4f} {telemetry.units} | Estimated MMI: {mmi}")
        alert_triggered = mmi >= 4.0
        if alert_triggered:
            evidence.append(f"PLUM Threshold Exceeded! MMI {mmi} >= 4.0 target alert baseline.")
        return pga, mmi, alert_triggered, evidence


# ------------------------------------------------------------------------ self-test
def _demo() -> None:
    fs, dur, f0 = 100.0, 60, 3.0
    t = np.arange(int(fs * dur)) / fs
    rng = np.random.default_rng(0)  # fixed seed -> reproducible test
    lat0, lon0 = 37.0, -122.0
    km_lat, km_lon = 111.19, 111.19 * math.cos(math.radians(lat0))

    def wave(r_km: float, onset_extra: float = 0.0, amp_scale: float = 1.0):
        rh = math.hypot(r_km, 8.0)
        t_arr = rh / 3.0 + onset_extra
        amp = 3.0 * (5.0 / (5.0 + rh)) ** 1.5 * amp_scale          # PGA, m/s^2
        env = np.clip((t - t_arr) / 1.0, 0, 1) * np.exp(-np.clip(t - t_arr, 0, None) / 6.0)
        v = amp / (2 * math.pi * f0) * env * np.sin(2 * math.pi * f0 * (t - t_arr))
        return v, np.gradient(v, 1 / fs)

    data: List[StationData] = []
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            x, y = ix * 20.0, iy * 20.0
            v, a = wave(math.hypot(x, y))
            noise = lambda: 1e-4 * rng.standard_normal(len(t))
            sid = f"G{ix + 2}{iy + 2}"
            comp = lambda s: {"vertical_z": s * 0.5 + noise(), "east_west_e": s * 0.8 + noise(), "north_south_n": s * 0.6 + noise()}
            if ix == 0 and iy == 0:  # co-located pair, velocity sensor clips
                clip = 0.02  # velocity sensor clips; its noise floor is lower than the accelerometer's
                vel = {k: np.clip(w * c + 1e-6 * rng.standard_normal(len(t)), -clip, clip)
                       for k, w, c in zip(AXES, (v, v, v), (0.5, 0.8, 0.6))}
                data.append(StationData(Station(sid, lat0, lon0), fs, accel=comp(a), velocity=vel))
            else:
                data.append(StationData(Station(sid, lat0 + y / km_lat, lon0 + x / km_lon), fs, accel=comp(a)))

    # isolated spike, long after the earthquake, neighbours quiet -> must be suppressed
    spike = np.zeros_like(t)
    spike[int(40 * fs): int(41 * fs)] = 3.0 * np.sin(2 * math.pi * 5 * t[: int(fs)])
    data.append(StationData(Station("SPK", lat0 + 60.0 / km_lat, lon0), fs,
                            accel={"vertical_z": spike, "east_west_e": spike * 0.1, "north_south_n": spike * 0.1}))

    res = PLUMEngine.run(data)
    print(f"detections: {len(res.detections)}")
    for d in res.detections:
        print(f"  t={d.detection_time_s:.0f}s  trigger={d.trigger_station} ({d.trigger_mmi})  "
              f"confirmed by {d.confirming_station} ({d.confirming_mmi})")
    print(f"suppressed triggers: {[(s.station, s.trigger_second) for s in res.suppressed]}")

    mmi_center = station_intensity(next(d for d in data if d.station.sensor_id == "G22"))
    print(f"G22 peak fused MMI: {mmi_center.max():.2f}")

    assert len(res.detections) == 1, "expected exactly one event"
    assert any(s.station == "SPK" for s in res.suppressed), "isolated spike must be suppressed"
    assert PLUMEngine.run(data).detections[0].evidence == res.detections[0].evidence, "must be deterministic"
    print("self-test passed")


if __name__ == "__main__":
    _demo()