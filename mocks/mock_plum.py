"""
Mock test data and execution scenarios for the PLUM (Propagation of Local Undamped Motion) Seismic Algorithm.

Includes:
- Single-sensor 3-axis strong motion waveform telemetry (compatible with EmergencyAgent & Orchestrator CityEvent)
- Critical seismic events (MMI >= 4.0 trigger)
- Minor ground motion events (MMI < 4.0 structural check)
- Sensor clipping / noise anomalies
- Multi-station continuous records for PLUMEngine.run state-machine validation
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure parent and src directory are in python path for standalone execution
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.plum_engine import PLUMEngine, PlumConfig, Station, StationData, WaveformTelemetry
from modules.emergency_agent import EmergencyAgent


# --------------------------------------------------------------------------- #
# Scenario 1: Critical Strong-Motion Earthquake (MMI >= 4.0 Alert Threshold)
# --------------------------------------------------------------------------- #
MOCK_PLUM_STRONG_MOTION_EVENT: Dict[str, Any] = {
    "id": "evt-plum-critical-001",
    "source": "twin_iot_sensor",
    "kind": "seismic_sensor",
    "lat": 36.8021,
    "lon": 10.1815,
    "data": {
        "rawWaveformTelemetry": {
            "samplingRateHz": 100,
            "components": {
                "vertical_z": [0.05, 0.45, 1.25, 2.10, 1.50],
                "east_west_e": [0.08, -0.35, 0.95, 1.80, 0.85],
                "north_south_n": [-0.03, 0.25, 0.85, 1.95, 0.75],
            },
            "units": "m/s^2",
            "clippingDetected": False,
            "noiseFloorExceeded": False,
        }
    },
}

# --------------------------------------------------------------------------- #
# Scenario 2: Minor Ground Tremor / Motion (MMI < 4.0, Structural Check Only)
# --------------------------------------------------------------------------- #
MOCK_PLUM_MINOR_MOTION_EVENT: Dict[str, Any] = {
    "id": "evt-plum-minor-002",
    "source": "twin_iot_sensor",
    "kind": "seismic_sensor",
    "lat": 36.8025,
    "lon": 10.1820,
    "data": {
        "rawWaveformTelemetry": {
            "samplingRateHz": 100,
            "components": {
                "vertical_z": [0.005, 0.012, 0.025, 0.018, 0.008],
                "east_west_e": [0.003, 0.015, 0.022, 0.014, 0.006],
                "north_south_n": [0.004, 0.010, 0.020, 0.015, 0.005],
            },
            "units": "m/s^2",
            "clippingDetected": False,
            "noiseFloorExceeded": False,
        }
    },
}

# --------------------------------------------------------------------------- #
# Scenario 3: Sensor Hardware Anomaly (Clipping & Noise Floor Exceeded)
# --------------------------------------------------------------------------- #
MOCK_PLUM_CLIPPING_ANOMALY_EVENT: Dict[str, Any] = {
    "id": "evt-plum-clip-003",
    "source": "twin_iot_sensor",
    "kind": "seismic_sensor",
    "lat": 36.8010,
    "lon": 10.1800,
    "data": {
        "rawWaveformTelemetry": {
            "samplingRateHz": 100,
            "components": {
                "vertical_z": [2.50, 2.50, 2.50, 2.50, 2.50],
                "east_west_e": [2.50, 2.50, 2.50, 2.50, 2.50],
                "north_south_n": [2.50, 2.50, 2.50, 2.50, 2.50],
            },
            "units": "m/s^2",
            "clippingDetected": True,
            "noiseFloorExceeded": True,
        }
    },
}


def get_mock_plum_event(scenario: str = "strong_motion") -> Dict[str, Any]:
    """Retrieve a mock PLUM event dictionary by scenario key."""
    scenarios = {
        "strong_motion": MOCK_PLUM_STRONG_MOTION_EVENT,
        "critical": MOCK_PLUM_STRONG_MOTION_EVENT,
        "minor": MOCK_PLUM_MINOR_MOTION_EVENT,
        "clipping": MOCK_PLUM_CLIPPING_ANOMALY_EVENT,
    }
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(scenarios.keys())}")
    return scenarios[scenario].copy()


def get_mock_station_network_data() -> List[StationData]:
    """
    Generates a reproducible 3-station network dataset:
    - Station 1 & Station 2 (close to each other) detect high shaking confirming an earthquake.
    - Station 3 (isolated spike) is suppressed by PLUM's 2-station rule.
    """
    import numpy as np

    fs = 50.0
    duration_s = 20
    n_samples = int(fs * duration_s)
    t = np.arange(n_samples) / fs

    # Seismic waveform pulse (t ~ 3s to 8s)
    wave = np.sin(2 * np.pi * 2.0 * t) * np.exp(-((t - 5.0) ** 2) / 2.0) * 1.5

    # Station 1 (Epicentral area)
    s1 = Station(sensor_id="STA-01", lat=36.800, lon=10.180)
    data1 = StationData(
        station=s1,
        sample_rate_hz=fs,
        accel={
            "vertical_z": wave,
            "east_west_e": wave * 0.8,
            "north_south_n": wave * 0.7,
        },
    )

    # Station 2 (Neighbour within 10 km, slight delay)
    s2 = Station(sensor_id="STA-02", lat=36.850, lon=10.190)
    wave_delayed = np.sin(2 * np.pi * 2.0 * (t - 1.5)) * np.exp(-((t - 6.5) ** 2) / 2.0) * 1.2
    data2 = StationData(
        station=s2,
        sample_rate_hz=fs,
        accel={
            "vertical_z": wave_delayed,
            "east_west_e": wave_delayed * 0.7,
            "north_south_n": wave_delayed * 0.6,
        },
    )

    # Station 3 (Far away, isolated noise spike at t=12s)
    s3 = Station(sensor_id="STA-03-ISOLATED", lat=37.500, lon=11.200)
    spike = np.zeros(n_samples)
    spike[int(12 * fs) : int(13 * fs)] = 2.0
    data3 = StationData(
        station=s3,
        sample_rate_hz=fs,
        accel={
            "vertical_z": spike,
            "east_west_e": spike * 0.1,
            "north_south_n": spike * 0.1,
        },
    )

    return [data1, data2, data3]


if __name__ == "__main__":
    print("=" * 60)
    print("PLUM MOCK DATA & ALGORITHM VERIFICATION")
    print("=" * 60)

    # 1. Direct Telemetry Test
    print("\n1. Testing direct PLUM strong-motion telemetry evaluation:")
    raw_telemetry = WaveformTelemetry(**MOCK_PLUM_STRONG_MOTION_EVENT["data"]["rawWaveformTelemetry"])
    pga, mmi, alert, evidence = PLUMEngine.process_telemetry(raw_telemetry)
    print(f"   Calculated PGA: {pga:.4f} m/s^2")
    print(f"   Estimated MMI:  {mmi}")
    print(f"   Alert Flag:     {alert}")
    print("   Evidence trail:")
    for ev in evidence:
        print(f"     - {ev}")

    # 2. Emergency Agent Pipeline Test
    print("\n2. Testing EmergencyAgent ingestion with mock event:")
    agent = EmergencyAgent()
    response = agent.process_event(MOCK_PLUM_STRONG_MOTION_EVENT)
    print(f"   Severity: {response.get('severity')} | Confidence: {response.get('confidence')}")
    print(f"   Alert Triggered: {response.get('plumAlertTriggered')} | Peak MMI: {response.get('peakMMI')}")
    print("   Actions generated:")
    for act in response.get("actions", []):
        print(f"     [{act.get('priority')}] {act.get('type')} on {act.get('resource')}: {act.get('detail')}")

    # 3. Multi-station Network PLUM Test
    print("\n3. Testing Multi-station PLUM detection & suppression:")
    network_data = get_mock_station_network_data()
    result = PLUMEngine.run(network_data, PlumConfig())
    print(f"   Detections confirmed: {len(result.detections)}")
    for d in result.detections:
        print(f"     -> Trigger: {d.trigger_station} ({d.trigger_mmi}) confirmed by {d.confirming_station} ({d.confirming_mmi}) at t={d.detection_time_s}s")
    print(f"   Suppressed triggers:  {len(result.suppressed)}")
    for s in result.suppressed:
        print(f"     -> Suppressed: {s.station} at t={s.trigger_second}s ({s.reason})")

    print("\nPLUM mock verification finished successfully.")
