"""
Mock fixtures and datasets for City Brain modules.
"""
from mocks.mock_plum import (
    MOCK_PLUM_STRONG_MOTION_EVENT,
    MOCK_PLUM_MINOR_MOTION_EVENT,
    MOCK_PLUM_CLIPPING_ANOMALY_EVENT,
    get_mock_plum_event,
    get_mock_station_network_data,
)
from mocks.mock_crime_hotspot import (
    MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT,
    MOCK_CRIME_ELEVATED_RISK_EVENT,
    MOCK_CRIME_LOW_RISK_EVENT,
    get_mock_crime_event,
)

__all__ = [
    "MOCK_PLUM_STRONG_MOTION_EVENT",
    "MOCK_PLUM_MINOR_MOTION_EVENT",
    "MOCK_PLUM_CLIPPING_ANOMALY_EVENT",
    "get_mock_plum_event",
    "get_mock_station_network_data",
    "MOCK_CRIME_GUNSHOT_HIGH_RISK_EVENT",
    "MOCK_CRIME_ELEVATED_RISK_EVENT",
    "MOCK_CRIME_LOW_RISK_EVENT",
    "get_mock_crime_event",
]
