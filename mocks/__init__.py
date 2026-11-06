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
from mocks.mock_flood import (
    MOCK_FLOOD_CRITICAL_SCENARIO,
    MOCK_FLOOD_WATCH_SCENARIO,
    MOCK_FLOOD_DRY_SCENARIO,
    MOCK_FLOOD_MISSING_INPUTS_SCENARIO,
    MOCK_FLOOD_STALE_SCENARIO,
    get_mock_flood_scenario,
)
from mocks.mock_catalog_quake import (
    MOCK_QUAKE_CRITICAL_NEARBY,
    MOCK_QUAKE_WATCH_NEARBY,
    MOCK_QUAKE_DISTANT_STRONG,
    get_mock_catalog_quake,
)
from mocks.mock_tsunami import (
    MOCK_TSUNAMI_OFFSHORE_QUAKE_EVENT,
    MOCK_SEA_LEVEL_ANOMALY_EVENT,
    MOCK_ONSHORE_QUAKE_EVENT,
    get_mock_tsunami_event,
)
from mocks.mock_storm import (
    MOCK_STORM_GUST_WARNING_EVENT,
    MOCK_STORM_RADAR_ROTATION_EVENT,
    MOCK_STORM_OFFICIAL_ALERT_EVENT,
    MOCK_STORM_CALM_EVENT,
    get_mock_storm_event,
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
    "MOCK_FLOOD_CRITICAL_SCENARIO",
    "MOCK_FLOOD_WATCH_SCENARIO",
    "MOCK_FLOOD_DRY_SCENARIO",
    "MOCK_FLOOD_MISSING_INPUTS_SCENARIO",
    "MOCK_FLOOD_STALE_SCENARIO",
    "get_mock_flood_scenario",
    "MOCK_QUAKE_CRITICAL_NEARBY",
    "MOCK_QUAKE_WATCH_NEARBY",
    "MOCK_QUAKE_DISTANT_STRONG",
    "get_mock_catalog_quake",
    "MOCK_TSUNAMI_OFFSHORE_QUAKE_EVENT",
    "MOCK_SEA_LEVEL_ANOMALY_EVENT",
    "MOCK_ONSHORE_QUAKE_EVENT",
    "get_mock_tsunami_event",
    "MOCK_STORM_GUST_WARNING_EVENT",
    "MOCK_STORM_RADAR_ROTATION_EVENT",
    "MOCK_STORM_OFFICIAL_ALERT_EVENT",
    "MOCK_STORM_CALM_EVENT",
    "get_mock_storm_event",
]
