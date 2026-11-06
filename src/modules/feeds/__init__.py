"""
External Data Feed Adapters for City Brain.
Fetches external telemetry and normalizes it into CityEvent objects.
Engines themselves do not call the network; adapters emit CityEvents into the Orchestrator.
"""
from .open_meteo import OpenMeteoAdapter
from .usgs import USGSAdapter
from .emsc import EMSCAdapter
from .glofas import GloFASAdapter
from .weather_alert import WeatherAlertAdapter

__all__ = [
    "OpenMeteoAdapter",
    "USGSAdapter",
    "EMSCAdapter",
    "GloFASAdapter",
    "WeatherAlertAdapter",
]

