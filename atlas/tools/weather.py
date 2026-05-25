# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/weather.py — MCP tools: local and city weather via Open-Meteo."""

from __future__ import annotations

import logging
import os

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP  # type: ignore[import]

from atlas.tools._location import get_mac_coordinates, reverse_geocode

load_dotenv()

logger = logging.getLogger(__name__)
mcp = FastMCP(name="weather")

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes → English descriptions
_WMO: dict[int, str] = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy",
    3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "moderate showers",
    82: "violent showers", 85: "light snow showers",
    86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}


def _fetch_forecast(lat: float, lon: float) -> str:
    """Fetch current weather for the given coordinates from Open-Meteo.

    Respects the ``TEMPERATURE_UNIT`` environment variable:
      - ``"C"`` (default) → Celsius, Open-Meteo default
      - ``"F"``           → Fahrenheit, passed to Open-Meteo natively
    """
    temperature_unit = os.getenv("TEMPERATURE_UNIT", "C").upper()
    unit_str = "°F" if temperature_unit == "F" else "°C"

    params: dict[str, object] = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,apparent_temperature,weathercode,windspeed_10m",
        "wind_speed_unit": "kmh",
        "timezone": "auto",
    }
    if temperature_unit == "F":
        params["temperature_unit"] = "fahrenheit"

    resp = httpx.get(_FORECAST_URL, params=params, timeout=8.0)
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current", {})

    temp = current.get("temperature_2m", "?")
    feels = current.get("apparent_temperature", "?")
    code = int(current.get("weathercode", -1))
    wind = current.get("windspeed_10m", "?")
    desc = _WMO.get(code, f"code {code}")

    return (
        f"{desc.capitalize()}, {temp}{unit_str} (feels like {feels}{unit_str}), "
        f"wind {wind} km/h"
    )


@mcp.tool()
def get_local_weather() -> str:
    """Return the current weather at the user's actual location via macOS CoreLocation.

    Uses a real GPS/Wi-Fi fix (accurate) rather than IP geolocation.
    Weather data comes from Open-Meteo (no API key required).
    Temperature unit follows the ``TEMPERATURE_UNIT`` env variable (``C`` or ``F``).

    Example output: ``Lyon — Partly cloudy, 18°C (feels like 16°C), wind 12 km/h``
    """
    try:
        lat, lon = get_mac_coordinates()
        place = reverse_geocode(lat, lon)
        result = _fetch_forecast(lat, lon)
        return f"{place} — {result}"
    except Exception as exc:
        logger.warning("Local weather failed: %s", exc)
        return f"[Weather unavailable: {exc}]"


@mcp.tool()
def get_city_weather(city: str) -> str:
    """Return the current weather for a named city.

    Args:
        city: City name (e.g. ``"Paris"``, ``"Tokyo"``).

    Temperature unit follows the ``TEMPERATURE_UNIT`` env variable
    (``C`` or ``F``).

    Example output: ``Paris, Île-de-France, FR — Overcast, 12°C (feels like 9°C), wind 20 km/h``
    """
    try:
        geo_resp = httpx.get(
            _GEOCODE_URL,
            params={"name": city, "count": 1, "language": "en"},
            timeout=5.0,
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results", [])
        if not results:
            return f"[City not found: {city!r}]"
        first = results[0]
        lat = float(first["latitude"])
        lon = float(first["longitude"])
        display_name = first.get("name", city)
        result = _fetch_forecast(lat, lon)
        return f"{display_name} — {result}"
    except Exception as exc:
        logger.warning("City weather failed for %r: %s", city, exc)
        return f"[Weather unavailable for {city!r}: {exc}]"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
