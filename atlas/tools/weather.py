# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/weather.py — MCP tools: local and city weather via Open-Meteo."""

from __future__ import annotations

import logging

import httpx
from mcp.server.fastmcp import FastMCP  # type: ignore[import]

logger = logging.getLogger(__name__)
mcp = FastMCP(name="weather")

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_IP_API_URL = "http://ip-api.com/json/"

# WMO weather interpretation codes → French descriptions
_WMO: dict[int, str] = {
    0: "ciel dégagé", 1: "principalement dégagé", 2: "partiellement nuageux",
    3: "couvert", 45: "brouillard", 48: "brouillard givrant",
    51: "bruine légère", 53: "bruine modérée", 55: "bruine dense",
    61: "pluie légère", 63: "pluie modérée", 65: "pluie forte",
    71: "neige légère", 73: "neige modérée", 75: "neige forte",
    77: "grains de neige", 80: "averses légères", 81: "averses modérées",
    82: "averses violentes", 85: "averses de neige légères",
    86: "averses de neige fortes", 95: "orage", 96: "orage avec grêle légère",
    99: "orage avec grêle forte",
}


def _fetch_forecast(lat: float, lon: float) -> str:
    """Fetch current weather for the given coordinates from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,apparent_temperature,weathercode,windspeed_10m",
        "wind_speed_unit": "kmh",
        "timezone": "auto",
    }
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
        f"{desc.capitalize()}, {temp}°C (ressenti {feels}°C), "
        f"vent {wind} km/h"
    )


@mcp.tool()
def get_local_weather() -> str:
    """Return the current weather at the user's approximate location (via IP).

    Uses Open-Meteo (no API key required).

    Example output: ``Partiellement nuageux, 18°C (ressenti 16°C), vent 12 km/h``
    """
    try:
        geo = httpx.get(_IP_API_URL, timeout=5.0).json()
        lat = float(geo["lat"])
        lon = float(geo["lon"])
        city = geo.get("city", "")
        result = _fetch_forecast(lat, lon)
        return f"{city} — {result}" if city else result
    except Exception as exc:
        logger.warning("Local weather failed: %s", exc)
        return f"[Weather unavailable: {exc}]"


@mcp.tool()
def get_city_weather(city: str) -> str:
    """Return the current weather for a named city.

    Args:
        city: City name (e.g. ``"Paris"``, ``"Tokyo"``).

    Example output: ``Paris — Couvert, 12°C (ressenti 9°C), vent 20 km/h``
    """
    try:
        geo_resp = httpx.get(
            _GEOCODE_URL,
            params={"name": city, "count": 1, "language": "fr"},
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
