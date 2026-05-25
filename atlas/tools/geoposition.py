# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/geoposition.py — MCP tool: current city/region/country via macOS CoreLocation."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP  # type: ignore[import]

from atlas.tools._location import get_mac_coordinates, reverse_geocode

logger = logging.getLogger(__name__)
mcp = FastMCP(name="geoposition")


@mcp.tool()
def get_current_place() -> str:
    """Return the current city, region, and country using macOS CoreLocation.

    Uses the device GPS/Wi-Fi fix (accurate) and an offline GeoNames reverse
    geocoding database (no internet required after install).  Returns an error
    string if CoreLocation permission is denied or the fix times out.

    Example output: ``Lyon, Auvergne-Rhône-Alpes, FR``
    """
    try:
        lat, lon = get_mac_coordinates()
        return reverse_geocode(lat, lon)
    except Exception as exc:
        logger.warning("Geolocation failed: %s", exc)
        return f"[Geolocation unavailable: {exc}]"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
