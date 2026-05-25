# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/geoposition.py — MCP tool: current city/country via IP geolocation."""

from __future__ import annotations

import logging

import httpx
from mcp.server.fastmcp import FastMCP  # type: ignore[import]

logger = logging.getLogger(__name__)
mcp = FastMCP(name="geoposition")


@mcp.tool()
def get_current_place() -> str:
    """Return the current city and country based on the public IP address.

    Uses the ip-api.com free endpoint (no API key required).  Returns an
    error string if the request fails or times out.

    Example output: ``Lyon, France``
    """
    try:
        resp = httpx.get("http://ip-api.com/json/", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        city = data.get("city", "?")
        country = data.get("country", "?")
        return f"{city}, {country}"
    except Exception as exc:
        logger.warning("Geolocation failed: %s", exc)
        return f"[Geolocation unavailable: {exc}]"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
