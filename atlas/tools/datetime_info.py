# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/datetime_info.py — MCP tool: current date and time."""

from __future__ import annotations

from datetime import datetime

from mcp.server.fastmcp import FastMCP  # type: ignore[import]

mcp = FastMCP(name="datetime_info")


@mcp.tool()
def get_datetime() -> str:
    """Return the current local date and time in a human-readable format.

    Example output: ``Dimanche 25 mai 2026, 14:32:07``
    """
    now = datetime.now()
    # French day and month names
    days_fr = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    months_fr = [
        "janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre",
    ]
    day_name = days_fr[now.weekday()]
    month_name = months_fr[now.month - 1]
    return (
        f"{day_name} {now.day} {month_name} {now.year}, "
        f"{now.strftime('%H:%M:%S')}"
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
