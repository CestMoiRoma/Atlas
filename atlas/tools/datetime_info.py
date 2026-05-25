# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/datetime_info.py — MCP tool: current date and time."""

from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP  # type: ignore[import]

load_dotenv()

mcp = FastMCP(name="datetime_info")


@mcp.tool()
def get_datetime() -> str:
    """Return the current local date and time in a human-readable format.

    Respects the ``TIME_FORMAT`` environment variable:
      - ``"24h"`` (default) → ``HH:MM:SS``
      - ``"12h"``           → ``HH:MM:SS AM/PM``

    Example output (24h): ``Sunday, May 25 2026, 14:32:07``
    Example output (12h): ``Sunday, May 25 2026, 02:32:07 PM``
    """
    now = datetime.now()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    day_name = days[now.weekday()]
    month_name = months[now.month - 1]

    time_format = os.getenv("TIME_FORMAT", "24h").lower()
    if time_format == "12h":
        time_str = now.strftime("%I:%M:%S %p")
    else:
        time_str = now.strftime("%H:%M:%S")

    return f"{day_name}, {month_name} {now.day} {now.year}, {time_str}"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
