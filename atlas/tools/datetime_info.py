# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/datetime_info.py — MCP tool: current date and time."""

from __future__ import annotations

from datetime import datetime

from mcp.server.fastmcp import FastMCP  # type: ignore[import]

mcp = FastMCP(name="datetime_info")


@mcp.tool()
def get_datetime() -> str:
    """Return the current local date and time in a human-readable format.

    Example output: ``Sunday, May 25 2026, 14:32:07``
    """
    now = datetime.now()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    day_name = days[now.weekday()]
    month_name = months[now.month - 1]
    return (
        f"{day_name}, {month_name} {now.day} {now.year}, "
        f"{now.strftime('%H:%M:%S')}"
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
