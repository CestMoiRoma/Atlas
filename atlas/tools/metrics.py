# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/metrics.py — MCP tool: macOS system metrics (CPU, RAM, disk)."""

from __future__ import annotations

import psutil
from mcp.server.fastmcp import FastMCP  # type: ignore[import]

mcp = FastMCP(name="metrics")


@mcp.tool()
def get_mac_metrics() -> str:
    """Return current CPU usage, RAM usage, and disk usage for the Mac.

    Reads live system counters via psutil — no elevated privileges required.

    Example output::

        CPU : 34 %
        RAM : 11.2 GB used / 16.0 GB (70 %)
        Disk: 234 GB used / 500 GB (47 %)
    """
    cpu_pct = psutil.cpu_percent(interval=0.5)

    mem = psutil.virtual_memory()
    ram_used_gb = mem.used / 1_073_741_824
    ram_total_gb = mem.total / 1_073_741_824

    disk = psutil.disk_usage("/")
    disk_used_gb = disk.used / 1_073_741_824
    disk_total_gb = disk.total / 1_073_741_824

    return (
        f"CPU  : {cpu_pct:.0f} %\n"
        f"RAM  : {ram_used_gb:.1f} GB used / {ram_total_gb:.1f} GB ({mem.percent:.0f} %)\n"
        f"Disk : {disk_used_gb:.0f} GB used / {disk_total_gb:.0f} GB ({disk.percent:.0f} %)"
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
