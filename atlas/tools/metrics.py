# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/metrics.py — MCP tool: live system metrics via psutil + ioreg."""

from __future__ import annotations

import datetime
import plistlib
import subprocess

import psutil
from mcp.server.fastmcp import FastMCP  # type: ignore[import]

mcp = FastMCP(
    name="metrics",
    instructions=(
        "Returns real-time Mac system metrics. "
        "Call for: 'stats du mac', 'état du mac', 'comment va le mac', "
        "'cpu', 'ram', 'mémoire', 'charge du système', 'utilisation', 'performances', "
        "'combien de ram', 'disque', 'batterie', 'depuis quand le mac est allumé'."
    ),
)


def _gpu_usage_pct() -> str:
    """Read GPU utilisation via ioreg.

    Tries IOAccelerator (Intel) then AGXAccelerator (Apple Silicon).
    Returns the percentage as a string, or ``"N/A"`` if unavailable.
    """
    for cls in ("IOAccelerator", "AGXAccelerator"):
        try:
            raw = subprocess.check_output(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", cls, "-a"],
                timeout=5,
            )
            if not raw.strip():
                continue
            for entry in plistlib.loads(raw):
                util = entry.get("PerformanceStatistics", {}).get("Device Utilization %")
                if util is not None:
                    return f"{util} %"
        except Exception:  # noqa: BLE001
            continue
    return "N/A"


@mcp.tool()
def get_mac_metrics() -> dict:
    """Return live system metrics for the Mac.

    Collects CPU, GPU, RAM, disk, network, battery, thread/process counts, and
    uptime.  No elevated privileges required.

    Returns:
        A dict with keys:
        ``cpu_pct``, ``gpu_pct``, ``thread_count``, ``process_count``,
        ``ram_total_gb``, ``ram_used_gb``, ``ram_pct``,
        ``disk_total_gb``, ``disk_used_gb``, ``disk_pct``,
        ``net_sent_mb``, ``net_recv_mb``,
        ``battery_pct``, ``battery_plugged``, ``uptime``.
    """
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    net = psutil.net_io_counters()
    gb = 1024 ** 3
    mb = 1024 ** 2

    thread_count = sum(
        p.info["num_threads"]
        for p in psutil.process_iter(["num_threads"])
        if p.info["num_threads"]
    )

    battery = psutil.sensors_battery()
    uptime = str(
        datetime.timedelta(
            seconds=int(datetime.datetime.now().timestamp() - psutil.boot_time())
        )
    )

    return {
        "cpu_pct":        psutil.cpu_percent(interval=0.5),
        "gpu_pct":        _gpu_usage_pct(),
        "thread_count":   thread_count,
        "process_count":  len(psutil.pids()),
        "ram_total_gb":   round(vm.total / gb, 2),
        "ram_used_gb":    round(vm.used / gb, 2),
        "ram_pct":        vm.percent,
        "disk_total_gb":  round(disk.total / gb, 2),
        "disk_used_gb":   round(disk.used / gb, 2),
        "disk_pct":       disk.percent,
        "net_sent_mb":    round(net.bytes_sent / mb, 2),
        "net_recv_mb":    round(net.bytes_recv / mb, 2),
        "battery_pct":    battery.percent if battery else "N/A",
        "battery_plugged": battery.power_plugged if battery else "N/A",
        "uptime":         uptime,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
