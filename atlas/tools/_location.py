# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/_location.py — Private helper: GPS coordinates + offline reverse geocoding.

Strategy
--------
1. **macOS + pyobjc available** → CoreLocation (native GPS/Wi-Fi fix, accurate).
2. **CoreLocation unavailable or failed** → ip-api.com (IP-based, approximate fallback).

Not an MCP server.  Imported by ``geoposition.py`` and ``weather.py``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import httpx
import reverse_geocoder  # type: ignore[import]

logger = logging.getLogger(__name__)

_IP_API_URL = "http://ip-api.com/json/"

# ── CoreLocation (macOS only) ─────────────────────────────────────────────────

_CORELOCATION_AVAILABLE: bool = False

if sys.platform == "darwin":
    try:
        import objc  # type: ignore[import]
        from CoreLocation import (  # type: ignore[import]
            CLLocationManager,
            kCLLocationAccuracyBest,
        )
        from Foundation import NSDate, NSRunLoop  # type: ignore[import]

        _CORELOCATION_AVAILABLE = True
    except ImportError:
        logger.debug("pyobjc not available — CoreLocation disabled, using ip-api.com fallback")


if _CORELOCATION_AVAILABLE:

    class _LocationDelegate(objc.lookUpClass("NSObject")):  # type: ignore[misc]
        def init(self) -> "_LocationDelegate":
            self = objc.super(_LocationDelegate, self).init()
            if self:
                self.location: Any = None
                self.error: str | None = None
                self.is_done: bool = False
            return self

        def locationManager_didUpdateLocations_(  # noqa: N802
            self, manager: Any, locations: Any
        ) -> None:
            if locations:
                self.location = locations.lastObject()
                self.is_done = True

        def locationManager_didFailWithError_(  # noqa: N802
            self, manager: Any, error: Any
        ) -> None:
            self.error = error.localizedDescription()
            self.is_done = True

    def _corelocation_coordinates() -> tuple[float, float]:
        """Return (lat, lon) via macOS CoreLocation GPS/Wi-Fi fix."""
        with objc.autorelease_pool():
            manager = CLLocationManager.alloc().init()
            delegate = _LocationDelegate.alloc().init()
            manager.setDelegate_(delegate)
            manager.setDesiredAccuracy_(kCLLocationAccuracyBest)
            manager.startUpdatingLocation()

            run_loop = NSRunLoop.currentRunLoop()
            while not delegate.is_done:
                run_loop.runMode_beforeDate_(
                    "NSDefaultRunLoopMode",
                    NSDate.dateWithTimeIntervalSinceNow_(0.1),
                )

            manager.stopUpdatingLocation()

            if delegate.error:
                raise RuntimeError(f"CoreLocation error: {delegate.error}")

            coords = delegate.location.coordinate()
            return float(coords.latitude), float(coords.longitude)


# ── IP fallback (cross-platform) ──────────────────────────────────────────────

def _ip_api_coordinates() -> tuple[float, float]:
    """Return (lat, lon) from public IP geolocation via ip-api.com."""
    resp = httpx.get(_IP_API_URL, timeout=5.0)
    resp.raise_for_status()
    data = resp.json()
    return float(data["lat"]), float(data["lon"])


# ── Public API ────────────────────────────────────────────────────────────────

def get_mac_coordinates() -> tuple[float, float]:
    """Return ``(latitude, longitude)`` for the current device.

    Resolution order:
    1. macOS + pyobjc → CoreLocation (GPS/Wi-Fi, accurate).
    2. Otherwise, or if CoreLocation raises → ip-api.com (IP-based, approximate).

    Raises:
        RuntimeError: If every available method fails.
    """
    if _CORELOCATION_AVAILABLE:
        try:
            return _corelocation_coordinates()
        except Exception as exc:
            logger.warning("CoreLocation failed (%s) — falling back to ip-api.com", exc)

    try:
        return _ip_api_coordinates()
    except Exception as exc:
        raise RuntimeError(f"All geolocation methods failed: {exc}") from exc


def reverse_geocode(lat: float, lon: float) -> str:
    """Offline GeoNames reverse lookup (cross-platform).

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        Human-readable place string, e.g. ``"Lyon, Auvergne-Rhône-Alpes, FR"``.
    """
    results = reverse_geocoder.search((lat, lon), verbose=False)
    if not results:
        return "Unknown location"
    r = results[0]
    parts = [p for p in [r.get("name"), r.get("admin1"), r.get("cc")] if p]
    return ", ".join(parts) if parts else "Unknown location"
