# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/_location.py — Private helper: macOS CoreLocation + offline reverse geocoding.

Not an MCP server.  Imported by ``geoposition.py`` and ``weather.py``.
"""

from __future__ import annotations

import objc
import reverse_geocoder  # type: ignore[import]
from CoreLocation import (  # type: ignore[import]
    CLLocationManager,
    kCLLocationAccuracyBest,
)
from Foundation import NSDate, NSRunLoop  # type: ignore[import]


class _LocationDelegate(objc.lookUpClass("NSObject")):  # type: ignore[misc]
    def init(self) -> "_LocationDelegate":
        self = objc.super(_LocationDelegate, self).init()
        if self:
            self.location = None
            self.error: str | None = None
            self.is_done: bool = False
        return self

    def locationManager_didUpdateLocations_(  # noqa: N802
        self, manager: object, locations: object
    ) -> None:
        if locations:
            self.location = locations.lastObject()  # type: ignore[union-attr]
            self.is_done = True

    def locationManager_didFailWithError_(  # noqa: N802
        self, manager: object, error: object
    ) -> None:
        self.error = error.localizedDescription()  # type: ignore[union-attr]
        self.is_done = True


def get_mac_coordinates() -> tuple[float, float]:
    """Block until CoreLocation delivers a GPS fix.

    Returns:
        ``(latitude, longitude)`` as floats.

    Raises:
        RuntimeError: If CoreLocation reports an error (e.g. permission denied).
    """
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


def reverse_geocode(lat: float, lon: float) -> str:
    """Offline GeoNames reverse lookup.

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
