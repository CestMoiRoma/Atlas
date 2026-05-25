# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/audio_gate.py
========================
Async-first audio gate — prevents the VAD from capturing Atlas's own TTS output.

The gate is a shared :class:`asyncio.Event` that is *cleared* (closed) while
Atlas is speaking and *set* (open) the rest of the time.  The STT recording
loop checks the gate before each audio chunk; if the gate is closed the chunk
is discarded, avoiding a feedback loop where Atlas hears itself and triggers
another turn.

Usage (from the TTS module)::

    async with audio_gate.closed():
        await speak_subprocess(text)

Usage (from the STT / VAD loop)::

    if audio_gate.is_open():
        process_chunk(chunk)

The gate starts open so recording works immediately on startup.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AudioGate:
    """Async-safe gate that mutes VAD recording during TTS playback.

    The internal :class:`asyncio.Event` is *set* when the gate is open
    (normal operation) and *cleared* when the gate is closed (TTS playing).
    This follows the standard ``asyncio.Event`` convention: ``wait()`` returns
    when the event is set (i.e. gate open / recording allowed).
    """

    def __init__(self) -> None:
        self._event: asyncio.Event = asyncio.Event()
        self._event.set()  # gate starts open

    # ── State queries ─────────────────────────────────────────────────────────

    def is_open(self) -> bool:
        """Return ``True`` if recording is currently allowed."""
        return self._event.is_set()

    async def wait_until_open(self) -> None:
        """Suspend the current coroutine until the gate is open."""
        await self._event.wait()

    # ── Manual control ────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the gate — VAD recording is allowed."""
        self._event.set()

    def close(self) -> None:
        """Close the gate — VAD recording is suppressed."""
        self._event.clear()

    # ── Context manager ───────────────────────────────────────────────────────

    @asynccontextmanager
    async def closed(self) -> AsyncIterator[None]:
        """Async context manager that closes the gate for its duration.

        Guarantees the gate is re-opened even if the body raises an exception::

            async with gate.closed():
                await tts_subprocess(text)
            # gate is open again here
        """
        self.close()
        try:
            yield
        finally:
            self.open()


#: Module-level singleton — shared across all pipeline modules.
#: Import this instance directly rather than creating your own::
#:
#:     from atlas.core.audio_gate import gate
gate: AudioGate = AudioGate()
