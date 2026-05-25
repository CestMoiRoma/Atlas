# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/wake_word.py
=======================
Always-on wake word listener backed by ``livekit-wakeword`` (ONNX inference).

The listener runs on the default microphone and yields the keyword string each
time the configured model fires above the detection threshold.  Debouncing
prevents the same activation from firing multiple times in quick succession.

Usage::

    from atlas.core.wake_word import WakeWordListener
    from atlas.config import Config

    cfg = Config.from_env()
    listener = WakeWordListener(cfg)

    async for keyword in listener.listen():
        print(f"Wake word detected: {keyword!r}")
        break  # handle one activation then re-enter the loop
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from atlas.config import Config

logger = logging.getLogger(__name__)


class WakeWordListener:
    """Wraps ``livekit-wakeword`` for async use in the Atlas pipeline.

    Args:
        config: Atlas runtime configuration.  The relevant fields are
                ``wake_word_models``, ``wake_word_threshold``, and
                ``wake_word_debounce``.
    """

    def __init__(self, config: Config) -> None:
        self._models: list[Path] = config.wake_word_models
        self._threshold: float = config.wake_word_threshold
        self._debounce: float = config.wake_word_debounce
        self._ww_model: object = None  # WakeWordModel, loaded lazily

    async def _get_model(self) -> object:
        """Load the WakeWordModel once, reuse on subsequent calls."""
        if self._ww_model is None:
            from livekit.wakeword import WakeWordModel  # type: ignore[import]

            model_strs = [str(m) for m in self._models]
            logger.info("Loading wake word model(s): %s", model_strs)
            self._ww_model = await asyncio.to_thread(
                WakeWordModel, models=model_strs
            )
        return self._ww_model

    async def listen(self) -> AsyncIterator[str]:
        """Async generator that yields the keyword string on each detection.

        Runs forever until cancelled.  The caller is expected to ``break``
        after the first yield and re-enter the loop for the next activation::

            async for keyword in listener.listen():
                await handle_activation(keyword)
                break  # re-enters listen() on the next loop iteration
        """
        from livekit.wakeword import WakeWordListener as _LiveKitListener  # type: ignore[import]

        model = await self._get_model()
        logger.info(
            "Wake word listener starting — models=%s  threshold=%.2f",
            [str(m) for m in self._models],
            self._threshold,
        )

        async with _LiveKitListener(
            model,  # type: ignore[arg-type]
            threshold=self._threshold,
            debounce=self._debounce,
        ) as listener:
            while True:
                detection = await listener.wait_for_detection()
                keyword: str = getattr(detection, "name", "atlas")
                logger.info(
                    "Wake word detected: %r (confidence=%.2f)",
                    keyword,
                    getattr(detection, "confidence", 0.0),
                )
                yield keyword
