# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/tts.py
=================
Text-to-speech module — wraps the macOS ``say`` command.

The ``TTS`` class cleans the input text before speaking: it strips Markdown
formatting (bold, italic, headers, code blocks, bullet lists) so the model's
voice-formatted responses never produce audible syntax characters.

The audio gate is closed for the duration of playback so the VAD microphone
does not capture Atlas's own voice and trigger a spurious wake-word event.

Usage::

    from atlas.core.tts import TTS
    from atlas.config import Config

    tts = TTS(Config.from_env())
    await tts.speak("Bonjour, comment puis-je vous aider ?")
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess

from atlas.config import Config
from atlas.core.audio_gate import gate

logger = logging.getLogger(__name__)

# ── Markdown stripping ────────────────────────────────────────────────────────
# Ordered from most specific to most general so substitutions don't interfere.

_MD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Fenced code blocks (``` or ~~~)
    (re.compile(r"```.*?```", re.DOTALL), " "),
    (re.compile(r"~~~.*?~~~", re.DOTALL), " "),
    # Inline code
    (re.compile(r"`[^`]+`"), " "),
    # Bold + italic combinations ***text*** / ___text___
    (re.compile(r"\*{3}(.+?)\*{3}", re.DOTALL), r"\1"),
    (re.compile(r"_{3}(.+?)_{3}", re.DOTALL), r"\1"),
    # Bold **text** / __text__
    (re.compile(r"\*{2}(.+?)\*{2}", re.DOTALL), r"\1"),
    (re.compile(r"_{2}(.+?)_{2}", re.DOTALL), r"\1"),
    # Italic *text* / _text_
    (re.compile(r"\*(.+?)\*"), r"\1"),
    (re.compile(r"_(.+?)_"), r"\1"),
    # ATX headers (# Title)
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    # Unordered list markers (- item / * item / + item)
    (re.compile(r"^[\-\*\+]\s+", re.MULTILINE), ""),
    # Ordered list markers (1. item)
    (re.compile(r"^\d+\.\s+", re.MULTILINE), ""),
    # Horizontal rules (--- / *** / ___)
    (re.compile(r"^[-\*_]{3,}\s*$", re.MULTILINE), ""),
    # Blockquotes (> text)
    (re.compile(r"^>\s?", re.MULTILINE), ""),
    # Links [text](url) → text
    (re.compile(r"\[(.+?)\]\(.+?\)"), r"\1"),
    # Images ![alt](url) → alt
    (re.compile(r"!\[(.+?)\]\(.+?\)"), r"\1"),
    # Wikilinks [[note]] → note
    (re.compile(r"\[\[(.+?)\]\]"), r"\1"),
    # Remaining angle-bracket tags
    (re.compile(r"<[^>]+>"), " "),
    # Multiple spaces / blank lines → single space
    (re.compile(r"\s{2,}"), " "),
]


def _strip_markdown(text: str) -> str:
    """Remove Markdown formatting characters from *text*, leaving plain prose."""
    for pattern, replacement in _MD_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


# ── TTS class ─────────────────────────────────────────────────────────────────

class TTS:
    """Async TTS wrapper around the macOS ``say`` command.

    Args:
        config: Atlas runtime configuration.  Uses ``config.tts_rate`` for the
                optional speaking rate override.
    """

    def __init__(self, config: Config) -> None:
        self._rate: int | None = config.tts_rate

    async def speak(self, text: str) -> None:
        """Speak *text* aloud, closing the audio gate for the duration.

        The text is cleaned of Markdown formatting before being passed to
        ``say``.  An empty string after cleaning is a no-op.

        Args:
            text: Raw text from the LLM (may contain Markdown).
        """
        clean = _strip_markdown(text)
        if not clean:
            logger.debug("TTS: nothing to speak after stripping Markdown")
            return

        logger.info("TTS ▶  %r", clean[:120] + ("…" if len(clean) > 120 else ""))

        async with gate.closed():
            await self._run_say(clean)

    async def _run_say(self, text: str) -> None:
        """Invoke the macOS ``say`` command asynchronously."""
        cmd = ["say"]
        if self._rate is not None:
            cmd += ["-r", str(self._rate)]
        cmd.append(text)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0 and stderr:
                logger.warning("say exited %d: %s", proc.returncode, stderr.decode().strip())
        except FileNotFoundError:
            logger.error("`say` command not found — TTS requires macOS")
        except Exception as exc:
            logger.error("TTS subprocess error: %s", exc)
