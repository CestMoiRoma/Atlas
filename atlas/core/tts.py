# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/tts.py
=================
Text-to-speech module with platform fallbacks.

Backend selection (first available wins):

===========  ==================================================
Platform     Backend
===========  ==================================================
macOS        ``say`` (built-in, accurate WPM rate control)
Linux        ``espeak-ng`` → ``espeak`` → ``spd-say``
Windows      PowerShell + SAPI5 (``System.Speech``, always present)
===========  ==================================================

The ``TTS`` class cleans the input text before speaking: it strips Markdown
formatting (bold, italic, headers, code blocks, bullet lists) so the model's
voice-formatted responses never produce audible syntax characters.

The audio gate is closed for the duration of playback so the VAD microphone
does not capture Atlas's own voice and trigger a spurious wake-word event.

Usage::

    from atlas.core.tts import TTS
    from atlas.config import Config

    tts = TTS(Config.from_env())
    await tts.speak("Hello, how can I help you?")
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import shutil
import sys

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


# ── Backend detection ─────────────────────────────────────────────────────────

def _detect_backend() -> str:
    """Return the identifier of the best available TTS backend.

    Resolution order:
    - macOS  → ``"say"``
    - Windows → ``"powershell"`` (SAPI5 via ``System.Speech``)
    - Linux/other → first of ``espeak-ng``, ``espeak``, ``spd-say`` found in PATH
    - Nothing found → ``"none"``
    """
    if sys.platform == "darwin":
        return "say"
    if sys.platform == "win32":
        return "powershell"
    for candidate in ("espeak-ng", "espeak", "spd-say"):
        if shutil.which(candidate):
            return candidate
    return "none"


_BACKEND: str = _detect_backend()


# ── TTS class ─────────────────────────────────────────────────────────────────

class TTS:
    """Async TTS wrapper with automatic platform backend selection.

    Args:
        config: Atlas runtime configuration.  Uses ``config.tts_rate`` for the
                optional speaking rate override (WPM on macOS/espeak, ignored on
                Windows SAPI and spd-say).
    """

    def __init__(self, config: Config) -> None:
        self._rate: int | None = config.tts_rate
        logger.debug("TTS backend: %s", _BACKEND)

    async def speak(self, text: str) -> None:
        """Speak *text* aloud, closing the audio gate for the duration.

        The text is cleaned of Markdown formatting before being passed to the
        TTS backend.  An empty string after cleaning is a no-op.

        Args:
            text: Raw text from the LLM (may contain Markdown).
        """
        clean = _strip_markdown(text)
        if not clean:
            logger.debug("TTS: nothing to speak after stripping Markdown")
            return

        logger.info("TTS ▶  %r", clean[:120] + ("…" if len(clean) > 120 else ""))

        async with gate.closed():
            await self._dispatch(clean)

    async def _dispatch(self, text: str) -> None:
        """Build and run the platform-specific TTS command."""
        cmd: list[str]

        if _BACKEND == "say":
            # macOS built-in — -r sets words per minute
            cmd = ["say"]
            if self._rate is not None:
                cmd += ["-r", str(self._rate)]
            cmd.append(text)

        elif _BACKEND in ("espeak-ng", "espeak"):
            # Linux — -s sets speed in words per minute (same scale as `say -r`)
            cmd = [_BACKEND]
            if self._rate is not None:
                cmd += ["-s", str(self._rate)]
            cmd.append(text)

        elif _BACKEND == "spd-say":
            # Linux speech-dispatcher — --wait blocks until playback ends
            # Rate uses a different scale (-100..100); WPM conversion is not
            # straightforward so tts_rate is ignored for this backend.
            cmd = ["spd-say", "--wait", text]

        elif _BACKEND == "powershell":
            # Windows — SAPI5 via System.Speech (always present on Windows 7+).
            # Single quotes in text are escaped by doubling (PowerShell convention).
            # The whole script is base64-encoded to avoid any shell quoting issues.
            escaped = text.replace("'", "''")
            script = (
                "Add-Type -AssemblyName System.Speech; "
                f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{escaped}')"
            )
            encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
            cmd = ["powershell", "-EncodedCommand", encoded]

        else:
            logger.error(
                "No TTS backend available (platform=%s). "
                "Install espeak-ng on Linux: sudo apt install espeak-ng",
                sys.platform,
            )
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0 and stderr:
                logger.warning(
                    "TTS backend %r exited %d: %s",
                    _BACKEND, proc.returncode, stderr.decode().strip(),
                )
        except FileNotFoundError:
            logger.error("TTS backend %r not found in PATH", _BACKEND)
        except Exception as exc:
            logger.error("TTS subprocess error: %s", exc)
