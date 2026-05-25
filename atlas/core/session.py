# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/session.py
=====================
Session journal — writes a crash-safe Markdown log of every conversation.

One ``SessionLog`` instance covers one conversation.  The log is created when
``SessionLog`` is instantiated and closed (with its footer and Obsidian graph
wiring) when ``close()`` is called.  Each ``append_turn()`` call appends
immediately to disk, so the file is always readable even if Atlas crashes mid-
session.

Obsidian graph integration
--------------------------
On ``close()``:

* The session file gains a ``[[Sessions]]`` backlink to the vault's central
  session index node.
* ``<vault>/Sessions.md`` gains a forward link to this session file so the
  full conversation history appears as a connected graph in Obsidian.

File format
-----------
::

    ---
    type: session
    date: 2026-05-25
    start: 14:32:01
    ---

    # Session du 2026-05-25 à 14:32

    ## Tour 1 · 14:32:18 · Roma
    > Ce que l'utilisateur a dit

    🔧 `memory__memory_write` · `memory__memory_link`

    Réponse d'Atlas.

    ---

    *Session fermée à 14:45:00 — 3 tours, 12 min, utilisateur : Roma*

    [[Sessions]]
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from atlas.config import Config

logger = logging.getLogger(__name__)

_SESSIONS_HUB_NAME = "Sessions.md"
_SESSIONS_DIR_NAME = "Sessions"


class SessionLog:
    """Crash-safe Markdown session journal for a single conversation.

    Args:
        config: Atlas runtime configuration.  Uses ``config.vault_path`` to
                determine where to write the session files.
    """

    def __init__(self, config: Config) -> None:
        self._vault = config.vault_path.resolve()
        sessions_dir = self._vault / _SESSIONS_DIR_NAME
        sessions_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        self._stem = now.strftime("%Y-%m-%d_%H%M%S")
        self._path = sessions_dir / f"{self._stem}.md"
        self._turn: int = 0
        self._speakers: set[str] = set()
        self._start = now

        self._ensure_sessions_hub()
        self._write_header(now)
        logger.info("Session log opened → %s", self._path)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Absolute path to the session Markdown file."""
        return self._path

    @property
    def stem(self) -> str:
        """Filename stem (without ``.md``), e.g. ``2026-05-25_143201``."""
        return self._stem

    @property
    def turn_count(self) -> int:
        """Number of turns appended so far."""
        return self._turn

    # ── Writing ───────────────────────────────────────────────────────────────

    def append_turn(
        self,
        speaker: str,
        user_text: str,
        tools_called: list[str],
        reply: str,
    ) -> None:
        """Append one completed conversation turn to the session file.

        Written immediately to disk — the file is always consistent even if
        Atlas crashes before ``close()`` is called.

        Args:
            speaker:      Display name of the identified speaker.
            user_text:    Raw transcription of the user's utterance.
            tools_called: Names of every MCP tool called during this turn.
            reply:        Final text reply produced by Atlas.
        """
        self._turn += 1
        self._speakers.add(speaker)
        ts = datetime.now().strftime("%H:%M:%S")
        tools_str = " · ".join(f"`{t}`" for t in tools_called) if tools_called else "—"

        block = (
            f"\n## Tour {self._turn} · {ts} · {speaker}\n"
            f"> {user_text}\n\n"
            f"🔧 {tools_str}\n\n"
            f"{reply}\n\n"
            f"---\n"
        )
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(block)

    def close(self) -> None:
        """Write the session footer and wire it into the Obsidian graph.

        Idempotent — safe to call more than once (subsequent calls are no-ops
        because the footer marker will already be present).
        """
        end = datetime.now()
        elapsed_min = int((end - self._start).total_seconds() // 60)
        speakers_str = ", ".join(sorted(self._speakers)) if self._speakers else "—"
        plural_tours = "s" if self._turn != 1 else ""
        plural_users = "s" if len(self._speakers) > 1 else ""

        footer = (
            f"\n*Session fermée à {end.strftime('%H:%M:%S')} — "
            f"{self._turn} tour{plural_tours}, "
            f"{elapsed_min} min, "
            f"utilisateur{plural_users} : {speakers_str}*\n"
            f"\n[[Sessions]]\n"
        )

        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(footer)

        self._register_in_hub()
        logger.info(
            "Session log closed (%d turns, %d min) → %s",
            self._turn, elapsed_min, self._path,
        )

    # ── Obsidian graph helpers ─────────────────────────────────────────────────

    def _ensure_sessions_hub(self) -> None:
        """Create ``Sessions.md`` hub node if it does not exist yet."""
        hub = self._vault / _SESSIONS_HUB_NAME
        if not hub.exists():
            hub.parent.mkdir(parents=True, exist_ok=True)
            hub.write_text(
                "---\ntype: sessions_index\n---\n\n"
                "# Sessions\n\n"
                "Index de toutes les conversations d'Atlas.\n\n"
                "## Journal\n",
                encoding="utf-8",
            )
            logger.info("Sessions hub created → %s", hub)

    def _register_in_hub(self) -> None:
        """Append a forward link to this session in ``Sessions.md``."""
        hub = self._vault / _SESSIONS_HUB_NAME
        try:
            self._ensure_sessions_hub()
            with hub.open("a", encoding="utf-8") as fh:
                fh.write(f"- [[{_SESSIONS_DIR_NAME}/{self._stem}]]\n")
        except Exception as exc:
            logger.warning("Could not update Sessions hub: %s", exc)

    def _write_header(self, now: datetime) -> None:
        """Write YAML frontmatter and the session title to a new file."""
        self._path.write_text(
            f"---\n"
            f"type: session\n"
            f"date: {now.strftime('%Y-%m-%d')}\n"
            f"start: {now.strftime('%H:%M:%S')}\n"
            f"---\n\n"
            f"# Session du {now.strftime('%Y-%m-%d')} à {now.strftime('%H:%M')}\n",
            encoding="utf-8",
        )
