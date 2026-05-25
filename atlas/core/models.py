# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/models.py
====================
Shared data models used across the Atlas pipeline.

All classes are plain dataclasses — no external dependencies — so they can be
imported anywhere in the codebase without triggering heavyweight loads
(SpeechBrain, sounddevice, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class User:
    """A registered Atlas user with their voice embedding and profile data.

    Attributes:
        id:                 Database primary key. ``0`` means the guest user.
        name:               Display name (e.g. ``"Roma"``).
        user_tag:           Obsidian tag used to attribute vault notes
                            (e.g. ``"roma"``).
        age:                Optional age in years.
        gender:             Optional gender string.
        profession:         Optional occupation string.
        preferred_address:  Primary nickname Atlas uses in TTS (e.g. ``"chef"``).
        other_addresses:    Additional nicknames Atlas may rotate through.
        embedding:          128-dim (or 192-dim) float32 ECAPA-TDNN speaker
                            embedding. ``None`` until at least one voice sample
                            has been recorded.
    """

    id: int
    name: str
    user_tag: str
    age: int | None = None
    gender: str | None = None
    profession: str | None = None
    preferred_address: str = ""
    other_addresses: list[str] = field(default_factory=list)
    embedding: "np.ndarray | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.preferred_address:
            self.preferred_address = self.name

    @property
    def is_guest(self) -> bool:
        """True when this represents an unrecognised speaker."""
        return self.id == 0

    @property
    def all_addresses(self) -> list[str]:
        """Primary address followed by all alternates."""
        return [self.preferred_address] + self.other_addresses


@dataclass(frozen=True)
class SpeakerMatch:
    """Result of a speaker identification attempt.

    Attributes:
        user:   The best-matching :class:`User`, or the guest sentinel if no
                user scored above the fallback floor.
        score:  Cosine similarity (0.0–1.0) of the audio against the matched
                user's embedding. ``0.0`` for the guest sentinel.
        method: How the match was determined:

                ``"match"``     — score ≥ ``SPEAKER_ID_THRESHOLD``
                ``"fallback"``  — score < threshold but ≥ fallback floor; last
                                  known user was returned instead.
                ``"guest"``     — score below fallback floor, or no registered
                                  users with an embedding exist.
    """

    user: User
    score: float
    method: str  # "match" | "fallback" | "guest"


# ── Sentinel ──────────────────────────────────────────────────────────────────

#: Singleton guest user returned when no registered user is recognised.
#: Using a single instance keeps identity checks simple (``user is GUEST_USER``).
GUEST_USER: User = User(
    id=0,
    name="Invité",
    user_tag="user_unknown",
    preferred_address="Invité",
)
