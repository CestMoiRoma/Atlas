# SPDX-License-Identifier: AGPL-3.0-or-later
"""
tests/unit/test_models.py
=========================
Unit tests for atlas.core.models — User, SpeakerMatch, GUEST_USER.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas.core.models import GUEST_USER, SpeakerMatch, User


# ---------------------------------------------------------------------------
# User dataclass
# ---------------------------------------------------------------------------


class TestUser:
    def _make_user(self, **kwargs) -> User:
        defaults = dict(
            id=1,
            name="Roma",
            user_tag="user_roma",
            age=28,
            gender="M",
            profession="Développeur",
            preferred_address="Roma",
            other_addresses=["chef", "boss"],
            embedding=None,
        )
        return User(**{**defaults, **kwargs})

    def test_basic_construction(self) -> None:
        u = self._make_user()
        assert u.name == "Roma"
        assert u.user_tag == "user_roma"

    def test_is_guest_false_for_real_user(self) -> None:
        assert not self._make_user().is_guest

    def test_is_guest_true_for_guest_user(self) -> None:
        assert GUEST_USER.is_guest

    def test_all_addresses_includes_preferred(self) -> None:
        u = self._make_user(preferred_address="chef", other_addresses=["patron"])
        addresses = u.all_addresses
        assert "chef" in addresses
        assert "patron" in addresses

    def test_all_addresses_deduplicates(self) -> None:
        u = self._make_user(preferred_address="Roma", other_addresses=["Roma", "boss"])
        assert u.all_addresses.count("Roma") == 1

    def test_all_addresses_empty_other(self) -> None:
        u = self._make_user(preferred_address="Roma", other_addresses=[])
        assert u.all_addresses == ["Roma"]

    def test_embedding_excluded_from_repr(self) -> None:
        vec = np.ones(192, dtype=np.float32)
        u = self._make_user(embedding=vec)
        r = repr(u)
        assert "embedding" not in r.lower()

    def test_none_embedding_allowed(self) -> None:
        u = self._make_user(embedding=None)
        assert u.embedding is None

    def test_numpy_embedding_stored(self) -> None:
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        u = self._make_user(embedding=vec)
        assert np.allclose(u.embedding, vec)


# ---------------------------------------------------------------------------
# GUEST_USER singleton
# ---------------------------------------------------------------------------


class TestGuestUser:
    def test_id_is_zero(self) -> None:
        assert GUEST_USER.id == 0

    def test_name_is_set(self) -> None:
        assert GUEST_USER.name  # non-empty

    def test_user_tag_unknown(self) -> None:
        assert "unknown" in GUEST_USER.user_tag.lower()

    def test_no_embedding(self) -> None:
        assert GUEST_USER.embedding is None

    def test_is_guest_property(self) -> None:
        assert GUEST_USER.is_guest is True


# ---------------------------------------------------------------------------
# SpeakerMatch
# ---------------------------------------------------------------------------


class TestSpeakerMatch:
    def _guest_match(self) -> SpeakerMatch:
        return SpeakerMatch(user=GUEST_USER, score=0.0, method="guest")

    def test_construction(self) -> None:
        m = SpeakerMatch(user=GUEST_USER, score=0.95, method="match")
        assert m.score == pytest.approx(0.95)
        assert m.method == "match"

    def test_method_match(self) -> None:
        m = SpeakerMatch(user=GUEST_USER, score=0.9, method="match")
        assert m.method == "match"

    def test_method_fallback(self) -> None:
        m = SpeakerMatch(user=GUEST_USER, score=0.4, method="fallback")
        assert m.method == "fallback"

    def test_method_guest(self) -> None:
        assert self._guest_match().method == "guest"

    def test_frozen_raises_on_mutation(self) -> None:
        m = self._guest_match()
        with pytest.raises((AttributeError, TypeError)):
            m.score = 1.0  # type: ignore[misc]

    def test_user_reference(self) -> None:
        u = User(
            id=2,
            name="Alice",
            user_tag="user_alice",
            age=None,
            gender=None,
            profession=None,
            preferred_address="Alice",
            other_addresses=[],
            embedding=None,
        )
        m = SpeakerMatch(user=u, score=0.88, method="match")
        assert m.user.name == "Alice"
