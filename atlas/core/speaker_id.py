# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/speaker_id.py
========================
Speaker identification using SpeechBrain ECAPA-TDNN embeddings.

How it works
------------
1. The SpeechBrain ``EncoderClassifier`` (ECAPA-TDNN architecture, trained on
   VoxCeleb) encodes a raw audio waveform into a compact speaker embedding
   vector (~192 dimensions).

2. Each registered user has one reference embedding stored in the SQLite
   database (the average of all their recorded voice samples).

3. At every turn, the new audio is encoded and its cosine similarity is computed
   against every registered user's embedding.  The user with the highest score
   is returned if the score exceeds ``config.speaker_id_threshold``.

4. **Soft fallback** — if the best score is below the threshold but above
   ``config.speaker_fallback_min_score``, the last confirmed real user is
   returned instead.  This handles mic distance and background noise without
   treating the user as a guest on every noisy utterance.

5. **Self-improving** — during sleeping mode, ``recompute_all_embeddings()``
   re-averages each user's embedding from all WAV samples saved in their
   ``voice_templates/`` directory.  The more Atlas is used, the more robust
   the embeddings become.

Voice sample persistence
------------------------
Every utterance that scores ≥ threshold is saved as a WAV file in
``config.voice_templates_dir / user.name /``.  The directory is capped at
``config.max_samples_per_user`` files (oldest deleted first).  Re-averaging
is triggered every ``config.embed_update_every`` new saves.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

import numpy as np
import soundfile as sf

from atlas.config import Config
from atlas.core.models import GUEST_USER, SpeakerMatch, User

logger = logging.getLogger(__name__)

# Module-level cached model — loaded once on first use via warm_up()
_encoder: object | None = None
_encoder_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _encoder_lock
    if _encoder_lock is None:
        _encoder_lock = asyncio.Lock()
    return _encoder_lock


async def warm_up(config: Config) -> None:
    """Pre-load the SpeechBrain model so the first turn has no load latency.

    Safe to call multiple times — the model is only loaded once.

    Args:
        config: Atlas runtime configuration.  Uses ``config.speaker_model``
                and ``config.speaker_savedir``.
    """
    global _encoder
    async with _get_lock():
        if _encoder is not None:
            return
        logger.info("Loading SpeechBrain ECAPA-TDNN from %s …", config.speaker_savedir)
        loop = asyncio.get_running_loop()
        _encoder = await loop.run_in_executor(
            None, _load_encoder, config.speaker_model, str(config.speaker_savedir)
        )
        logger.info("SpeechBrain model ready.")


def _load_encoder(source: str, savedir: str) -> object:
    """Blocking model load — called from a thread pool."""
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore[import]
    return EncoderClassifier.from_hparams(
        source=source,
        savedir=savedir,
        run_opts={"device": "cpu"},
    )


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Encode *audio* into a normalised speaker embedding.

    Args:
        audio:       Float32 mono waveform.
        sample_rate: Sample rate of *audio* in Hz.

    Returns:
        1-D float32 numpy array (normalised L2 unit vector).
    """
    import torch  # type: ignore[import]

    if _encoder is None:
        raise RuntimeError("SpeechBrain model not loaded — call warm_up() first")

    tensor = torch.tensor(audio).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        emb = _encoder.encode_batch(tensor)    # (1, 1, D)

    vec: np.ndarray = emb.squeeze().cpu().numpy()
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit vectors (fast path: just dot product).

    Returns 0.0 when the two vectors have different shapes (e.g. a stored
    embedding was produced by a different model version) so the caller can
    treat the user as a non-match rather than crashing.
    """
    if a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))


# ── Identification ────────────────────────────────────────────────────────────

class SpeakerIdentifier:
    """Identifies the speaker from a raw audio waveform.

    Args:
        config: Atlas runtime configuration.
        db_conn: Open SQLite connection from ``atlas.db.user_db.init_db()``.
    """

    def __init__(self, config: Config, db_conn: sqlite3.Connection) -> None:
        self._cfg = config
        self._db = db_conn
        self._sample_counts: dict[int, int] = {}  # user_id → unsaved samples since last re-avg

    async def identify(
        self,
        audio: np.ndarray,
        fallback_user: User | None = None,
    ) -> SpeakerMatch:
        """Identify the speaker in *audio* against all registered users.

        Args:
            audio:         Float32 mono waveform at ``config.audio_sample_rate`` Hz.
            fallback_user: Last known real user.  Used when the score is below
                           threshold but above the fallback floor.

        Returns:
            A :class:`~atlas.core.models.SpeakerMatch` with the best-matching
            user (or the guest sentinel), the cosine similarity score, and the
            match method string.
        """
        from atlas.db.user_db import get_all_users, update_embedding  # noqa: PLC0415

        if audio.size == 0:
            return SpeakerMatch(user=GUEST_USER, score=0.0, method="guest")

        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None, _embed, audio, self._cfg.audio_sample_rate
        )

        users = get_all_users(self._db)
        candidates = [u for u in users if u.embedding is not None]

        if not candidates:
            logger.debug("No registered users with embeddings — returning guest")
            return SpeakerMatch(user=GUEST_USER, score=0.0, method="guest")

        scores = [(u, _cosine(embedding, u.embedding)) for u in candidates]  # type: ignore[arg-type]
        best_user, best_score = max(scores, key=lambda x: x[1])

        logger.info(
            "Speaker scores: %s",
            "  ".join(f"{u.name}={s:.3f}" for u, s in scores),
        )

        if best_score >= self._cfg.speaker_id_threshold:
            # Confirmed match — save sample and possibly re-average
            await loop.run_in_executor(
                None, self._save_sample, best_user, audio
            )
            self._sample_counts[best_user.id] = self._sample_counts.get(best_user.id, 0) + 1
            if self._sample_counts[best_user.id] >= self._cfg.embed_update_every:
                new_emb = await loop.run_in_executor(
                    None, self._recompute_embedding, best_user
                )
                if new_emb is not None:
                    update_embedding(self._db, best_user.id, new_emb)
                self._sample_counts[best_user.id] = 0

            return SpeakerMatch(user=best_user, score=best_score, method="match")

        if best_score >= self._cfg.speaker_fallback_min_score and fallback_user is not None:
            logger.info(
                "Soft fallback → %r (score %.3f < threshold %.3f)",
                fallback_user.name, best_score, self._cfg.speaker_id_threshold,
            )
            return SpeakerMatch(user=fallback_user, score=best_score, method="fallback")

        logger.info("No match (best score %.3f) — returning guest", best_score)
        return SpeakerMatch(user=GUEST_USER, score=best_score, method="guest")

    # ── Sample persistence ────────────────────────────────────────────────────

    def _save_sample(self, user: User, audio: np.ndarray) -> None:
        """Save *audio* as a WAV file in the user's voice template directory."""
        cfg = self._cfg
        user_dir = cfg.voice_templates_dir / user.name
        user_dir.mkdir(parents=True, exist_ok=True)

        # Prune oldest files if over cap
        existing = sorted(user_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        while len(existing) >= cfg.max_samples_per_user:
            existing.pop(0).unlink()

        import time  # noqa: PLC0415
        wav_path = user_dir / f"{int(time.time() * 1000)}.wav"
        sf.write(str(wav_path), audio, cfg.audio_sample_rate, subtype="PCM_16")
        logger.debug("Voice sample saved → %s", wav_path)

    def _recompute_embedding(self, user: User) -> np.ndarray | None:
        """Average all saved WAV samples for *user* into a new embedding."""
        user_dir = self._cfg.voice_templates_dir / user.name
        wavs = list(user_dir.glob("*.wav"))
        if not wavs:
            return None

        embeddings: list[np.ndarray] = []
        for wav_path in wavs:
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            try:
                emb = _embed(audio, sr)
                embeddings.append(emb)
            except Exception as exc:
                logger.warning("Failed to embed %s: %s", wav_path, exc)

        if not embeddings:
            return None

        averaged = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(averaged)
        result = (averaged / norm) if norm > 0 else averaged
        logger.info("Embedding re-averaged for %r from %d samples", user.name, len(embeddings))
        return result


# ── Sleeping mode re-average ──────────────────────────────────────────────────

async def recompute_all_embeddings(
    config: Config,
    db_conn: sqlite3.Connection,
) -> None:
    """Re-average voice embeddings for all registered users from saved WAV files.

    Called by the sleeping mode monitor after a period of inactivity.  Updates
    the database in place — subsequent identifications immediately benefit from
    the improved embeddings.

    Args:
        config:  Atlas runtime configuration.
        db_conn: Open SQLite connection.
    """
    from atlas.db.user_db import get_all_users, update_embedding  # noqa: PLC0415

    identifier = SpeakerIdentifier(config, db_conn)
    users = get_all_users(db_conn)
    loop = asyncio.get_running_loop()

    for user in users:
        try:
            new_emb = await loop.run_in_executor(
                None, identifier._recompute_embedding, user
            )
            if new_emb is not None:
                update_embedding(db_conn, user.id, new_emb)
        except Exception as exc:
            logger.warning("Re-average failed for %r: %s", user.name, exc)
