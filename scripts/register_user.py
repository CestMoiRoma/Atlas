# SPDX-License-Identifier: AGPL-3.0-or-later
"""
scripts/register_user.py
========================
Register a new Atlas user or update an existing one.

Usage — new user::

    python scripts/register_user.py --name "Roma" --age 28 --profession "Developer"

Usage — update profile fields::

    python scripts/register_user.py --name "Roma" --update --profession "Lead Dev"

Usage — re-record voice samples::

    python scripts/register_user.py --name "Roma" --update --re-record

The script guides the user through recording ``N_SAMPLES`` short voice clips
(default 5).  Each clip is encoded into a speaker embedding via SpeechBrain
ECAPA-TDNN and the average is stored in the database.

Recorded WAV files are saved to ``VOICE_TEMPLATES_DIR / name /`` for future
re-averaging during sleeping mode.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import re
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from atlas.config import Config, ConfigError
from atlas.core.models import User
from atlas.core.speaker_id import warm_up
from atlas.db.user_db import get_user_by_name, init_db, upsert_user, update_embedding

N_SAMPLES = 5          # Voice clips to record
CLIP_DURATION = 4.0    # Seconds per clip
SAMPLE_RATE = 16000


def _slugify(name: str) -> str:
    """Convert a display name to a safe Obsidian tag (lowercase, underscores)."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())


def _record_clip(n: int, total: int) -> np.ndarray:
    """Record a single voice clip from the default microphone."""
    print(f"\n  Clip {n}/{total} — speak naturally for {CLIP_DURATION:.0f} s "
          f"(press Enter when ready)", end="", flush=True)
    input()
    print("  🎙  Recording…", end=" ", flush=True)
    audio = sd.rec(
        int(CLIP_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("done.", flush=True)
    return audio[:, 0]


async def _build_embedding(clips: list[np.ndarray], config: Config) -> np.ndarray:
    """Average SpeechBrain embeddings from a list of voice clips."""
    import torch  # type: ignore[import]
    from atlas.core.speaker_id import _embed  # noqa: PLC0415

    await warm_up(config)

    embeddings: list[np.ndarray] = []
    for clip in clips:
        emb = await asyncio.get_running_loop().run_in_executor(
            None, _embed, clip, SAMPLE_RATE
        )
        embeddings.append(emb)

    averaged = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(averaged)
    return (averaged / norm) if norm > 0 else averaged


def _save_clips(clips: list[np.ndarray], name: str, config: Config) -> None:
    """Persist recorded WAV files for future re-averaging."""
    user_dir = config.voice_templates_dir / name
    user_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        wav_path = user_dir / f"{int(time.time() * 1000)}.wav"
        sf.write(str(wav_path), clip, SAMPLE_RATE, subtype="PCM_16")
        time.sleep(0.01)  # Ensure unique filenames
    print(f"  Voice samples saved → {user_dir}", flush=True)


async def _run(args: argparse.Namespace) -> None:
    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"\n[Error] Configuration: {exc}\n", file=sys.stderr)
        sys.exit(1)

    db = init_db(config.speaker_db_path)
    name: str = args.name.strip()
    user_tag = _slugify(name)

    existing = get_user_by_name(db, name)

    if existing and not args.update:
        print(f"\n[Error] User {name!r} already exists. Use --update to modify.", file=sys.stderr)
        sys.exit(1)

    if not existing and args.update:
        print(f"\n[Error] User {name!r} not found. Remove --update to create a new user.", file=sys.stderr)
        sys.exit(1)

    # ── Profile ──────────────────────────────────────────────────────────────
    age = args.age or (existing.age if existing else None)
    gender = args.gender or (existing.gender if existing else None)
    profession = args.profession or (existing.profession if existing else None)
    preferred = args.preferred_address or (existing.preferred_address if existing else name)

    user_id = upsert_user(
        db,
        name=name,
        user_tag=user_tag,
        age=age,
        gender=gender,
        profession=profession,
        preferred_address=preferred,
    )

    print(f"\n  {'Updated' if args.update else 'Created'} user: {name!r}  (id={user_id}  tag={user_tag!r})")

    # ── Voice recording ───────────────────────────────────────────────────────
    do_record = (not args.update) or args.re_record
    if not do_record:
        print("  Profile updated — voice samples unchanged.")
        return

    print(f"\n  Recording {N_SAMPLES} voice clips. Speak in your natural voice, "
          f"at normal distance from the mic.")
    clips: list[np.ndarray] = []
    for i in range(1, N_SAMPLES + 1):
        clips.append(_record_clip(i, N_SAMPLES))

    print("\n  Computing speaker embedding…", flush=True)
    embedding = await _build_embedding(clips, config)

    update_embedding(db, user_id, embedding)
    _save_clips(clips, name, config)

    print(f"\n  \033[32m✓\033[0m  {name!r} registered successfully.")
    print(f"     Embedding shape : {embedding.shape}")
    print(f"     DB path         : {config.speaker_db_path}")
    print(f"\n  You can now start Atlas:")
    print(f"     python -m atlas.core.orchestrator\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register or update an Atlas user.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="Display name (e.g. 'Roma')")
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--gender", default=None)
    parser.add_argument("--profession", default=None)
    parser.add_argument("--preferred-address", dest="preferred_address", default=None,
                        help="Primary nickname Atlas uses in speech (default: --name)")
    parser.add_argument("--update", action="store_true",
                        help="Update an existing user's profile fields")
    parser.add_argument("--re-record", action="store_true",
                        help="Re-record voice samples for an existing user")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
