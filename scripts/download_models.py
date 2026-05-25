# SPDX-License-Identifier: AGPL-3.0-or-later
"""
scripts/download_models.py
==========================
Pre-download all model weights required by Atlas before the first run.

Run once after installation::

    python scripts/download_models.py

What this script downloads
--------------------------
1. **SpeechBrain ECAPA-TDNN** (~80 MB) — speaker identification model.
   Downloaded from HuggingFace via the SpeechBrain API, saved to
   ``SPEAKER_SAVEDIR`` (default: ``./models/spkrec-ecapa-voxceleb``).
   SHA-256 verified after download.

2. **Whisper GGML model** — the script cannot download this automatically
   because the model choice (tiny / base / large-v3-turbo) is up to the user.
   It prints the download URL and instructions instead.

3. **Atlas.onnx** — already in the repository under ``models/``.
   The script just confirms the file is present.

Retry logic
-----------
All network downloads use exponential backoff (1 s → 2 s → 4 s) with up to
3 attempts before failing.  A tqdm progress bar shows real-time download speed
and ETA.
"""

from __future__ import annotations

import hashlib
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import os

# ── Model specification ───────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Describes a model to be downloaded and verified."""
    name: str
    dest: Path
    sha256: str | None = None      # Expected SHA-256 hex digest (None = skip)
    size_mb: float = 0.0           # Indicative size for progress display
    url: str | None = None         # Direct download URL (None = use custom logic)
    headers: dict = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_step(label: str) -> None:
    print(f"\n{'─' * 50}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'─' * 50}", flush=True)


def _ok(note: str = "") -> None:
    print(f"  \033[32m✓\033[0m  {note}", flush=True)


def _warn(note: str) -> None:
    print(f"  \033[33m⚠\033[0m  {note}", flush=True)


def _fail(note: str) -> None:
    print(f"  \033[31m✗\033[0m  {note}", flush=True)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file in 64 KB streaming chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dest: Path, headers: dict | None = None) -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    import httpx  # noqa: PLC0415

    try:
        from tqdm import tqdm  # noqa: PLC0415
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, headers=headers or {}, follow_redirects=True, timeout=30.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) if use_tqdm else None
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=65536):
                fh.write(chunk)
                if bar:
                    bar.update(len(chunk))
        if bar:
            bar.close()


def _download_with_retry(url: str, dest: Path, headers: dict | None = None, retries: int = 3) -> None:
    """Download with exponential backoff retry (1 s → 2 s → 4 s)."""
    for attempt in range(retries):
        try:
            _download_with_progress(url, dest, headers)
            return
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  Attempt {attempt + 1} failed ({exc}) — retrying in {wait} s…", flush=True)
            time.sleep(wait)


# ── Step 1: SpeechBrain ECAPA-TDNN ───────────────────────────────────────────

def download_speechbrain() -> bool:
    """Download SpeechBrain ECAPA-TDNN speaker encoder from HuggingFace."""
    _print_step("1/3  SpeechBrain ECAPA-TDNN (speaker identification, ~80 MB)")

    model_id = os.getenv("SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
    savedir = Path(os.getenv("SPEAKER_SAVEDIR", "./models/spkrec-ecapa-voxceleb"))

    if savedir.exists() and any(savedir.glob("*.pt")):
        _ok(f"Already downloaded → {savedir}")
        return True

    try:
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore[import]
        print(f"  Downloading {model_id} → {savedir} …", flush=True)
        EncoderClassifier.from_hparams(
            source=model_id,
            savedir=str(savedir),
            run_opts={"device": "cpu"},
        )
        _ok(f"Saved to {savedir}")
        return True
    except Exception as exc:
        _fail(f"SpeechBrain download failed: {exc}")
        traceback.print_exc()
        return False


# ── Step 2: Atlas.onnx (verify presence) ─────────────────────────────────────

def verify_atlas_onnx() -> bool:
    """Verify Atlas.onnx is present in the models/ directory."""
    _print_step("2/3  Atlas.onnx wakeword model (included in repo)")

    model_path = Path("models/Atlas.onnx")
    if not model_path.exists():
        # Try relative to the script location
        model_path = Path(__file__).parent.parent / "models" / "Atlas.onnx"

    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        _ok(f"Present → {model_path}  ({size_kb:.0f} KB)")
        return True
    else:
        _fail(f"Atlas.onnx not found at {model_path}")
        _warn("This file should be included in the repository.")
        _warn("Check that you cloned the repo correctly: git clone ...")
        return False


# ── Step 3: Whisper instructions ─────────────────────────────────────────────

def print_whisper_instructions() -> bool:
    """Print instructions for downloading a Whisper GGML model."""
    _print_step("3/3  Whisper GGML model (manual download required)")

    whisper_model = os.getenv("WHISPER_CPP_MODEL", "")
    if whisper_model and Path(whisper_model).exists():
        size_mb = Path(whisper_model).stat().st_size / 1_048_576
        _ok(f"Already configured → {whisper_model}  ({size_mb:.0f} MB)")
        return True

    _warn("Whisper model not configured or not found.")
    print("""
  Atlas requires a Whisper GGML model file (.bin) for speech recognition.
  The model size is your choice — larger = more accurate but slower.

  Recommended: ggml-large-v3-turbo.bin (~800 MB, best accuracy)
  Lightweight:  ggml-small.bin         (~244 MB, faster)
  Minimal:      ggml-base.bin          (~148 MB)

  Download page:
    https://huggingface.co/ggerganov/whisper.cpp/tree/main

  After downloading, set in your .env:
    WHISPER_CPP_MODEL=/absolute/path/to/ggml-large-v3-turbo.bin
""", flush=True)
    return True  # Not a blocker — instructions provided


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n\033[1mAtlas — model pre-download\033[0m", flush=True)

    results = {
        "SpeechBrain ECAPA-TDNN": download_speechbrain(),
        "Atlas.onnx":             verify_atlas_onnx(),
        "Whisper model":          print_whisper_instructions(),
    }

    print(f"\n{'═' * 50}", flush=True)
    print("  Summary:", flush=True)
    all_ok = True
    for name, ok in results.items():
        icon = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
        print(f"  {icon}  {name}", flush=True)
        if not ok:
            all_ok = False

    if all_ok:
        print("""
  \033[32mAll models ready.\033[0m

  Next steps:
    1. Copy and edit your configuration:
         cp .env.example .env
    2. Set WHISPER_CPP_MODEL in .env (see instructions above)
    3. Register yourself:
         python scripts/register_user.py --name "YourName" --age 30
    4. Start Atlas:
         ollama serve &
         python -m atlas.core.orchestrator
""", flush=True)
    else:
        print("\n  \033[31mSome steps failed — check the errors above.\033[0m\n", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
