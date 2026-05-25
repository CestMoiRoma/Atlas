# SPDX-License-Identifier: AGPL-3.0-or-later
"""
scripts/embed_deep.py
=====================
Large-context embedding variant — 100K-token sliding window with 50% overlap.

Designed for large vaults or long documents where ``embed_memory.py``'s simple
chunking would lose too much cross-chunk context.  Sends each window to Ollama
for summarisation + embedding, then writes one note per window into the vault.

Usage::

    python scripts/embed_deep.py path/to/large_file.md
    python scripts/embed_deep.py path/to/large_file.md --window 50000 --overlap 0.4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import httpx

_DEFAULT_WINDOW = 100_000   # characters (~25K tokens at 4 chars/token)
_DEFAULT_OVERLAP = 0.5


def _sliding_windows(text: str, window: int, overlap: float) -> list[tuple[int, str]]:
    """Yield (index, chunk) with a sliding window and fractional overlap."""
    step = int(window * (1.0 - overlap))
    windows: list[tuple[int, str]] = []
    start = 0
    idx = 0
    while start < len(text):
        windows.append((idx, text[start:start + window]))
        start += step
        idx += 1
    return windows


async def _summarise_window(chunk: str, model: str, ollama_host: str) -> str:
    """Ask Ollama to produce a concise summary of a text window."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": (
                    "Summarise the following text in 3–5 bullet points, "
                    "capturing the key facts and entities:\n\n" + chunk[:_DEFAULT_WINDOW]
                ),
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


async def _run(args: argparse.Namespace) -> None:
    from atlas.config import Config, ConfigError  # noqa: PLC0415

    try:
        config = Config.from_env()
        ollama_host = config.ollama_host
        model = args.model or config.ollama_model
    except ConfigError:
        ollama_host = "http://localhost:11434"
        model = args.model or "llama3.2"

    vault = Path(args.vault) if args.vault else Path("./review_vault")
    input_path = Path(args.file)

    if not input_path.exists():
        print(f"[Error] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")
    window = args.window
    overlap = args.overlap
    windows = _sliding_windows(text, window, overlap)

    vault.mkdir(parents=True, exist_ok=True)
    print(f"Deep embedding {input_path.name}  ({len(windows)} windows, model={model!r})")

    for idx, chunk in windows:
        print(f"  Window {idx + 1}/{len(windows)} ({len(chunk):,} chars)…", end=" ", flush=True)
        try:
            summary = await _summarise_window(chunk, model, ollama_host)
            note_path = vault / f"{input_path.stem}_window_{idx:04d}.md"
            note_path.write_text(
                f"---\ntype: embed_window\nsource: {input_path.name}\n"
                f"window: {idx}\nchars: {len(chunk)}\n---\n\n"
                f"# Window {idx} — {input_path.stem}\n\n{summary}\n",
                encoding="utf-8",
            )
            print("ok", flush=True)
        except Exception as exc:
            print(f"failed: {exc}", flush=True)

    print(f"\nDone. {len(windows)} notes written → {vault}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Large-context sliding-window embedding for Atlas vault.")
    parser.add_argument("file", help="Path to the file to embed")
    parser.add_argument("--vault", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--window", type=int, default=_DEFAULT_WINDOW,
                        help=f"Window size in characters (default: {_DEFAULT_WINDOW})")
    parser.add_argument("--overlap", type=float, default=_DEFAULT_OVERLAP,
                        help=f"Overlap fraction 0.0–1.0 (default: {_DEFAULT_OVERLAP})")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
