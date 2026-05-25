# SPDX-License-Identifier: AGPL-3.0-or-later
"""
scripts/embed_memory.py
=======================
Embed any text file into an isolated review vault via Ollama.

Reads a Markdown or plain-text file, sends it to Ollama in chunks, and writes
structured notes into a dedicated review vault (separate from the main atlas_memory
vault so nothing is polluted during review).

Usage::

    python scripts/embed_memory.py path/to/file.md
    python scripts/embed_memory.py path/to/file.md --vault ./review_vault --model nomic-embed-text
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
import json


_CHUNK_SIZE = 4096   # characters per chunk
_OVERLAP = 200       # character overlap between chunks


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _OVERLAP) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


async def _embed_chunk(chunk: str, model: str, ollama_host: str) -> list[float]:
    """Get an embedding vector from Ollama for a text chunk."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{ollama_host}/api/embeddings",
            json={"model": model, "prompt": chunk},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def _run(args: argparse.Namespace) -> None:
    from atlas.config import Config, ConfigError  # noqa: PLC0415

    try:
        config = Config.from_env()
    except ConfigError:
        # For this script, we only need Ollama host
        config = None  # type: ignore[assignment]

    ollama_host = (config.ollama_host if config else "http://localhost:11434")
    model = args.model or "nomic-embed-text"
    vault = Path(args.vault) if args.vault else Path("./review_vault")
    input_path = Path(args.file)

    if not input_path.exists():
        print(f"[Error] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")
    chunks = _chunk_text(text)
    vault.mkdir(parents=True, exist_ok=True)

    print(f"Embedding {input_path.name}  ({len(chunks)} chunks, model={model!r})")

    results: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}/{len(chunks)}…", end=" ", flush=True)
        try:
            vector = await _embed_chunk(chunk, model, ollama_host)
            results.append({"chunk": i, "text": chunk[:100], "vector_dims": len(vector)})
            print("ok", flush=True)
        except Exception as exc:
            print(f"failed: {exc}", flush=True)

    # Write summary note
    out_path = vault / f"{input_path.stem}_embed_summary.md"
    out_path.write_text(
        f"---\ntype: embed_summary\nsource: {input_path.name}\nchunks: {len(chunks)}\n---\n\n"
        f"# Embedding summary: {input_path.name}\n\n"
        f"- Source: `{input_path}`\n"
        f"- Chunks: {len(chunks)}\n"
        f"- Model: `{model}`\n"
        f"- Dimensions: {results[0]['vector_dims'] if results else '?'}\n",
        encoding="utf-8",
    )
    print(f"\nDone. Summary → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a text file into a review vault via Ollama.")
    parser.add_argument("file", help="Path to the text/Markdown file to embed")
    parser.add_argument("--vault", default=None, help="Output vault directory (default: ./review_vault)")
    parser.add_argument("--model", default=None, help="Ollama embedding model (default: nomic-embed-text)")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
