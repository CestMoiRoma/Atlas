# SPDX-License-Identifier: AGPL-3.0-or-later
"""
scripts/unify_topics.py
=======================
Post-process vault Topics/ to merge duplicate or synonym topic nodes.

After a long period of use, the vault may accumulate near-duplicate topic
nodes (e.g. ``Développement.md`` and ``Dev.md``, or ``IA.md`` and
``Intelligence_Artificielle.md``).  This script:

1. Lists all ``Topics/*.md`` files.
2. Uses fuzzy string matching to find potential duplicates (similarity > threshold).
3. Presents candidates for review and merges confirmed pairs:
   - Content of the secondary note is appended to the primary.
   - All wikilinks pointing to the secondary are updated to point to the primary.
   - The secondary note is deleted.

Usage::

    python scripts/unify_topics.py
    python scripts/unify_topics.py --threshold 0.8 --auto   # auto-merge high-confidence pairs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os


def _similarity(a: str, b: str) -> float:
    """Simple character-level Jaccard similarity between two strings."""
    a_set = set(a.lower())
    b_set = set(b.lower())
    if not a_set and not b_set:
        return 1.0
    intersection = len(a_set & b_set)
    union = len(a_set | b_set)
    return intersection / union


def _find_duplicate_pairs(
    topics: list[Path], threshold: float
) -> list[tuple[Path, Path, float]]:
    """Return pairs of topic files with similarity above threshold."""
    pairs: list[tuple[Path, Path, float]] = []
    stems = [(p, p.stem) for p in topics]
    for i, (pa, sa) in enumerate(stems):
        for pb, sb in stems[i + 1:]:
            sim = _similarity(sa, sb)
            if sim >= threshold:
                pairs.append((pa, pb, sim))
    return sorted(pairs, key=lambda x: x[2], reverse=True)


def _update_wikilinks(vault: Path, old_name: str, new_name: str) -> int:
    """Replace all [[old_name]] wikilinks with [[new_name]] across the vault."""
    pattern = re.compile(rf"\[\[{re.escape(old_name)}\]\]", re.IGNORECASE)
    updated = 0
    for note in vault.rglob("*.md"):
        text = note.read_text(encoding="utf-8")
        new_text = pattern.sub(f"[[{new_name}]]", text)
        if new_text != text:
            note.write_text(new_text, encoding="utf-8")
            updated += 1
    return updated


def _merge(primary: Path, secondary: Path, vault: Path) -> None:
    """Merge secondary into primary: append content, update links, delete secondary."""
    sec_content = secondary.read_text(encoding="utf-8")
    with primary.open("a", encoding="utf-8") as fh:
        fh.write(f"\n\n---\n*Merged from [[{secondary.stem}]]*\n\n{sec_content}")

    updated = _update_wikilinks(vault, secondary.stem, primary.stem)
    secondary.unlink()
    print(f"  Merged {secondary.stem!r} → {primary.stem!r}  ({updated} links updated)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge duplicate vault Topics/.")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Similarity threshold for flagging duplicates (default: 0.75)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-merge pairs above threshold without asking")
    args = parser.parse_args()

    vault = Path(os.getenv("ATLAS_VAULT_PATH", "./atlas_memory")).resolve()
    topics_dir = vault / "Topics"

    if not topics_dir.exists():
        print(f"Topics/ directory not found at {topics_dir}")
        sys.exit(0)

    topics = sorted(topics_dir.glob("*.md"))
    if len(topics) < 2:
        print(f"Found {len(topics)} topic(s) — nothing to unify.")
        return

    pairs = _find_duplicate_pairs(topics, args.threshold)
    if not pairs:
        print(f"No duplicate topics found above threshold {args.threshold:.0%}.")
        return

    print(f"\nFound {len(pairs)} potential duplicate pair(s):\n")
    for pa, pb, sim in pairs:
        print(f"  {pa.stem!r}  ↔  {pb.stem!r}  (similarity {sim:.0%})")

    print()
    merged = 0
    for pa, pb, sim in pairs:
        if not pa.exists() or not pb.exists():
            continue  # Already merged in a previous iteration
        if args.auto:
            _merge(pa, pb, vault)
            merged += 1
        else:
            ans = input(f"  Merge {pb.stem!r} into {pa.stem!r}? [y/N] ").strip().lower()
            if ans == "y":
                _merge(pa, pb, vault)
                merged += 1

    print(f"\nDone — {merged} topic(s) merged.\n")


if __name__ == "__main__":
    main()
