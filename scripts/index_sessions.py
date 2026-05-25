# SPDX-License-Identifier: AGPL-3.0-or-later
"""
scripts/index_sessions.py
=========================
One-time migration — wire existing session files into the Sessions.md hub.

If you have session files in ``atlas_memory/Sessions/`` that were created
before the hub-wiring logic existed (or that were not closed properly), this
script scans them and adds their wikilinks to ``Sessions.md``.

Usage::

    python scripts/index_sessions.py
    python scripts/index_sessions.py --vault /path/to/atlas_memory
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index existing session files into Sessions.md hub.")
    parser.add_argument("--vault", default=None,
                        help="Vault path (default: ATLAS_VAULT_PATH from .env)")
    args = parser.parse_args()

    vault = Path(args.vault or os.getenv("ATLAS_VAULT_PATH", "./atlas_memory")).resolve()
    sessions_dir = vault / "Sessions"
    hub_path = vault / "Sessions.md"

    if not sessions_dir.exists():
        print(f"Sessions/ directory not found at {sessions_dir}")
        sys.exit(0)

    session_files = sorted(sessions_dir.glob("*.md"))
    if not session_files:
        print("No session files found.")
        return

    # Collect already-indexed stems from Sessions.md
    already: set[str] = set()
    if hub_path.exists():
        content = hub_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- [[Sessions/"):
                stem = line.removeprefix("- [[Sessions/").removesuffix("]]")
                already.add(stem)
    else:
        hub_path.parent.mkdir(parents=True, exist_ok=True)
        hub_path.write_text(
            "---\ntype: sessions_index\n---\n\n"
            "# Sessions\n\nIndex de toutes les conversations d'Atlas.\n\n## Journal\n",
            encoding="utf-8",
        )

    new_links: list[str] = []
    for sf in session_files:
        if sf.stem not in already:
            new_links.append(f"- [[Sessions/{sf.stem}]]")

    if not new_links:
        print(f"All {len(session_files)} session(s) already indexed.")
        return

    with hub_path.open("a", encoding="utf-8") as fh:
        for link in new_links:
            fh.write(link + "\n")

    print(f"Indexed {len(new_links)} new session(s) → {hub_path}")


if __name__ == "__main__":
    main()
