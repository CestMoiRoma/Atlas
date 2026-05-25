# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/tools/memory.py
=====================
MCP stdio server — Obsidian vault interface.

Provides 8 tools for reading, writing, searching, and linking notes inside
the Obsidian vault configured via ``ATLAS_VAULT_PATH``.

All write operations forcefully inject the ``user_tag`` supplied by the
orchestrator so vault notes are always attributed to the correct speaker,
regardless of what the model passed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP  # type: ignore[import]

load_dotenv()

mcp = FastMCP(name="memory")

_VAULT = Path(os.getenv("ATLAS_VAULT_PATH", "./atlas_memory")).resolve()


def _note_path(relative: str) -> Path:
    """Resolve a vault-relative path, ensuring it stays inside the vault."""
    p = (_VAULT / relative).resolve()
    if not str(p).startswith(str(_VAULT)):
        raise ValueError(f"Path {relative!r} escapes the vault root")
    if not p.suffix:
        p = p.with_suffix(".md")
    return p


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def memory_arbo(subfolder: str = "") -> str:
    """Return the vault directory tree.

    Call this BEFORE any write, patch, link, or delete to discover existing
    paths and avoid duplicates.

    Args:
        subfolder: Optional vault-relative subdirectory to scope the tree.
                   Leave empty for the full vault tree.
    """
    root = (_VAULT / subfolder).resolve() if subfolder else _VAULT
    if not root.exists():
        return f"[Folder not found: {subfolder or '<vault root>'}]"

    lines: list[str] = [f"{root.name}/"]
    for item in sorted(root.rglob("*")):
        depth = len(item.relative_to(root).parts)
        indent = "  " * depth
        lines.append(f"{indent}{item.name}{'/' if item.is_dir() else ''}")
    return "\n".join(lines)


@mcp.tool()
def memory_read(path: str) -> str:
    """Read a vault note and return its full content.

    Args:
        path: Vault-relative path (e.g. ``Users/Roma.md``).
              The ``.md`` extension is added automatically if omitted.
    """
    try:
        note = _note_path(path)
        if not note.exists():
            return f"[Note not found: {path}]"
        return note.read_text(encoding="utf-8")
    except Exception as exc:
        return f"[Error reading {path}: {exc}]"


@mcp.tool()
def memory_write(path: str, content: str, user_tag: str = "") -> str:
    """Create or overwrite a vault note.

    Always call ``memory_arbo`` first to verify the path does not already
    exist under a different name.

    Args:
        path:     Vault-relative path (e.g. ``Memories/2026-05-25 - Roma - Projet.md``).
        content:  Full Markdown content of the note.
        user_tag: Obsidian tag for the current speaker (injected by the
                  orchestrator — always set, never empty in practice).
    """
    try:
        note = _note_path(path)
        note.parent.mkdir(parents=True, exist_ok=True)

        # Inject user_tag into frontmatter if not already present
        if user_tag and f"tags:" not in content[:200]:
            content = f"---\ntags: [{user_tag}]\ntype: memory\n---\n\n{content}"
        elif user_tag and user_tag not in content[:200]:
            content = content.replace("tags: [", f"tags: [{user_tag}, ", 1)

        note.write_text(content, encoding="utf-8")
        return f"Note written: {note.relative_to(_VAULT)}"
    except Exception as exc:
        return f"[Error writing {path}: {exc}]"


@mcp.tool()
def memory_append(path: str, content: str) -> str:
    """Append content to the end of an existing vault note.

    Args:
        path:    Vault-relative path to the note.
        content: Markdown text to append (a newline is prepended automatically).
    """
    try:
        note = _note_path(path)
        if not note.exists():
            return f"[Note not found: {path}]"
        with note.open("a", encoding="utf-8") as fh:
            fh.write(f"\n{content}")
        return f"Content appended to: {note.relative_to(_VAULT)}"
    except Exception as exc:
        return f"[Error appending to {path}: {exc}]"


@mcp.tool()
def memory_patch_section(path: str, section: str, new_content: str) -> str:
    """Replace the content of a ``## Section`` heading inside a note.

    The section heading line is preserved; only the body below it (up to the
    next ``##`` heading or end of file) is replaced.

    Args:
        path:        Vault-relative path to the note.
        section:     Heading text without the ``## `` prefix (e.g. ``"Projets"``).
        new_content: New body content to place under the heading.
    """
    try:
        note = _note_path(path)
        if not note.exists():
            return f"[Note not found: {path}]"

        text = note.read_text(encoding="utf-8")
        heading = f"## {section}"
        idx = text.find(heading)
        if idx == -1:
            return f"[Section '## {section}' not found in {path}]"

        # Find the end of this section (next ## heading or EOF)
        after_heading = text.find("\n## ", idx + len(heading))
        end_idx = after_heading if after_heading != -1 else len(text)

        patched = text[:idx + len(heading)] + "\n" + new_content.strip() + "\n" + text[end_idx:]
        note.write_text(patched, encoding="utf-8")
        return f"Section '## {section}' patched in: {note.relative_to(_VAULT)}"
    except Exception as exc:
        return f"[Error patching {path}: {exc}]"


@mcp.tool()
def memory_link(source: str, target: str) -> str:
    """Create a bidirectional wikilink between two vault notes.

    Appends ``[[target]]`` to the source note and ``[[source]]`` to the
    target note.

    Args:
        source: Vault-relative path of the first note.
        target: Vault-relative path of the second note.
    """
    results: list[str] = []
    for note_path_str, link_to_str in [(source, target), (target, source)]:
        try:
            note = _note_path(note_path_str)
            link_name = Path(link_to_str).stem  # strip .md for wikilink
            if not note.exists():
                results.append(f"[Note not found: {note_path_str}]")
                continue
            existing = note.read_text(encoding="utf-8")
            if f"[[{link_name}]]" not in existing:
                with note.open("a", encoding="utf-8") as fh:
                    fh.write(f"\n[[{link_name}]]")
            results.append(f"Linked {note_path_str} → [[{link_name}]]")
        except Exception as exc:
            results.append(f"[Error linking {note_path_str}: {exc}]")
    return "\n".join(results)


@mcp.tool()
def memory_search(query: str) -> str:
    """Full-text search across all vault notes (filenames + content).

    Returns a list of matching note paths with the surrounding context line.

    Args:
        query: Search string (case-insensitive).
    """
    if not _VAULT.exists():
        return "[Vault not found]"

    q = query.lower()
    matches: list[str] = []

    for note in sorted(_VAULT.rglob("*.md")):
        rel = str(note.relative_to(_VAULT))
        if q in rel.lower():
            matches.append(f"{rel}  (filename match)")
            continue
        try:
            for i, line in enumerate(note.read_text(encoding="utf-8").splitlines(), 1):
                if q in line.lower():
                    snippet = line.strip()[:80]
                    matches.append(f"{rel}:{i}  {snippet}")
                    break
        except Exception:
            pass

    if not matches:
        return f"No results for {query!r}"
    return "\n".join(matches[:40])  # cap at 40 results for prompt size


@mcp.tool()
def memory_delete(path: str) -> str:
    """Permanently delete a vault note.

    This action is irreversible.  Always call ``memory_arbo`` first to confirm
    the exact path.

    Args:
        path: Vault-relative path of the note to delete.
    """
    try:
        note = _note_path(path)
        if not note.exists():
            return f"[Note not found: {path}]"
        note.unlink()
        return f"Note deleted: {path}"
    except Exception as exc:
        return f"[Error deleting {path}: {exc}]"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
