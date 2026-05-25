# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/tools/inbox.py
====================
MCP tools — document inbox reader.

The inbox is a directory where the user drops files for Atlas to read and
process.  Files are *never deleted* — Atlas can re-read them at any time.

Supported formats
-----------------
* ``.txt``, ``.md`` — plain text / Markdown
* ``.json`` — pretty-printed JSON
* ``.csv``, ``.tsv`` — formatted table (first 50 rows)
* ``.drawio``, ``.xml`` — draw.io diagrams — label extraction from shapes
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP  # type: ignore[import]

load_dotenv()
logger = logging.getLogger(__name__)
mcp = FastMCP(name="inbox")

_INBOX = Path(os.getenv("ATLAS_INBOX_PATH", "./atlas_inbox")).resolve()
_SUPPORTED = {".txt", ".md", ".json", ".csv", ".tsv", ".drawio", ".xml"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_csv(path: Path, delimiter: str = ",") -> str:
    """Format a CSV/TSV as a readable table (max 50 rows)."""
    rows: list[list[str]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= 51:
                rows.append(["… (truncated)"])
                break
            rows.append(row)
    if not rows:
        return "(empty)"
    # Column widths
    col_count = max(len(r) for r in rows)
    widths = [0] * col_count
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines: list[str] = []
    for row in rows:
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append(" | ".join(padded))
    return "\n".join(lines)


def _read_drawio(path: Path) -> str:
    """Extract visible text labels from a draw.io diagram XML."""
    try:
        import zlib, base64  # noqa: PLC0415, E401
        raw = path.read_text(encoding="utf-8")
        # draw.io files may be compressed (mxDiagram value attribute)
        root = ET.fromstring(raw)
        # Try to decompress if the diagram is stored compressed
        diagram_el = root.find(".//diagram")
        if diagram_el is not None and diagram_el.text:
            try:
                compressed = base64.b64decode(diagram_el.text)
                decompressed = zlib.decompress(compressed, -15).decode("utf-8")
                root = ET.fromstring(decompressed)
            except Exception:
                pass  # Already uncompressed XML

        labels: list[str] = []
        for el in root.iter():
            label = el.get("label") or el.get("value") or ""
            label = label.strip()
            if label and label not in labels:
                labels.append(label)
        return "draw.io labels:\n" + "\n".join(f"  • {l}" for l in labels) if labels else "(no labels found)"
    except Exception as exc:
        return f"[draw.io parse error: {exc}]"


def _read_file(path: Path) -> str:
    """Read and format a single inbox file."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(path.read_bytes())
        return json.dumps(data, ensure_ascii=False, indent=2)
    if suffix == ".csv":
        return _read_csv(path, ",")
    if suffix == ".tsv":
        return _read_csv(path, "\t")
    if suffix in (".drawio", ".xml"):
        return _read_drawio(path)
    return f"[Unsupported format: {suffix}]"


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def inbox_list() -> str:
    """List all files currently in the document inbox.

    Returns filenames with their sizes.  Call this before ``inbox_read`` to
    see what files are available.
    """
    if not _INBOX.exists():
        return "(inbox directory does not exist — no files)"

    files = [
        f for f in sorted(_INBOX.iterdir())
        if f.is_file() and f.suffix.lower() in _SUPPORTED
    ]
    if not files:
        return "(inbox is empty)"

    lines: list[str] = []
    for f in files:
        size_kb = f.stat().st_size / 1024
        lines.append(f"  {f.name}  ({size_kb:.1f} KB)")
    return f"Inbox ({len(files)} file{'s' if len(files) != 1 else ''}):\n" + "\n".join(lines)


@mcp.tool()
def inbox_read(filename: str) -> str:
    """Read and return the content of a file from the inbox.

    Files are never deleted — call this whenever you want to re-read a file.

    Args:
        filename: Exact filename as listed by ``inbox_list``
                  (e.g. ``"notes.md"``).

    Returns:
        File content as a string.  JSON is pretty-printed; CSV/TSV is
        formatted as a table; draw.io diagrams return extracted labels.
    """
    path = _INBOX / filename
    if not path.exists():
        return f"[File not found in inbox: {filename!r}]"
    if not path.is_file():
        return f"[Not a file: {filename!r}]"
    if path.suffix.lower() not in _SUPPORTED:
        return f"[Unsupported format: {path.suffix}]"

    try:
        return _read_file(path)
    except Exception as exc:
        logger.warning("inbox_read failed for %r: %s", filename, exc)
        return f"[Error reading {filename!r}: {exc}]"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
