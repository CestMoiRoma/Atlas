# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/wikipedia.py — MCP tools: Wikipedia search and article summary."""

from __future__ import annotations

import logging
import urllib.parse

import httpx
from mcp.server.fastmcp import FastMCP  # type: ignore[import]

logger = logging.getLogger(__name__)
mcp = FastMCP(name="wikipedia")

_API = "https://fr.wikipedia.org/api/rest_v1"
_SEARCH_API = "https://fr.wikipedia.org/w/api.php"


@mcp.tool()
def wikipedia_search(query: str) -> str:
    """Search Wikipedia (French) and return the top article titles and snippets.

    Use this first to find the correct article title, then call
    ``wikipedia_summary`` with the best result.

    Args:
        query: Search terms (e.g. ``"photosynthèse"``).

    Returns:
        Up to 5 results, each with its title and a short text snippet.
    """
    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "format": "json",
            "utf8": 1,
        }
        resp = httpx.get(_SEARCH_API, params=params, timeout=8.0)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return f"Aucun résultat Wikipedia pour : {query!r}"

        lines: list[str] = []
        for r in results:
            title = r["title"]
            snippet = r.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            lines.append(f"• {title} — {snippet[:120]}")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Wikipedia search failed: %s", exc)
        return f"[Wikipedia search unavailable: {exc}]"


@mcp.tool()
def wikipedia_summary(title: str) -> str:
    """Return the introduction section of a Wikipedia article — voice-sized.

    Args:
        title: Exact Wikipedia article title (from ``wikipedia_search`` results).

    Returns:
        First two paragraphs of the article introduction, plain text, no markup.
        Capped at ~500 characters for comfortable TTS output.
    """
    try:
        encoded = urllib.parse.quote(title.replace(" ", "_"))
        resp = httpx.get(
            f"{_API}/page/summary/{encoded}",
            headers={"Accept": "application/json"},
            timeout=8.0,
        )
        if resp.status_code == 404:
            return f"[Article non trouvé : {title!r}]"
        resp.raise_for_status()
        data = resp.json()
        extract: str = data.get("extract", "")
        if not extract:
            return f"[Aucun résumé disponible pour : {title!r}]"
        # Keep it voice-friendly: cap at ~500 chars, end at a sentence boundary
        if len(extract) > 500:
            cut = extract[:500].rfind(".")
            extract = extract[: cut + 1] if cut > 200 else extract[:500]
        return extract
    except Exception as exc:
        logger.warning("Wikipedia summary failed for %r: %s", title, exc)
        return f"[Wikipedia unavailable: {exc}]"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
