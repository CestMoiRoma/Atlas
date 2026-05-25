# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/mcp_client.py
========================
MCP (Model Context Protocol) stdio client for Atlas tool servers.

Architecture
------------
Each tool server is a standalone Python module under ``atlas/tools/`` that
implements a FastMCP stdio server.  The client spawns each server as a
subprocess on demand, exchanges JSON-RPC messages over stdin/stdout, and tears
the process down after the call.

Tool naming convention
----------------------
Tools are addressed with a qualified name ``<server>::<tool>``, e.g.
``memory::memory_write``.  When the LLM emits a tool call via Ollama the name
uses the double-underscore form ``<server>__<tool>`` — the client converts
between the two transparently.

Prerequisite ordering
---------------------
Certain vault-mutation tools require a prior ``memory_arbo`` call in the same
speech cycle to avoid writing to wrong paths or creating duplicates.
``TOOL_PREREQUISITES`` maps each such tool to its required predecessors.
The orchestrator enforces this ordering; the client itself does not — it is a
pure dispatcher.

Per-tool timeout
----------------
Every ``call_tool()`` dispatch is wrapped in ``asyncio.wait_for`` with
``config.mcp_tool_timeout`` (default 10 s).  Hanging network-dependent tools
(weather, Wikipedia) return a graceful error string instead of freezing the
pipeline.

Parallel dispatch
-----------------
``call_tools_parallel()`` accepts a list of independent tool calls and fires
them concurrently with ``asyncio.gather``.  Tools with prerequisites must be
dispatched sequentially by the orchestrator before calling this function.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from atlas.config import Config

logger = logging.getLogger(__name__)

# ── Server registry ───────────────────────────────────────────────────────────

#: Maps server name → Python module path used to spawn the stdio subprocess.
TOOL_SERVERS: dict[str, str] = {
    "memory":        "atlas.tools.memory",
    "datetime_info": "atlas.tools.datetime_info",
    "geoposition":   "atlas.tools.geoposition",
    "weather":       "atlas.tools.weather",
    "metrics":       "atlas.tools.metrics",
    "wikipedia":     "atlas.tools.wikipedia",
    "inbox":         "atlas.tools.inbox",
}

#: Tools that require specific predecessors to have been called first in the
#: current speech cycle.  Key = tool qualified name, value = list of required
#: predecessors (also qualified names).
TOOL_PREREQUISITES: dict[str, list[str]] = {
    "memory__memory_write":         ["memory__memory_arbo"],
    "memory__memory_patch_section": ["memory__memory_arbo"],
    "memory__memory_link":          ["memory__memory_arbo"],
    "memory__memory_delete":        ["memory__memory_arbo"],
    "memory__memory_append":        ["memory__memory_arbo"],
}


# ── Low-level MCP dispatch ────────────────────────────────────────────────────

async def _spawn_and_call(
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    python_bin: str,
) -> str:
    """Spawn a tool server subprocess, call one tool, and return the result.

    The server is started fresh for each call (stateless).  This is slightly
    slower than keeping servers alive (see planned persistent-process feature)
    but is simpler, more fault-tolerant, and sufficient for the current use.

    Args:
        server_name: Key from ``TOOL_SERVERS`` (e.g. ``"memory"``).
        tool_name:   Tool function name within the server (e.g. ``"memory_read"``).
        arguments:   Dict of arguments to pass to the tool.
        python_bin:  Python interpreter to use (from ``config.mcp_python``).

    Returns:
        String result from the tool, or an error message string.
    """
    module = TOOL_SERVERS.get(server_name)
    if module is None:
        return f"[Error: unknown tool server {server_name!r}]"

    from mcp import ClientSession, StdioServerParameters  # type: ignore[import]
    from mcp.client.stdio import stdio_client            # type: ignore[import]

    server_params = StdioServerParameters(
        command=python_bin,
        args=["-m", module],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                # Extract text content from the MCP result
                content = result.content
                if isinstance(content, list):
                    parts = [
                        c.text if hasattr(c, "text") else str(c)
                        for c in content
                    ]
                    return "\n".join(parts)
                return str(content)
    except Exception as exc:
        logger.error("MCP call %s::%s failed: %s", server_name, tool_name, exc)
        return f"[Error calling {server_name}::{tool_name}: {exc}]"


# ── Public API ────────────────────────────────────────────────────────────────

class MCPClient:
    """Atlas MCP client — dispatches tool calls to stdio server subprocesses.

    Args:
        config: Atlas runtime configuration.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config

    # ── Schema discovery ──────────────────────────────────────────────────────

    async def list_tool_schemas(self, server_name: str) -> list[dict[str, Any]]:
        """Return Ollama-compatible tool descriptors for all tools in *server_name*.

        Each descriptor has the form::

            {
                "type": "function",
                "function": {
                    "name": "<tool_name>",
                    "description": "...",
                    "parameters": { ... }   # JSON Schema
                }
            }

        The ``name`` field is prefixed with ``<server_name>__`` by the caller
        (orchestrator) so Ollama can route tool calls back to the correct server.

        Returns an empty list if the server cannot be reached.
        """
        from mcp import ClientSession, StdioServerParameters  # type: ignore[import]
        from mcp.client.stdio import stdio_client            # type: ignore[import]

        module = TOOL_SERVERS.get(server_name)
        if module is None:
            logger.warning("Unknown server %r — skipping schema discovery", server_name)
            return []

        server_params = StdioServerParameters(
            command=self._cfg.mcp_python,
            args=["-m", module],
        )

        schemas: list[dict[str, Any]] = []
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    for tool in tools_result.tools:
                        schemas.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                            },
                        })
        except Exception as exc:
            logger.warning("Schema discovery failed for %r: %s", server_name, exc)

        return schemas

    async def discover_all_schemas(self) -> list[dict[str, Any]]:
        """Query every registered server and return all Ollama tool descriptors.

        Tool names are prefixed with ``<server>__`` so the orchestrator can
        route calls back to the correct server.
        """
        all_schemas: list[dict[str, Any]] = []
        for server_name in TOOL_SERVERS:
            try:
                schemas = await self.list_tool_schemas(server_name)
                for schema in schemas:
                    schema["function"]["name"] = f"{server_name}__{schema['function']['name']}"
                all_schemas.extend(schemas)
                logger.info("Discovered %d tool(s) from %r", len(schemas), server_name)
            except Exception as exc:
                logger.warning("Could not discover tools from %r: %s", server_name, exc)
        return all_schemas

    # ── Single tool call ──────────────────────────────────────────────────────

    async def call_tool(
        self,
        qualified_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Dispatch a single tool call with per-tool timeout.

        Args:
            qualified_name: ``<server>__<tool>`` (Ollama convention) or
                            ``<server>::<tool>`` (internal convention).
            arguments:      Tool arguments as a dict.

        Returns:
            String result from the tool, or a timeout/error message string.
        """
        # Normalise separator
        sep = "__" if "__" in qualified_name else "::"
        if sep not in qualified_name:
            return f"[Error: malformed tool name {qualified_name!r}]"

        server_name, tool_name = qualified_name.split(sep, 1)

        log_args = str(arguments)[:120] if arguments else "—"
        logger.info("🔧 TOOL  %s::%s  args=%s", server_name, tool_name, log_args)

        try:
            result = await asyncio.wait_for(
                _spawn_and_call(server_name, tool_name, arguments, self._cfg.mcp_python),
                timeout=self._cfg.mcp_tool_timeout,
            )
        except asyncio.TimeoutError:
            msg = (
                f"[Timeout: {server_name}::{tool_name} did not respond "
                f"within {self._cfg.mcp_tool_timeout:.0f}s]"
            )
            logger.warning(msg)
            return msg

        snippet = result[:120].replace("\n", " ") + ("…" if len(result) > 120 else "")
        logger.info("✅ TOOL  %s::%s  → %s", server_name, tool_name, snippet)
        logger.debug("RAW_TOOL_RESULT  %s::%s\n%s", server_name, tool_name, result)
        return result

    # ── Parallel dispatch ─────────────────────────────────────────────────────

    async def call_tools_parallel(
        self,
        calls: list[tuple[str, dict[str, Any]]],
    ) -> list[str]:
        """Dispatch multiple independent tool calls concurrently.

        Args:
            calls: List of ``(qualified_name, arguments)`` tuples.
                   All calls must be independent (no prerequisites between them).

        Returns:
            List of result strings in the same order as *calls*.
            Exceptions are caught and returned as error strings — one failed
            tool never cancels the others.
        """
        tasks = [self.call_tool(name, args) for name, args in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: list[str] = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                name = calls[i][0]
                logger.error("Parallel tool %r raised: %s", name, res)
                output.append(f"[Error in {name}: {res}]")
            else:
                output.append(str(res))
        return output
