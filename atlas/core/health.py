# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/health.py
====================
Startup health check — validates all required dependencies before the main
loop begins.

Each check is independent; all checks run even if an earlier one fails so the
user gets a complete picture of what needs to be fixed in one pass.

Checks performed
----------------
1. **Ollama** — HTTP GET ``/api/tags`` on ``config.ollama_host``.  Confirms the
   daemon is running and reachable.
2. **whisper-cli** — ``shutil.which(config.whisper_bin)``.  Confirms the binary
   is on PATH.
3. **Whisper model** — ``config.whisper_model.exists()``.  Confirms the GGML
   model file has been downloaded.
4. **Wake word model(s)** — each path in ``config.wake_word_models`` must exist.
5. **Database** — attempts to open the SQLite file and run a trivial query.
6. **Vault** — ``config.vault_path`` must be an existing directory, or its
   parent must be writable so Atlas can create it.

Exit behaviour
--------------
If any *critical* check fails (Ollama, whisper-cli, whisper model, wake word
model) the function raises ``HealthCheckError`` which the launcher catches to
print diagnostics and call ``sys.exit(1)``.

Non-critical failures (DB, vault) are warned but do not prevent startup —
Atlas will create these on first use.

Usage::

    from atlas.core.health import run_health_check, HealthCheckError
    try:
        await run_health_check(config)
    except HealthCheckError:
        sys.exit(1)
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from atlas.config import Config

logger = logging.getLogger(__name__)

_OK   = "\033[32m✓\033[0m"
_WARN = "\033[33m⚠\033[0m"
_FAIL = "\033[31m✗\033[0m"


class HealthCheckError(RuntimeError):
    """Raised when one or more critical health checks fail."""


@dataclass
class CheckResult:
    name: str
    passed: bool
    critical: bool
    message: str


# ── Individual checks ─────────────────────────────────────────────────────────

async def _check_ollama(config: Config) -> CheckResult:
    url = f"{config.ollama_host.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        return CheckResult("Ollama daemon", True, True, f"reachable at {config.ollama_host}")
    except Exception as exc:
        return CheckResult(
            "Ollama daemon", False, True,
            f"not reachable at {config.ollama_host} — is `ollama serve` running?\n"
            f"  detail: {exc}",
        )


def _check_whisper_bin(config: Config) -> CheckResult:
    found = shutil.which(config.whisper_bin)
    if found:
        return CheckResult("whisper-cli binary", True, True, f"found at {found}")
    return CheckResult(
        "whisper-cli binary", False, True,
        f"{config.whisper_bin!r} not found in PATH — run: brew install whisper-cpp",
    )


def _check_whisper_model(config: Config) -> CheckResult:
    path = config.whisper_model
    if path.exists():
        size_mb = path.stat().st_size / 1_048_576
        return CheckResult("Whisper model", True, True, f"{path}  ({size_mb:.0f} MB)")
    return CheckResult(
        "Whisper model", False, True,
        f"file not found: {path}\n"
        f"  Download from: https://huggingface.co/ggerganov/whisper.cpp/tree/main\n"
        f"  Then set WHISPER_CPP_MODEL=/path/to/ggml-large-v3-turbo.bin in .env",
    )


def _check_wake_word_models(config: Config) -> list[CheckResult]:
    results: list[CheckResult] = []
    for model_path in config.wake_word_models:
        if model_path.exists():
            size_kb = model_path.stat().st_size / 1024
            results.append(CheckResult(
                f"Wake word model ({model_path.name})", True, True,
                f"{model_path}  ({size_kb:.0f} KB)",
            ))
        else:
            results.append(CheckResult(
                f"Wake word model ({model_path.name})", False, True,
                f"file not found: {model_path}",
            ))
    return results


def _check_database(config: Config) -> CheckResult:
    db_path = config.speaker_db_path
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("SELECT 1")
        conn.close()
        return CheckResult("Speaker database", True, False, str(db_path))
    except Exception as exc:
        return CheckResult(
            "Speaker database", False, False,
            f"cannot open {db_path}: {exc} (will be created on first run)",
        )


def _check_tts() -> CheckResult:
    """Detect and report the active TTS backend."""
    from atlas.core.tts import _BACKEND  # noqa: PLC0415
    if _BACKEND == "none":
        return CheckResult(
            "TTS backend", False, False,
            f"no TTS command found (platform={sys.platform}) — "
            "install espeak-ng on Linux: sudo apt install espeak-ng",
        )
    return CheckResult("TTS backend", True, False, _BACKEND)


def _check_vault(config: Config) -> CheckResult:
    vault = config.vault_path
    if vault.exists() and vault.is_dir():
        return CheckResult("Obsidian vault", True, False, str(vault))
    # Not a blocker — Atlas creates the vault on first memory write
    if vault.parent.exists():
        return CheckResult(
            "Obsidian vault", False, False,
            f"{vault} does not exist yet — Atlas will create it on first use",
        )
    return CheckResult(
        "Obsidian vault", False, False,
        f"parent directory {vault.parent} does not exist — check ATLAS_VAULT_PATH in .env",
    )


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_health_check(config: Config) -> None:
    """Run all startup checks and print a diagnostic summary.

    Raises:
        HealthCheckError: if any critical check fails.
    """
    print("\n─── Atlas startup health check ───────────────────────────────")

    results: list[CheckResult] = []

    # Async checks
    results.append(await _check_ollama(config))

    # Sync checks
    results.append(_check_whisper_bin(config))
    results.append(_check_whisper_model(config))
    results.extend(_check_wake_word_models(config))
    results.append(_check_tts())
    results.append(_check_database(config))
    results.append(_check_vault(config))

    # Display
    any_critical_failed = False
    for r in results:
        if r.passed:
            icon = _OK
        elif r.critical:
            icon = _FAIL
            any_critical_failed = True
        else:
            icon = _WARN

        print(f"  {icon}  {r.name}")
        if not r.passed:
            for line in r.message.splitlines():
                print(f"       {line}")

    print("──────────────────────────────────────────────────────────────\n")

    if any_critical_failed:
        raise HealthCheckError(
            "One or more critical checks failed — fix the issues above and restart Atlas."
        )

    logger.info("Health check passed.")
