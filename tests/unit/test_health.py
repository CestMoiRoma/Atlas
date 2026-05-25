# SPDX-License-Identifier: AGPL-3.0-or-later
"""
tests/unit/test_health.py
=========================
Unit tests for atlas.core.health — startup health checks with mocked deps.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas.core.health import CheckResult, HealthCheckError, run_health_check


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config(tmp_path: Path):
    """Config pointing at a fully valid (mocked) environment."""
    from atlas.config import Config

    # Create stubs so filesystem checks pass
    model_bin = tmp_path / "model.bin"
    model_bin.write_bytes(b"stub")
    onnx = tmp_path / "Atlas.onnx"
    onnx.write_bytes(b"stub")
    vault = tmp_path / "vault"
    vault.mkdir()
    db_path = tmp_path / "users.db"
    # Pre-create a valid SQLite DB
    conn = sqlite3.connect(str(db_path))
    conn.close()

    env = {
        "ATLAS_VAULT_PATH": str(vault),
        "WHISPER_MODEL_PATH": str(model_bin),
        "WAKE_WORD_MODEL_PATHS": str(onnx),
        "SPEAKER_DB_PATH": str(db_path),
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.2",
        "CLAUDE_API_KEY": "",
        "WHISPER_BIN": "whisper-cli",
    }
    with patch.dict(os.environ, env, clear=True):
        return Config.from_env()


# ---------------------------------------------------------------------------
# CheckResult dataclass
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_passed_result(self) -> None:
        r = CheckResult(name="Ollama", passed=True, critical=True, message="OK")
        assert r.passed
        assert r.critical

    def test_failed_non_critical(self) -> None:
        r = CheckResult(name="Vault", passed=False, critical=False, message="Not found")
        assert not r.passed
        assert not r.critical

    def test_message_stored(self) -> None:
        r = CheckResult(name="DB", passed=True, critical=True, message="Connected")
        assert r.message == "Connected"


# ---------------------------------------------------------------------------
# Individual check functions (mocked)
# ---------------------------------------------------------------------------


class TestHealthChecksIndividual:
    @pytest.mark.asyncio
    async def test_check_ollama_passes_on_200(self, config) -> None:
        from atlas.core.health import _check_ollama

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await _check_ollama(config)

        assert result.passed

    @pytest.mark.asyncio
    async def test_check_ollama_fails_on_connection_error(self, config) -> None:
        import httpx
        from atlas.core.health import _check_ollama

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await _check_ollama(config)

        assert not result.passed
        assert result.critical

    @pytest.mark.asyncio
    async def test_check_whisper_bin_fails_when_not_found(self, config) -> None:
        from atlas.core.health import _check_whisper_bin

        with patch("shutil.which", return_value=None):
            result = await _check_whisper_bin(config)

        assert not result.passed

    @pytest.mark.asyncio
    async def test_check_whisper_bin_passes_when_found(self, config) -> None:
        from atlas.core.health import _check_whisper_bin

        with patch("shutil.which", return_value="/usr/local/bin/whisper-cli"):
            result = await _check_whisper_bin(config)

        assert result.passed

    @pytest.mark.asyncio
    async def test_check_whisper_model_passes_when_exists(self, config) -> None:
        from atlas.core.health import _check_whisper_model

        result = await _check_whisper_model(config)
        # config.whisper_model_path is tmp stub that exists
        assert result.passed

    @pytest.mark.asyncio
    async def test_check_whisper_model_fails_when_missing(self, config, tmp_path) -> None:
        from atlas.config import Config
        from atlas.core.health import _check_whisper_model

        missing = tmp_path / "missing_model.bin"
        env = {
            "ATLAS_VAULT_PATH": str(tmp_path / "vault"),
            "WHISPER_MODEL_PATH": str(missing),
            "WAKE_WORD_MODEL_PATHS": str(tmp_path / "Atlas.onnx"),
            "OLLAMA_HOST": "http://localhost:11434",
            "OLLAMA_MODEL": "llama3.2",
            "CLAUDE_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=True):
            bad_config = Config.from_env()

        result = await _check_whisper_model(bad_config)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_check_wake_word_models_passes(self, config) -> None:
        from atlas.core.health import _check_wake_word_models

        result = await _check_wake_word_models(config)
        assert result.passed

    @pytest.mark.asyncio
    async def test_check_database_passes_on_valid_db(self, config) -> None:
        from atlas.core.health import _check_database

        result = await _check_database(config)
        assert result.passed

    @pytest.mark.asyncio
    async def test_check_vault_passes_when_dir_exists(self, config) -> None:
        from atlas.core.health import _check_vault

        result = await _check_vault(config)
        assert result.passed


# ---------------------------------------------------------------------------
# run_health_check — integration of all checks
# ---------------------------------------------------------------------------


class TestRunHealthCheck:
    @pytest.mark.asyncio
    async def test_passes_all_mocked_checks(self, config, capsys) -> None:
        """All checks mocked as passing — no exception raised."""
        passing = CheckResult(name="x", passed=True, critical=True, message="ok")

        async def _pass(cfg):
            return passing

        check_fns = [
            "atlas.core.health._check_ollama",
            "atlas.core.health._check_whisper_bin",
            "atlas.core.health._check_whisper_model",
            "atlas.core.health._check_wake_word_models",
            "atlas.core.health._check_database",
            "atlas.core.health._check_vault",
        ]

        patches = [patch(fn, side_effect=_pass) for fn in check_fns]
        for p in patches:
            p.start()
        try:
            await run_health_check(config)  # Should not raise
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_raises_on_critical_failure(self, config) -> None:
        """A critical failure should raise HealthCheckError."""
        failing = CheckResult(
            name="Ollama", passed=False, critical=True, message="Connection refused"
        )

        async def _fail(cfg):
            return failing

        with patch("atlas.core.health._check_ollama", side_effect=_fail):
            with pytest.raises(HealthCheckError):
                await run_health_check(config)

    @pytest.mark.asyncio
    async def test_no_raise_on_non_critical_failure(self, config) -> None:
        """A non-critical failure should NOT raise — just print a warning."""
        warning = CheckResult(
            name="Vault", passed=False, critical=False, message="Not configured"
        )

        # All critical checks pass, vault (non-critical) fails
        async def _pass_ollama(cfg):
            return CheckResult("Ollama", True, True, "ok")

        async def _pass_whisper_bin(cfg):
            return CheckResult("whisper-cli", True, True, "ok")

        async def _pass_whisper_model(cfg):
            return CheckResult("Whisper model", True, True, "ok")

        async def _pass_ww(cfg):
            return CheckResult("Wake word", True, True, "ok")

        async def _pass_db(cfg):
            return CheckResult("Database", True, True, "ok")

        async def _warn_vault(cfg):
            return warning

        with (
            patch("atlas.core.health._check_ollama", side_effect=_pass_ollama),
            patch("atlas.core.health._check_whisper_bin", side_effect=_pass_whisper_bin),
            patch("atlas.core.health._check_whisper_model", side_effect=_pass_whisper_model),
            patch("atlas.core.health._check_wake_word_models", side_effect=_pass_ww),
            patch("atlas.core.health._check_database", side_effect=_pass_db),
            patch("atlas.core.health._check_vault", side_effect=_warn_vault),
        ):
            await run_health_check(config)  # Must not raise
