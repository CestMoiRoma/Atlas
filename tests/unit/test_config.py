# SPDX-License-Identifier: AGPL-3.0-or-later
"""
tests/unit/test_config.py
=========================
Unit tests for atlas.config — Config dataclass and OllamaOptions.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from atlas.config import Config, ConfigError, OllamaOptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_ENV: dict[str, str] = {
    "ATLAS_VAULT_PATH": "/tmp/vault",
    "WHISPER_MODEL_PATH": "/tmp/ggml-base.bin",
    "WAKE_WORD_MODEL_PATHS": "/tmp/Atlas.onnx",
    "CLAUDE_API_KEY": "",         # optional — empty is fine
    "OLLAMA_HOST": "http://localhost:11434",
    "OLLAMA_MODEL": "llama3.2",
}


def _env(**overrides: str) -> dict[str, str]:
    """Return minimal valid env merged with overrides."""
    return {**_MINIMAL_ENV, **overrides}


# ---------------------------------------------------------------------------
# OllamaOptions
# ---------------------------------------------------------------------------


class TestOllamaOptions:
    def test_defaults_all_none(self) -> None:
        opts = OllamaOptions()
        assert opts.temperature is None
        assert opts.num_ctx is None
        assert opts.top_p is None
        assert opts.top_k is None
        assert opts.repeat_penalty is None

    def test_to_dict_omits_none(self) -> None:
        opts = OllamaOptions(temperature=0.7, num_ctx=4096)
        d = opts.to_dict()
        assert d == {"temperature": 0.7, "num_ctx": 4096}
        assert "top_p" not in d

    def test_to_dict_empty_when_all_none(self) -> None:
        assert OllamaOptions().to_dict() == {}

    def test_all_fields_included(self) -> None:
        opts = OllamaOptions(
            temperature=0.5,
            num_ctx=2048,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
        )
        d = opts.to_dict()
        assert len(d) == 5


# ---------------------------------------------------------------------------
# Config.from_env — happy path
# ---------------------------------------------------------------------------


class TestConfigFromEnv:
    def test_loads_minimal_env(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            cfg = Config.from_env()
        assert cfg.ollama_host == "http://localhost:11434"
        assert cfg.ollama_model == "llama3.2"

    def test_vault_path_is_path(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            cfg = Config.from_env()
        assert isinstance(cfg.vault_path, Path)
        assert str(cfg.vault_path) == "/tmp/vault"

    def test_whisper_model_path_is_path(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            cfg = Config.from_env()
        assert isinstance(cfg.whisper_model_path, Path)

    def test_wake_word_models_is_list_of_paths(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            cfg = Config.from_env()
        assert isinstance(cfg.wake_word_models, list)
        assert all(isinstance(p, Path) for p in cfg.wake_word_models)
        assert len(cfg.wake_word_models) == 1

    def test_multiple_wake_word_models(self) -> None:
        env = _env(WAKE_WORD_MODEL_PATHS="/tmp/a.onnx,/tmp/b.onnx")
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        assert len(cfg.wake_word_models) == 2

    def test_default_no_speech_threshold(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            cfg = Config.from_env()
        assert cfg.whisper_no_speech_threshold == pytest.approx(0.6)

    def test_custom_no_speech_threshold(self) -> None:
        env = _env(WHISPER_NO_SPEECH_THRESHOLD="0.8")
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        assert cfg.whisper_no_speech_threshold == pytest.approx(0.8)

    def test_default_tool_timeout(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            cfg = Config.from_env()
        assert cfg.mcp_tool_timeout == pytest.approx(10.0)

    def test_custom_tool_timeout(self) -> None:
        env = _env(MCP_TOOL_TIMEOUT="30.0")
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        assert cfg.mcp_tool_timeout == pytest.approx(30.0)

    def test_frozen_raises_on_mutation(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            cfg = Config.from_env()
        with pytest.raises((AttributeError, TypeError)):
            cfg.ollama_model = "changed"  # type: ignore[misc]

    def test_ollama_options_from_env(self) -> None:
        env = _env(OLLAMA_TEMPERATURE="0.5", OLLAMA_NUM_CTX="8192")
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        d = cfg.ollama_options_dict()
        assert d["temperature"] == pytest.approx(0.5)
        assert d["num_ctx"] == 8192

    def test_ollama_options_empty_when_unset(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            cfg = Config.from_env()
        assert cfg.ollama_options_dict() == {}


# ---------------------------------------------------------------------------
# Config.from_env — validation errors
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_missing_vault_path_raises(self) -> None:
        env = {k: v for k, v in _env().items() if k != "ATLAS_VAULT_PATH"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="ATLAS_VAULT_PATH"):
                Config.from_env()

    def test_missing_whisper_model_raises(self) -> None:
        env = {k: v for k, v in _env().items() if k != "WHISPER_MODEL_PATH"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="WHISPER_MODEL_PATH"):
                Config.from_env()

    def test_missing_wake_word_models_raises(self) -> None:
        env = {k: v for k, v in _env().items() if k != "WAKE_WORD_MODEL_PATHS"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="WAKE_WORD_MODEL_PATHS"):
                Config.from_env()

    def test_invalid_float_threshold_raises(self) -> None:
        env = _env(WHISPER_NO_SPEECH_THRESHOLD="not_a_float")
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError):
                Config.from_env()

    def test_invalid_tool_timeout_raises(self) -> None:
        env = _env(MCP_TOOL_TIMEOUT="abc")
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError):
                Config.from_env()
