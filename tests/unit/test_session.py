# SPDX-License-Identifier: AGPL-3.0-or-later
"""
tests/unit/test_session.py
==========================
Unit tests for atlas.core.session — SessionLog open, append, close, hub wiring.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from atlas.config import Config, ConfigError
from atlas.core.session import SessionLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_vault(tmp_path: Path) -> Path:
    """Return a fresh temporary vault directory."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture()
def config(tmp_vault: Path, tmp_path: Path) -> Config:
    """Config pointing at the temporary vault."""
    env = {
        "ATLAS_VAULT_PATH": str(tmp_vault),
        "WHISPER_MODEL_PATH": str(tmp_path / "model.bin"),
        "WAKE_WORD_MODEL_PATHS": str(tmp_path / "Atlas.onnx"),
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.2",
        "CLAUDE_API_KEY": "",
    }
    with patch.dict(os.environ, env, clear=True):
        return Config.from_env()


# ---------------------------------------------------------------------------
# SessionLog lifecycle
# ---------------------------------------------------------------------------


class TestSessionLogOpen:
    def test_creates_sessions_directory(self, config: Config) -> None:
        log = SessionLog(config)
        assert (config.vault_path / "Sessions").is_dir()

    def test_creates_session_file(self, config: Config) -> None:
        log = SessionLog(config)
        assert log.path.exists()

    def test_session_file_in_sessions_subdir(self, config: Config) -> None:
        log = SessionLog(config)
        assert log.path.parent.name == "Sessions"

    def test_session_file_has_md_extension(self, config: Config) -> None:
        log = SessionLog(config)
        assert log.path.suffix == ".md"

    def test_frontmatter_written(self, config: Config) -> None:
        log = SessionLog(config)
        content = log.path.read_text(encoding="utf-8")
        assert "---" in content
        assert "type: session" in content

    def test_sessions_hub_created(self, config: Config) -> None:
        _log = SessionLog(config)
        hub = config.vault_path / "Sessions.md"
        assert hub.exists()


class TestSessionLogAppend:
    def test_append_turn_writes_to_file(self, config: Config) -> None:
        log = SessionLog(config)
        log.append_turn(
            speaker="Roma",
            user_text="What time is it?",
            tools_called=[],
            reply="It's 10 AM.",
        )
        content = log.path.read_text(encoding="utf-8")
        assert "What time is it?" in content
        assert "It's 10 AM." in content

    def test_append_increments_turn_counter(self, config: Config) -> None:
        log = SessionLog(config)
        log.append_turn("Roma", "Q1", [], "R1")
        log.append_turn("Roma", "Q2", [], "R2")
        assert log.turn_count == 2

    def test_append_records_speakers(self, config: Config) -> None:
        log = SessionLog(config)
        log.append_turn("Roma", "Hello", [], "Hi there")
        log.append_turn("Alice", "How are you?", [], "Fine thanks")
        assert "Roma" in log.speakers
        assert "Alice" in log.speakers

    def test_append_with_tools_called(self, config: Config) -> None:
        log = SessionLog(config)
        log.append_turn(
            speaker="Roma",
            user_text="What's the weather?",
            tools_called=["weather__get_weather"],
            reply="It's 22°C.",
        )
        content = log.path.read_text(encoding="utf-8")
        assert "weather" in content.lower()

    def test_multiple_appends_all_present(self, config: Config) -> None:
        log = SessionLog(config)
        for i in range(5):
            log.append_turn("Roma", f"Question {i}", [], f"Answer {i}")
        content = log.path.read_text(encoding="utf-8")
        for i in range(5):
            assert f"Question {i}" in content


class TestSessionLogClose:
    def test_close_writes_footer(self, config: Config) -> None:
        log = SessionLog(config)
        log.append_turn("Roma", "Test", [], "Done")
        log.close()
        content = log.path.read_text(encoding="utf-8")
        # Footer should mention turns or stats
        assert "1" in content  # at least one turn logged

    def test_close_adds_sessions_backlink(self, config: Config) -> None:
        log = SessionLog(config)
        log.close()
        content = log.path.read_text(encoding="utf-8")
        assert "[[Sessions" in content

    def test_close_registers_in_hub(self, config: Config) -> None:
        log = SessionLog(config)
        log.close()
        hub = config.vault_path / "Sessions.md"
        hub_content = hub.read_text(encoding="utf-8")
        assert log.path.stem in hub_content

    def test_double_close_is_safe(self, config: Config) -> None:
        log = SessionLog(config)
        log.close()
        log.close()  # Should not raise

    def test_close_without_appends(self, config: Config) -> None:
        log = SessionLog(config)
        log.close()  # Empty session — should not raise
        assert log.path.exists()


# ---------------------------------------------------------------------------
# Hub wiring
# ---------------------------------------------------------------------------


class TestSessionsHub:
    def test_hub_has_session_link(self, config: Config) -> None:
        log = SessionLog(config)
        log.close()
        hub_content = (config.vault_path / "Sessions.md").read_text(encoding="utf-8")
        assert f"[[Sessions/{log.path.stem}]]" in hub_content

    def test_second_session_appended_to_hub(self, config: Config) -> None:
        log1 = SessionLog(config)
        log1.close()
        log2 = SessionLog(config)
        log2.close()
        hub_content = (config.vault_path / "Sessions.md").read_text(encoding="utf-8")
        assert log1.path.stem in hub_content
        assert log2.path.stem in hub_content
