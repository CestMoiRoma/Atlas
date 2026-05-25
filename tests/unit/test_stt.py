# SPDX-License-Identifier: AGPL-3.0-or-later
"""
tests/unit/test_stt.py
======================
Unit tests for atlas.core.stt — VAD helpers and no_speech_prob filtering.

Heavy I/O (sounddevice, whisper-cli subprocess) is mocked so tests run in CI
without any audio hardware or model files.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from atlas.core.stt import STT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config(tmp_path: Path):
    """Minimal Config for STT tests."""
    from atlas.config import Config

    env = {
        "ATLAS_VAULT_PATH": str(tmp_path / "vault"),
        "WHISPER_MODEL_PATH": str(tmp_path / "model.bin"),
        "WAKE_WORD_MODEL_PATHS": str(tmp_path / "Atlas.onnx"),
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.2",
        "CLAUDE_API_KEY": "",
        "WHISPER_NO_SPEECH_THRESHOLD": "0.6",
    }
    with patch.dict(os.environ, env, clear=True):
        return Config.from_env()


@pytest.fixture()
def stt(config) -> STT:
    return STT(config)


# ---------------------------------------------------------------------------
# _parse_whisper_json — internal helper exposed via private method tests
# ---------------------------------------------------------------------------


def _make_whisper_json(text: str = "Bonjour.", no_speech_prob: float = 0.1) -> str:
    """Build a minimal whisper --output-format json payload."""
    return json.dumps(
        {
            "transcription": [
                {
                    "text": text,
                    "timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"},
                    "offsets": {"from": 0, "to": 1000},
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": -0.3,
                    "compression_ratio": 1.2,
                    "no_speech_prob": no_speech_prob,
                }
            ]
        }
    )


class TestParseWhisperJson:
    def test_returns_text_below_threshold(self, stt: STT) -> None:
        raw = _make_whisper_json("Bonjour.", no_speech_prob=0.1)
        result = stt._parse_whisper_json(raw)
        assert result == "Bonjour."

    def test_returns_empty_above_threshold(self, stt: STT) -> None:
        raw = _make_whisper_json("Merci.", no_speech_prob=0.9)
        result = stt._parse_whisper_json(raw)
        assert result == ""

    def test_returns_empty_at_exact_threshold(self, stt: STT) -> None:
        # At exactly 0.6 → should be discarded (>= threshold)
        raw = _make_whisper_json("Hmm.", no_speech_prob=0.6)
        result = stt._parse_whisper_json(raw)
        assert result == ""

    def test_strips_leading_trailing_whitespace(self, stt: STT) -> None:
        raw = _make_whisper_json("  Bonjour le monde.  ", no_speech_prob=0.05)
        result = stt._parse_whisper_json(raw)
        assert result == "Bonjour le monde."

    def test_handles_empty_transcription_list(self, stt: STT) -> None:
        raw = json.dumps({"transcription": []})
        result = stt._parse_whisper_json(raw)
        assert result == ""

    def test_handles_invalid_json(self, stt: STT) -> None:
        result = stt._parse_whisper_json("not json at all")
        assert result == ""

    def test_handles_missing_no_speech_prob(self, stt: STT) -> None:
        """If no_speech_prob key is absent, treat as 0 (keep transcript)."""
        raw = json.dumps(
            {
                "transcription": [
                    {
                        "text": "Bonjour.",
                        "timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"},
                    }
                ]
            }
        )
        result = stt._parse_whisper_json(raw)
        assert result == "Bonjour."

    def test_custom_threshold_respected(self, tmp_path: Path) -> None:
        """Config with threshold=0.3 should discard text at 0.4."""
        from atlas.config import Config

        env = {
            "ATLAS_VAULT_PATH": str(tmp_path / "vault"),
            "WHISPER_MODEL_PATH": str(tmp_path / "model.bin"),
            "WAKE_WORD_MODEL_PATHS": str(tmp_path / "Atlas.onnx"),
            "OLLAMA_HOST": "http://localhost:11434",
            "OLLAMA_MODEL": "llama3.2",
            "CLAUDE_API_KEY": "",
            "WHISPER_NO_SPEECH_THRESHOLD": "0.3",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        custom_stt = STT(cfg)
        raw = _make_whisper_json("Test.", no_speech_prob=0.4)
        result = custom_stt._parse_whisper_json(raw)
        assert result == ""


# ---------------------------------------------------------------------------
# STT.transcribe — subprocess mock
# ---------------------------------------------------------------------------


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self, stt: STT, tmp_path: Path) -> None:
        """Mock whisper-cli to verify transcribe() returns parsed text."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)  # minimal stub

        good_json = _make_whisper_json("Quelle heure est-il ?", no_speech_prob=0.05)

        with patch.object(stt, "_transcribe_sync", return_value="Quelle heure est-il ?"):
            result = await stt.transcribe(wav_file)

        assert result == "Quelle heure est-il ?"

    @pytest.mark.asyncio
    async def test_transcribe_empty_on_phantom(self, stt: STT, tmp_path: Path) -> None:
        """High no_speech_prob → empty string returned."""
        wav_file = tmp_path / "silence.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(stt, "_transcribe_sync", return_value=""):
            result = await stt.transcribe(wav_file)

        assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_missing_file_raises(self, stt: STT, tmp_path: Path) -> None:
        """Transcribing a non-existent file should raise FileNotFoundError."""
        with pytest.raises((FileNotFoundError, OSError)):
            await stt.transcribe(tmp_path / "nonexistent.wav")
