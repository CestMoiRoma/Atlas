# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/config.py
===============
Centralised, typed configuration for Atlas.

All runtime parameters are loaded once at startup via ``Config.from_env()``,
which reads the active ``.env`` file and the process environment.  Every
field is explicitly typed; invalid or missing required values raise
``ConfigError`` with a human-readable message before the main loop starts.

Usage::

    from atlas.config import Config
    cfg = Config.from_env()
    print(cfg.ollama_model)

Tests can inject a config directly without touching the environment::

    cfg = Config(ollama_model="test-model", whisper_model=Path("/tmp/test.bin"), ...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class ConfigError(ValueError):
    """Raised when a required configuration value is missing or invalid."""


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a valid float") from exc


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a valid integer") from exc


def _get_bool(key: str, default: bool = False) -> bool:
    raw = _get(key).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _get_path(key: str, default: str = "") -> Path:
    raw = _get(key, default)
    return Path(raw) if raw else Path(default)


def _get_optional_float(key: str) -> float | None:
    raw = _get(key)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a valid float") from exc


@dataclass(frozen=True)
class OllamaOptions:
    """Generation parameters forwarded verbatim to ``ollama.chat(options=...)``.

    Only parameters that are explicitly set in the environment are included.
    Unset parameters are omitted so Ollama falls back to the model's own
    defaults — no silent zeroing.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    num_ctx: int | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    min_p: float | None = None
    num_predict: int | None = None
    tfs_z: float | None = None
    typical_p: float | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return only the fields that are explicitly set (not None)."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_env(cls) -> "OllamaOptions":
        _int_keys = {"top_k", "num_ctx", "num_predict", "repeat_last_n", "seed"}
        _mapping = {
            "temperature":    "OLLAMA_TEMPERATURE",
            "top_p":          "OLLAMA_TOP_P",
            "top_k":          "OLLAMA_TOP_K",
            "num_ctx":        "OLLAMA_NUM_CTX",
            "repeat_penalty": "OLLAMA_REPEAT_PENALTY",
            "repeat_last_n":  "OLLAMA_REPEAT_LAST_N",
            "min_p":          "OLLAMA_MIN_P",
            "num_predict":    "OLLAMA_NUM_PREDICT",
            "tfs_z":          "OLLAMA_TFS_Z",
            "typical_p":      "OLLAMA_TYPICAL_P",
            "seed":           "OLLAMA_SEED",
        }
        kwargs: dict[str, Any] = {}
        for field_name, env_key in _mapping.items():
            raw = _get(env_key)
            if not raw:
                continue
            try:
                kwargs[field_name] = int(raw) if field_name in _int_keys else float(raw)
            except ValueError as exc:
                raise ConfigError(f"{env_key}={raw!r} is not a valid number") from exc
        return cls(**kwargs)


@dataclass(frozen=True)
class Config:
    """Complete Atlas runtime configuration.

    Instantiate via ``Config.from_env()`` in production.
    Pass fields directly in tests to avoid environment coupling.
    """

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_options: OllamaOptions = field(default_factory=OllamaOptions)

    # ── STT ───────────────────────────────────────────────────────────────────
    whisper_bin: str = "whisper-cli"
    whisper_model: Path = field(default_factory=lambda: Path("/path/to/model.bin"))
    whisper_language: str = "en"
    whisper_no_speech_threshold: float = 0.6

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_rate: int | None = None  # words per minute; None = system default

    # ── Wake word ─────────────────────────────────────────────────────────────
    wake_word_models: list[Path] = field(default_factory=lambda: [Path("models/Atlas.onnx")])
    wake_word_threshold: float = 0.5
    wake_word_debounce: float = 2.0

    # ── Audio ─────────────────────────────────────────────────────────────────
    audio_input_device: int = -1
    audio_sample_rate: int = 16000

    # ── VAD ───────────────────────────────────────────────────────────────────
    vad_energy_threshold: int = 300
    vad_speech_timeout: float = 8.0
    vad_silence_after: float = 1.2
    vad_max_duration: float = 30.0

    # ── Paths ─────────────────────────────────────────────────────────────────
    inbox_path: Path = field(default_factory=lambda: Path("./atlas_inbox"))
    vault_path: Path = field(default_factory=lambda: Path("./atlas_memory"))
    speaker_db_path: Path = field(default_factory=lambda: Path("./atlas_users.db"))
    voice_templates_dir: Path = field(default_factory=lambda: Path("./user_voice_templates"))

    # ── MCP ───────────────────────────────────────────────────────────────────
    mcp_python: str = "python3"
    mcp_tool_timeout: float = 10.0
    max_tool_rounds: int = 10

    # ── Speaker ID ────────────────────────────────────────────────────────────
    speaker_id_threshold: float = 0.75
    speaker_fallback_min_score: float = 0.30
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_savedir: Path = field(default_factory=lambda: Path("./models/spkrec-ecapa-voxceleb"))
    embed_update_every: int = 5
    max_samples_per_user: int = 50

    # ── Sleeping mode ─────────────────────────────────────────────────────────
    sleep_timeout: float = 180.0

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file: str = ""  # empty = terminal only

    # ── Thinking ─────────────────────────────────────────────────────────────
    nothink: bool = False
    think_depth: str = "moderate"

    # ── Personalization ───────────────────────────────────────────────────────
    #: Language injected into the voice-rules system prompt ("English", "French", …)
    response_language: str = "English"
    #: Extra instructions appended verbatim after the default voice rules.
    voice_rules_extra: str = ""
    #: Phrases Atlas speaks at random upon wake-word detection.
    wake_ack_phrases: list[str] = field(
        default_factory=lambda: [
            "Yes?", "Listening.", "Yes, go ahead.", "Here.", "I'm listening.",
        ]
    )
    #: Message spoken when the tool-call loop cap is hit.
    atlas_loop_message: str = "I seem to be stuck in a loop. Could you rephrase that?"
    #: Message spoken on an unrecoverable LLM response error.
    atlas_error_message: str = "Sorry, I couldn't process that request."
    #: Message spoken when Atlas enters sleeping mode.
    atlas_sleep_message: str = "Going to sleep."
    #: Optional path to a file whose content overrides the built-in MEMORY_GRAPH block.
    memory_graph_file: str = ""

    # ── Display units ─────────────────────────────────────────────────────────
    #: Clock format used by the datetime tool: ``"24h"`` or ``"12h"``.
    time_format: str = "24h"
    #: Temperature unit used by weather tools: ``"C"`` (Celsius) or ``"F"`` (Fahrenheit).
    temperature_unit: str = "C"

    # ── Derived helpers ───────────────────────────────────────────────────────

    def ollama_options_dict(self) -> dict[str, Any]:
        """Return Ollama generation options as a plain dict (omits None values)."""
        return self.ollama_options.to_dict()

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "Config":
        """Load and validate configuration from the active .env and environment.

        Raises ``ConfigError`` with a clear message if any required value is
        missing or has an invalid type.
        """
        whisper_model_raw = _get("WHISPER_CPP_MODEL")
        if not whisper_model_raw or whisper_model_raw == "/path/to/model.bin":
            raise ConfigError(
                "WHISPER_CPP_MODEL is not set in your .env file.\n"
                "Download a GGML model from:\n"
                "  https://huggingface.co/ggerganov/whisper.cpp/tree/main\n"
                "Then set: WHISPER_CPP_MODEL=/absolute/path/to/ggml-large-v3-turbo.bin"
            )

        wake_word_raw = _get("WAKE_WORD_MODELS", "models/Atlas.onnx")
        wake_word_paths = [Path(p.strip()) for p in wake_word_raw.split(",") if p.strip()]

        _default_wake_ack = ["Yes?", "Listening.", "Yes, go ahead.", "Here.", "I'm listening."]
        wake_ack_raw = _get("WAKE_ACK_PHRASES")
        wake_ack_phrases = (
            [p.strip() for p in wake_ack_raw.split(",") if p.strip()]
            if wake_ack_raw else _default_wake_ack
        )

        tts_rate_raw = _get("TTS_RATE")
        tts_rate: int | None = None
        if tts_rate_raw:
            try:
                tts_rate = int(tts_rate_raw)
            except ValueError as exc:
                raise ConfigError(f"TTS_RATE={tts_rate_raw!r} is not a valid integer") from exc

        return cls(
            # Ollama
            ollama_host=_get("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=_get("OLLAMA_MODEL", "llama3.2"),
            ollama_options=OllamaOptions.from_env(),
            # STT
            whisper_bin=_get("WHISPER_CPP_BIN", "whisper-cli"),
            whisper_model=Path(whisper_model_raw),
            whisper_language=_get("WHISPER_CPP_LANGUAGE", "en"),
            whisper_no_speech_threshold=_get_float("WHISPER_NO_SPEECH_THRESHOLD", 0.6),
            # TTS
            tts_rate=tts_rate,
            # Wake word
            wake_word_models=wake_word_paths,
            wake_word_threshold=_get_float("WAKE_WORD_THRESHOLD", 0.5),
            wake_word_debounce=_get_float("WAKE_WORD_DEBOUNCE", 2.0),
            # Audio
            audio_input_device=_get_int("AUDIO_INPUT_DEVICE", -1),
            audio_sample_rate=_get_int("AUDIO_SAMPLE_RATE", 16000),
            # VAD
            vad_energy_threshold=_get_int("VAD_ENERGY_THRESHOLD", 300),
            vad_speech_timeout=_get_float("VAD_SPEECH_TIMEOUT", 8.0),
            vad_silence_after=_get_float("VAD_SILENCE_AFTER", 1.2),
            vad_max_duration=_get_float("VAD_MAX_DURATION", 30.0),
            # Paths
            inbox_path=_get_path("ATLAS_INBOX_PATH", "./atlas_inbox"),
            vault_path=_get_path("ATLAS_VAULT_PATH", "./atlas_memory"),
            speaker_db_path=_get_path("SPEAKER_DB_PATH", "./atlas_users.db"),
            voice_templates_dir=_get_path("VOICE_TEMPLATES_DIR", "./user_voice_templates"),
            # MCP
            mcp_python=_get("MCP_PYTHON", "python3"),
            mcp_tool_timeout=_get_float("MCP_TOOL_TIMEOUT", 10.0),
            max_tool_rounds=_get_int("MAX_TOOL_ROUNDS", 10),
            # Speaker ID
            speaker_id_threshold=_get_float("SPEAKER_ID_THRESHOLD", 0.75),
            speaker_fallback_min_score=_get_float("SPEAKER_FALLBACK_MIN_SCORE", 0.30),
            speaker_model=_get("SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb"),
            speaker_savedir=_get_path("SPEAKER_SAVEDIR", "./models/spkrec-ecapa-voxceleb"),
            embed_update_every=_get_int("EMBED_UPDATE_EVERY", 5),
            max_samples_per_user=_get_int("MAX_SAMPLES_PER_USER", 50),
            # Sleeping mode
            sleep_timeout=_get_float("SLEEP_TIMEOUT", 180.0),
            # Logging
            log_file=_get("LOG_FILE", ""),
            # Thinking
            nothink=_get_bool("NOTHINK", False),
            think_depth=_get("THINK_DEPTH", "moderate"),
            # Personalization
            response_language=_get("ATLAS_RESPONSE_LANGUAGE", "English"),
            voice_rules_extra=_get("VOICE_RULES_EXTRA", ""),
            wake_ack_phrases=wake_ack_phrases,
            atlas_loop_message=_get(
                "ATLAS_LOOP_MESSAGE",
                "I seem to be stuck in a loop. Could you rephrase that?",
            ),
            atlas_error_message=_get(
                "ATLAS_ERROR_MESSAGE",
                "Sorry, I couldn't process that request.",
            ),
            atlas_sleep_message=_get("ATLAS_SLEEP_MESSAGE", "Going to sleep."),
            memory_graph_file=_get("MEMORY_GRAPH_FILE", ""),
            # Display units
            time_format=_get("TIME_FORMAT", "24h"),
            temperature_unit=_get("TEMPERATURE_UNIT", "C"),
        )
