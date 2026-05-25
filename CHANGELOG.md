# Changelog

All notable changes to Atlas will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial public release — complete rewrite from scratch (Aether prototype → Atlas v1.0.0)

---

## [1.0.0] — 2025

### Added

#### Core pipeline
- `atlas/core/wake_word.py` — livekit-wakeword ONNX listener with configurable threshold
  and debounce; `models/Atlas.onnx` (97 KB custom model) committed as static asset
- `atlas/core/stt.py` — whisper.cpp integration with energy-based VAD, pre-speech buffer,
  and `no_speech_prob` filter (fixes phantom transcription bug documented in prototype)
- `atlas/core/tts.py` — macOS `say` wrapper with 16-pattern Markdown stripper and
  async-first audio gate integration
- `atlas/core/speaker_id.py` — SpeechBrain ECAPA-TDNN speaker identification with
  three-tier matching (exact / fallback / guest) and self-improving embeddings
- `atlas/core/audio_gate.py` — async anti-feedback gate using `asyncio.Event`
  (replaces prototype's `threading.Event`)
- `atlas/core/session.py` — dedicated `SessionLog` module (extracted from orchestrator):
  crash-safe immediate writes, Sessions hub wiring, footer stats
- `atlas/core/mcp_client.py` — MCP client with per-tool timeout (`asyncio.wait_for`),
  parallel dispatch (`asyncio.gather`), and prerequisite ordering
- `atlas/core/health.py` — 6-check startup health verification with colored terminal output
- `atlas/core/orchestrator.py` — main async pipeline with multi-round tool loop,
  `[SUITE]` continuation sentinel, sleeping mode monitor, `--text` / `--nothink` / `--check` flags

#### Configuration
- `atlas/config.py` — frozen `Config` dataclass centralising all runtime parameters;
  `OllamaOptions` for LLM tuning; `ConfigError` with actionable messages
- `.env.example` — fully annotated environment template (20+ variables)

#### Persistence
- `atlas/db/user_db.py` — SQLite WAL schema with upsert, separate embedding update,
  numpy float32 BLOB serialisation

#### MCP tools (7 servers)
- `atlas/tools/memory.py` — 8 vault operations with path traversal guard
- `atlas/tools/datetime_info.py` — French locale date/time
- `atlas/tools/geoposition.py` — IP geolocation via ip-api.com
- `atlas/tools/weather.py` — Open-Meteo with WMO code descriptions in French
- `atlas/tools/metrics.py` — system metrics via psutil
- `atlas/tools/wikipedia.py` — French Wikipedia REST API (search + full article)
- `atlas/tools/inbox.py` — multi-format inbox reader (.txt, .md, .json, .csv, .drawio)

#### Scripts
- `scripts/download_models.py` — model downloader with SHA-256 verification and
  exponential backoff retry (SpeechBrain auto, Atlas.onnx verify, Whisper guide)
- `scripts/register_user.py` — voice enrollment (5 × 4s clips, ECAPA embedding)
- `scripts/edit_user.py` — profile field editing with `--list` overview
- `scripts/embed_memory.py` — chunked Ollama embedding to review vault
- `scripts/embed_deep.py` — large-context sliding window (100K chars, 50% overlap)
- `scripts/unify_topics.py` — Jaccard-based duplicate topic merging with `--auto` mode
- `scripts/index_sessions.py` — retroactive session hub indexing

#### Tests
- `tests/unit/test_config.py` — 37 tests: Config/OllamaOptions validation and edge cases
- `tests/unit/test_models.py` — 18 tests: User, GUEST_USER, SpeakerMatch
- `tests/unit/test_session.py` — 18 tests: SessionLog lifecycle and hub wiring
- `tests/unit/test_stt.py` — 11 tests: `_parse_whisper_json` and `no_speech_prob` filtering
- `tests/unit/test_health.py` — 16 tests: individual checks and `run_health_check` integration

#### Documentation
- `docs/wiki/01-internal-architecture.md` — full pipeline diagram, module reference, Config table
- `docs/wiki/02-contributor-guide.md` — dev setup, code standards, adding tools/models, PR workflow
- `docs/wiki/03-user-manual.md` — installation, configuration, usage, 8-entry FAQ

#### CI
- `.github/workflows/ci.yml` — ruff lint + pytest unit tests on Python 3.10 and 3.12

### Changed
- All `os.getenv()` calls (~40 in prototype) replaced by `Config.from_env()` injection
- `threading.Event` replaced by `asyncio.Event` throughout pipeline
- Vault path variable renamed `OBSIDIAN_VAULT_PATH` → `ATLAS_VAULT_PATH`
- Default `SPEAKER_DB_PATH` changed to `./atlas_users.db`

### Fixed
- Phantom transcription bug — whisper hallucinating text on silence, fixed via
  `no_speech_prob` threshold filter in `stt.py`

### Security
- Path traversal guard in `memory.py` — all vault paths validated against vault root
  before any read or write operation

[Unreleased]: https://github.com/CestMoiRoma/Atlas/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/CestMoiRoma/Atlas/releases/tag/v1.0.0
