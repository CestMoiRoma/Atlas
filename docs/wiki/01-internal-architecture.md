# Internal Architecture

> This document describes Atlas internals for contributors and maintainers.
> For installation and usage, see [03-user-manual.md](03-user-manual.md).

---

## Table of contents

1. [Overview — voice pipeline](#1-overview--voice-pipeline)
2. [Package modules `atlas/`](#2-package-modules-atlas)
   - [config.py](#configpy)
   - [db/user_db.py](#dbuser_dbpy)
   - [core/models.py](#coremodels.py)
   - [core/audio_gate.py](#coreaudio_gatepy)
   - [core/wake_word.py](#corewake_wordpy)
   - [core/stt.py](#corestttpy)
   - [core/tts.py](#corettspy)
   - [core/speaker_id.py](#corespeaker_idpy)
   - [core/session.py](#coresessionpy)
   - [core/mcp_client.py](#coremcp_clientpy)
   - [core/health.py](#corehealthpy)
   - [core/orchestrator.py](#coreorchestratorpy)
3. [Multi-round tool loop](#3-multi-round-tool-loop)
4. [MCP servers — atlas/tools/](#4-mcp-servers--atlastools)
5. [Obsidian memory system](#5-obsidian-memory-system)
6. [Sleeping mode](#6-sleeping-mode)
7. [Config reference](#7-config-reference)

---

## 1. Overview — voice pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ATLAS — Voice Pipeline                            │
└──────────────────────────────────────────────────────────────────────────┘

  Microphone
      │
      ▼
 ┌─────────────┐    keyword      ┌──────────────────────────┐
 │  WakeWord   │ ──────────────▶ │  STT  (whisper-cli)      │
 │  Listener   │  (Atlas.onnx)   │  VAD + no_speech_prob    │
 └─────────────┘                 └────────────┬─────────────┘
                                              │ raw text
                                              ▼
                                 ┌──────────────────────────┐
                                 │  SpeakerIdentifier       │
                                 │  ECAPA-TDNN + cosine     │
                                 └────────────┬─────────────┘
                                              │ User + SpeakerMatch
                                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │                      Orchestrator                         │
 │                                                           │
 │  system prompt  ──▶  Ollama AsyncClient (LLM)            │
 │                           │                              │
 │                    ┌──────▼──────┐                       │
 │                    │ tool_calls? │                       │
 │                    └──┬──────────┘                       │
 │                       │ yes           ┌─────────────────┐ │
 │                       └────────────▶  │  MCPClient      │ │
 │                                       │  timeout+gather │ │
 │                                       └────────┬────────┘ │
 │                       ┌──────────────────────◀─┘          │
 │                       │ tool results                       │
 │                       ▼                                    │
 │                  next round (MAX_TOOL_ROUNDS)              │
 │                       │ final reply                        │
 └───────────────────────┼───────────────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │  TTS (macOS say)        │◀── AudioGate (anti-feedback)
            │  + strip Markdown       │
            └─────────────────────────┘
                         │
                         ▼
                   SessionLog.append_turn()
                   → Sessions/<date>.md
```

**8 steps in order:**

| # | Step | Module |
|---|------|--------|
| 1 | Wake word detection | `core/wake_word.py` |
| 2 | Recording + VAD | `core/stt.py` |
| 3 | Whisper transcription | `core/stt.py` |
| 4 | Speaker identification | `core/speaker_id.py` |
| 5 | System prompt construction | `core/orchestrator.py` |
| 6 | LLM inference + tool loop | `core/orchestrator.py` + `core/mcp_client.py` |
| 7 | Speech synthesis | `core/tts.py` |
| 8 | Session persistence | `core/session.py` |

---

## 2. Package modules `atlas/`

### `config.py`

Single entry point for all configuration. Two dataclasses:

```python
@dataclass(frozen=True)
class OllamaOptions:
    temperature: float | None = None
    num_ctx: int | None = None
    # … other generation params

    def to_dict(self) -> dict: ...  # Returns only non-None fields

@dataclass(frozen=True)
class Config:
    @classmethod
    def from_env(cls) -> "Config": ...  # Raises ConfigError if required var missing/invalid
```

`Config.from_env()` reads the `.env` file via `python-dotenv`, validates types, and raises
`ConfigError` (a `ValueError` subclass) with an actionable message if a required variable
is absent or malformed.

**Advantages over scattered `os.getenv()` calls:**
- All variables documented in one place
- Direct injection in tests without global monkeypatching
- Frozen → immutable at runtime, catches accidental mutation bugs

---

### `db/user_db.py`

SQLite WAL layer, idempotent schema (created on first `init_db()`).

```
Table: users
  id               INTEGER PRIMARY KEY
  name             TEXT UNIQUE NOT NULL
  user_tag         TEXT
  age              INTEGER
  gender           TEXT
  profession       TEXT
  preferred_address TEXT
  other_addresses  TEXT   -- JSON list
  embedding        BLOB   -- numpy float32, serialised via tobytes()/frombuffer()
```

**Public API:**

| Function | Description |
|----------|-------------|
| `init_db(path)` | Create/open DB, enable WAL, return `Connection` |
| `upsert_user(db, ...)` | INSERT … ON CONFLICT DO UPDATE — never touches the embedding |
| `update_embedding(db, name, vec)` | BLOB only — separate to avoid accidental overwrites |
| `get_all_users(db)` | Returns `list[User]` with BLOB → numpy deserialization |
| `get_user_by_name(db, name)` | Returns `User | None` |

---

### `core/models.py`

Lightweight dataclasses with no external dependencies.

```python
@dataclass
class User:
    id: int
    name: str
    user_tag: str
    preferred_address: str
    other_addresses: list[str]
    embedding: np.ndarray | None  # excluded from __repr__

    @property
    def is_guest(self) -> bool: ...
    @property
    def all_addresses(self) -> list[str]: ...  # preferred + others, deduplicated

@dataclass(frozen=True)
class SpeakerMatch:
    user: User
    score: float        # cosine 0.0–1.0
    method: str         # "match" | "fallback" | "guest"

GUEST_USER: User  # Singleton guest (id=0)
```

---

### `core/audio_gate.py`

Async lock to prevent the STT from recording while TTS is speaking.

```python
class AudioGate:
    def open(self) -> None         # event.set()
    def close(self) -> None        # event.clear()
    def is_open(self) -> bool
    async def wait_until_open(self) -> None

    @asynccontextmanager
    async def closed(self):
        # Closes, yields, reopens — guaranteed even on exception
```

**Module singleton:** `gate: AudioGate = AudioGate()` — imported directly by STT and TTS.

**Why `asyncio.Event` instead of `threading.Event`?**  
The entire pipeline is `async`. `asyncio.Event` avoids blocking the event loop:
`await event.wait()` yields control to other coroutines instead of blocking an OS thread.

---

### `core/wake_word.py`

```python
class WakeWordListener:
    async def listen(self) -> AsyncIterator[str]:
        # Async generator — yields the detected keyword string
        # Checks debounce (_last_fired) to avoid repeated triggers
```

Uses `livekit-wakeword` with the `models/Atlas.onnx` model (97 KB, committed in the repo).
Confidence threshold is configurable via `WAKE_WORD_THRESHOLD` (default: `0.5`).

---

### `core/stt.py`

**Recording — energy VAD:**

```
Chunks of 512 samples @ 16 kHz
         │
         ▼
  pre-buffer 5 chunks (keeps the start of speech)
         │
  RMS > threshold ? ──No──▶ silence_count++
         │ Yes                      │
  is_speaking = True     silence_count > MAX_SILENCE ?
         │                          │ Yes
  accumulate audio        stop + return raw PCM
         │
  MAX_DURATION reached ? ──Yes──▶ forced stop
```

**Transcription — `no_speech_prob` filter:**

```bash
whisper-cli --output-format json --no-timestamps -f audio.wav
```

The returned JSON contains `result[0].no_speech_prob`. If this value exceeds
`WHISPER_NO_SPEECH_THRESHOLD` (default: `0.6`), the transcription is discarded
(returns `""`). This fixes the documented *phantom transcription* bug where
whisper hallucinates "Thank you." or "Subtitles by..." on silence.

---

### `core/tts.py`

```python
class TTS:
    async def speak(self, text: str) -> None:
        clean = _strip_markdown(text)
        async with gate.closed():          # Close gate during synthesis
            await _run_say(clean, ...)
```

**16 Markdown patterns stripped:**
fenced code blocks, inline code, bold/italic (4 combinations), ATX headers `#`,
lists `-`/`*`/`1.`, blockquotes `>`, links `[text](url)`, wikilinks `[[...]]`,
HTML tags.

---

### `core/speaker_id.py`

**Identification in 3 steps:**

```
audio PCM  →  ECAPA-TDNN  →  L2-normalised vector (192 dims)
                                      │
                          cosine score vs each registered user
                                      │
         score > SPEAKER_ID_THRESHOLD (0.75) ? ──▶ method="match"
                │ No
         score > SPEAKER_FALLBACK_MIN_SCORE (0.30) ? ──▶ method="fallback" (most likely user)
                │ No
         ──▶ GUEST_USER, method="guest"
```

**Self-improving embeddings:**
- Each match saves a WAV in `user_voice_templates/<user_tag>/sample_N.wav`
- Pruning if `> MAX_VOICE_SAMPLES` (default: 50)
- `_recompute_embedding()`: averages embeddings of all saved WAVs

**Sleeping mode:** `recompute_all_embeddings()` re-averages all users from
samples accumulated since the last sleep cycle.

---

### `core/session.py`

```python
class SessionLog:
    path: Path         # Sessions/YYYY-MM-DD_HHMMSS.md
    turn_count: int
    speakers: set[str]

    def append_turn(self, speaker, user_text, tools_called, reply) -> None
    # Immediate write (crash-safe — no buffer)

    def close(self) -> None
    # Footer stats + [[Sessions]] backlink + registration in Sessions.md
```

**Session file format:**

```markdown
---
type: session
date: 2025-01-15
start: 10:23:45
---

# Session 2025-01-15 at 10:23

## Turn 1 · 10:23:52 · Roma
> What the user said

🔧 `datetime__get_datetime`

Atlas's reply.

---
*Session closed at 10:35:00 — 1 turn, 11 min, speaker: Roma*
[[Sessions]]
```

---

### `core/mcp_client.py`

**7 MCP servers registered in `TOOL_SERVERS`:**

| Name | Module |
|------|--------|
| `memory` | `atlas.tools.memory` |
| `datetime` | `atlas.tools.datetime_info` |
| `geoposition` | `atlas.tools.geoposition` |
| `weather` | `atlas.tools.weather` |
| `metrics` | `atlas.tools.metrics` |
| `wikipedia` | `atlas.tools.wikipedia` |
| `inbox` | `atlas.tools.inbox` |

**Tool prerequisites** (`TOOL_PREREQUISITES`):  
`memory_write`, `memory_patch`, `memory_link`, `memory_delete`, `memory_append`
all require `memory__memory_arbo` first. The orchestrator enforces this
order before dispatching.

**Per-tool timeout:**

```python
result = await asyncio.wait_for(
    _dispatch_tool(server, name, args),
    timeout=config.mcp_tool_timeout   # default 10.0s
)
```

**Parallel dispatch:**

```python
# Independent calls → asyncio.gather
results = await asyncio.gather(*tasks, return_exceptions=True)
```

---

### `core/health.py`

6 startup checks, invoked by `python -m atlas.core.orchestrator --check`:

| Check | Critical | What is verified |
|-------|----------|-----------------|
| Ollama | ✓ | HTTP GET `/` → 200 |
| whisper-cli | ✓ | `shutil.which("whisper-cli")` |
| Whisper model | ✓ | `WHISPER_MODEL_PATH` exists |
| Wake word models | ✓ | All paths in `WAKE_WORD_MODEL_PATHS` exist |
| Database | ✓ | `sqlite3.connect()` succeeds |
| Vault | ✗ | `ATLAS_VAULT_PATH` exists (warning if absent) |

Coloured terminal output: `✓` green / `⚠` yellow / `✗` red.  
`HealthCheckError` raised if at least one critical check fails.

---

### `core/orchestrator.py`

Main pipeline entry point. Modes:

| Flag | Behaviour |
|------|-----------|
| *(default)* | Wake word + full audio pipeline |
| `--text` | Stdin → bypass wake word/STT, reads one line, ideal for debugging |
| `--check` | Run health check then exit |
| `--nothink` | Disable Ollama thinking tokens (faster) |

---

## 3. Multi-round tool loop

```python
MAX_TOOL_ROUNDS = 10   # Anti-infinite-loop cap

for round_n in range(MAX_TOOL_ROUNDS):
    response = await ollama.chat(messages, tools=schemas)

    if response has tool_calls:
        # 1. Enforce prerequisites (memory_arbo if needed)
        # 2. Dispatch independent calls in parallel
        # 3. Add results to context
        continue                            # → next round

    text = response.message.content

    if text.endswith("[SUITE]"):            # Continuation sentinel
        tts.speak(text.removesuffix("[SUITE]"))
        messages.append(user="continue")
        continue

    break  # Final reply → TTS → session log
```

**`[SUITE]` sentinel:** allows the LLM to produce a long response in multiple
spoken parts without exceeding the context window. Capped at `_QUESTION_SENTINEL_CAP = 3`
successive sentinel iterations without tool calls to prevent loops.

---

## 4. MCP servers — `atlas/tools/`

Each tool is a **FastMCP stdio** server — spawned as a subprocess by
`MCPClient`. Communication via the MCP protocol over stdin/stdout.

| File | Exposed tools |
|------|--------------|
| `memory.py` | `memory_arbo`, `memory_read`, `memory_write`, `memory_patch_section`, `memory_link`, `memory_delete`, `memory_append`, `memory_search` |
| `datetime_info.py` | `get_datetime` |
| `geoposition.py` | `get_current_place` |
| `weather.py` | `get_local_weather`, `get_city_weather` |
| `metrics.py` | `get_mac_metrics` |
| `wikipedia.py` | `wikipedia_search`, `wikipedia_summary` |
| `inbox.py` | `inbox_list`, `inbox_read` |

**Vault traversal protection (`memory.py`):**

```python
def _note_path(vault: Path, name: str) -> Path:
    target = (vault / name).resolve()
    if not target.is_relative_to(vault.resolve()):
        raise ValueError("Path traversal not allowed")
    return target
```

---

## 5. Obsidian memory system

**Vault structure (`ATLAS_VAULT_PATH`):**

```
atlas_memory/
├── Sessions.md          ← Hub index of all sessions
├── Sessions/
│   ├── 2025-01-15_102345.md
│   └── 2025-01-16_090012.md
├── Topics/
│   ├── Python.md
│   └── Development.md
├── People/
│   └── Roma.md
└── Notes/
    └── ...
```

**Auto-tagging:** `memory_write` automatically injects `user_tag` into the
YAML frontmatter of every created note:

```yaml
---
type: memory
tags: [user_roma]
---
```

**Wikilinks:** Notes cross-reference each other via `[[NoteName]]`. The
`memory_link` tool adds a bidirectional link. `memory_arbo` returns the full
directory tree to contextualise write calls.

**Sessions hub (`Sessions.md`):**  
`SessionLog.close()` automatically appends `- [[Sessions/YYYY-MM-DD_HHMMSS]]` to
`Sessions.md`. `scripts/index_sessions.py` retroactively indexes orphan sessions.

---

## 6. Sleeping mode

**Trigger:** inactivity > `SLEEP_TIMEOUT` seconds (default: 180s).

**Actions:**

1. TTS announces going to sleep (`"Going to sleep."`)
2. `recompute_all_embeddings(config, db)` re-averages all user embeddings
   from voice samples accumulated since the last sleep cycle
3. `SessionLog.close()` — closes and archives the current session
4. New `SessionLog()` created — ready for the next conversation
5. Returns to listening for the wake word

Implemented in `_sleeping_mode_monitor()` as a background asyncio task —
does not block the main pipeline.

---

## 7. Config reference

| Field | `.env` variable | Default | Required |
|-------|----------------|---------|----------|
| `ollama_host` | `OLLAMA_HOST` | `http://localhost:11434` | No |
| `ollama_model` | `OLLAMA_MODEL` | `llama3.2` | No |
| `whisper_bin` | `WHISPER_CPP_BIN` | `whisper-cli` | No |
| `whisper_model` | `WHISPER_CPP_MODEL` | — | **Yes** |
| `whisper_language` | `WHISPER_CPP_LANGUAGE` | `en` | No |
| `whisper_no_speech_threshold` | `WHISPER_NO_SPEECH_THRESHOLD` | `0.6` | No |
| `wake_word_models` | `WAKE_WORD_MODELS` | `models/Atlas.onnx` | No |
| `wake_word_threshold` | `WAKE_WORD_THRESHOLD` | `0.5` | No |
| `wake_word_debounce` | `WAKE_WORD_DEBOUNCE` | `2.0` | No |
| `vault_path` | `ATLAS_VAULT_PATH` | `./atlas_memory` | No |
| `speaker_db_path` | `SPEAKER_DB_PATH` | `./atlas_users.db` | No |
| `voice_templates_dir` | `VOICE_TEMPLATES_DIR` | `./user_voice_templates` | No |
| `speaker_id_threshold` | `SPEAKER_ID_THRESHOLD` | `0.75` | No |
| `speaker_fallback_min_score` | `SPEAKER_FALLBACK_MIN_SCORE` | `0.30` | No |
| `mcp_tool_timeout` | `MCP_TOOL_TIMEOUT` | `10.0` | No |
| `sleep_timeout` | `SLEEP_TIMEOUT` | `180` | No |
| `tts_rate` | `TTS_RATE` | *(say default)* | No |
| `nothink` | `NOTHINK` | `false` | No |
| `think_depth` | `THINK_DEPTH` | `moderate` | No |

`ollama_options_dict()` returns only non-`None` options — only those fields
are sent to the Ollama API to avoid overriding the model's own defaults.
