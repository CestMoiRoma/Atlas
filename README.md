# Atlas

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![macOS](https://img.shields.io/badge/platform-macOS%2013%2B-lightgrey.svg)](https://www.apple.com/macos/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> Local-first, modular AI voice assistant. No cloud. No subscriptions.  
> Your personal assistant — who knows you, and remembers everything.

---

## Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | [Ollama](https://ollama.ai) — local inference, any compatible model |
| **STT** | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — offline transcription |
| **TTS** | macOS `say` — zero-latency native synthesis |
| **Wake word** | [livekit-wakeword](https://github.com/livekit/pipecat) — Atlas.onnx (97 KB, custom model) |
| **Speaker ID** | [SpeechBrain](https://speechbrain.github.io) ECAPA-TDNN — cosine similarity |
| **Tools** | [FastMCP](https://github.com/jlowin/fastmcp) stdio servers — 7 built-in tools |
| **Memory** | [Obsidian](https://obsidian.md)-compatible Markdown vault — persistent knowledge graph |
| **Config** | Frozen `Config` dataclass — typed, validated, injectable |

---

## How It Works

```
 Mic  ──▶  WakeWord  ──▶  STT (VAD + whisper)  ──▶  Speaker ID
                                                          │
                                              ┌───────────▼────────────┐
                                              │       Orchestrator      │
                                              │  system prompt          │
                                              │  Ollama LLM  ◀──────┐  │
                                              │       │             │  │
                                              │  tool calls?        │  │
                                              │       │ yes         │  │
                                              │       ▼             │  │
                                              │   MCPClient         │  │
                                              │  (parallel,timeout) │  │
                                              │       │             │  │
                                              │  results ───────────┘  │
                                              │       │ final reply     │
                                              └───────┼────────────────┘
                                                      │
                                               TTS (say)  +  SessionLog
```

---

## Quick Start

**1. Install system dependencies**

```bash
brew install python@3.12 portaudio
# Install Ollama from https://ollama.ai, then:
ollama pull llama3.2
```

**2. Install whisper.cpp**

```bash
git clone https://github.com/ggerganov/whisper.cpp && cd whisper.cpp
cmake -B build && cmake --build build --config Release -j
cp build/bin/whisper-cli /usr/local/bin/
bash models/download-ggml-model.sh base
```

**3. Install Atlas**

```bash
git clone https://github.com/CestMoiRoma/Atlas.git && cd Atlas
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/download_models.py
```

**4. Configure**

```bash
cp .env.example .env
# Set ATLAS_VAULT_PATH, WHISPER_MODEL_PATH, and WAKE_WORD_MODEL_PATHS
```

**5. Register yourself and start**

```bash
python scripts/register_user.py --name "You" --preferred-address "chef"
python -m atlas.core.orchestrator
```

Say **"Atlas"** to wake it up.

---

## Features

### 🎤 Voice Pipeline
- Custom wakeword detection (`Atlas.onnx`, 97 KB ONNX model)
- Energy-based VAD with pre-speech buffering
- `no_speech_prob` filter — eliminates whisper phantom transcriptions
- Anti-feedback audio gate (async-first, `asyncio.Event`)

### 👤 Speaker Identification
- ECAPA-TDNN embeddings — identifies who is speaking
- Three-tier matching: exact match → soft fallback → guest
- Self-improving: voice samples accumulate and embeddings re-average in sleeping mode

### 🧠 Persistent Memory
- Obsidian-compatible Markdown vault
- Auto-tagged notes with user attribution in YAML frontmatter
- Session hub (`Sessions.md`) — every conversation indexed
- Wikilinks between notes — knowledge graph grows over time

### 🔧 Modular Tools (MCP)
- `datetime` — current date and time in French
- `geoposition` — location via ip-api.com
- `weather` — Open-Meteo + geocoding, WMO codes in French
- `metrics` — CPU, RAM, disk via psutil
- `wikipedia` — French Wikipedia REST API
- `memory` — vault read/write/patch/link/delete/append
- `inbox` — read `.txt`, `.md`, `.json`, `.csv`, `.drawio` files

**Per-tool timeout** (configurable, default 10s) — a slow network call never freezes the pipeline.  
**Parallel dispatch** — independent tool calls run concurrently via `asyncio.gather`.

### 🤖 LLM Controls
- Multi-round tool loop with prerequisite enforcement (e.g. `memory_arbo` before `memory_write`)
- `[SUITE]` continuation sentinel — long answers delivered in voice-natural chunks
- `--nothink` flag — disables Ollama thinking tokens for lower latency
- `--text` mode — bypass audio entirely, useful for scripting and debugging

---

## Adding a Tool

```python
# atlas/tools/my_tool.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my_tool")

@mcp.tool()
async def my_tool_action(param: str) -> str:
    """What this tool does."""
    return f"Result: {param}"

if __name__ == "__main__":
    mcp.run()
```

Then add `"my_tool": "atlas.tools.my_tool"` to `TOOL_SERVERS` in `atlas/core/mcp_client.py`.  
See [docs/wiki/02-contributor-guide.md](docs/wiki/02-contributor-guide.md) for the full 5-step guide.

---

## Known Issues

**Phantom transcriptions** — whisper.cpp hallucinates text (e.g. *"Merci."*, *"Sous-titres réalisés par..."*) on near-silence inputs. Atlas filters these via the `no_speech_prob` field in whisper's JSON output. Tune `WHISPER_NO_SPEECH_THRESHOLD` in `.env` if needed (default: `0.6`).

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/wiki/01-internal-architecture.md](docs/wiki/01-internal-architecture.md) | Pipeline internals, module reference, Config fields |
| [docs/wiki/02-contributor-guide.md](docs/wiki/02-contributor-guide.md) | Dev setup, code standards, adding tools/models |
| [docs/wiki/03-user-manual.md](docs/wiki/03-user-manual.md) | Installation, configuration, FAQ |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.  
Atlas uses [Conventional Commits](https://www.conventionalcommits.org/) and AGPLv3.

---

## License

Copyright © 2025 CestMoiRoma  
Licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE).

Third-party attributions: see [NOTICE](NOTICE).
