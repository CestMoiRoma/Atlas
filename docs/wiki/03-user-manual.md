# User Manual

Atlas is a local-first AI voice assistant for macOS — fully offline after initial
configuration. This guide covers installation, configuration, and everyday use.

---

## Table of contents

1. [System requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [`.env` configuration](#3-env-configuration)
4. [Registering a user](#4-registering-a-user)
5. [Starting Atlas](#5-starting-atlas)
6. [Command-line options](#6-command-line-options)
7. [Voice commands and memory](#7-voice-commands-and-memory)
8. [Utility scripts](#8-utility-scripts)
9. [FAQ](#9-faq)

---

## 1. System requirements

| Component | Required | Notes |
|-----------|---------|-------|
| macOS | 13 Ventura or later | Required for `say`, CoreLocation, PortAudio |
| Python | 3.10+ | Available via Homebrew (`brew install python@3.12`) |
| Ollama | Latest | [ollama.ai](https://ollama.ai) — runs in background |
| whisper.cpp | `whisper-cli` in `$PATH` | See §2.4 |
| PortAudio | — | `brew install portaudio` — required by `sounddevice` |
| Disk | ~1.5 GB minimum | Whisper models (~800 MB) + SpeechBrain (~80 MB) |
| RAM | 8 GB recommended | For Ollama + Whisper model in memory simultaneously |

---

## 2. Installation

### 2.1 Install system dependencies

```bash
# Homebrew required — https://brew.sh
brew install python@3.12 portaudio
```

### 2.2 Install Ollama and download an LLM

```bash
# Install Ollama (macOS installer at ollama.ai)
# Then start the server:
ollama serve &

# Download a language model (e.g. llama3.2 ~2 GB)
ollama pull llama3.2
```

### 2.3 Install Atlas

```bash
git clone https://github.com/CestMoiRoma/Atlas.git
cd Atlas
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2.4 Install whisper.cpp

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
cmake -B build && cmake --build build --config Release -j
# Copy the binary to $PATH:
cp build/bin/whisper-cli /usr/local/bin/whisper-cli
```

Download a Whisper GGML model (e.g. `base` for a good speed/quality balance):

```bash
# From the whisper.cpp root:
bash models/download-ggml-model.sh base
# → File: models/ggml-base.bin (~142 MB)
```

### 2.5 Download Atlas models

```bash
# From the Atlas repo root, with .venv activated:
python scripts/download_models.py
```

Automatically downloads SpeechBrain ECAPA-TDNN (~80 MB) into
`models/spkrec-ecapa-voxceleb/`.

---

## 3. `.env` configuration

```bash
cp .env.example .env
```

Edit `.env` — **required** variables:

```bash
# Obsidian vault — Markdown directory where Atlas stores its memory
ATLAS_VAULT_PATH=/Users/you/Documents/atlas_memory

# Whisper GGML model (absolute path recommended)
WHISPER_CPP_MODEL=/Users/you/whisper.cpp/models/ggml-base.bin

# Wake word model (already in the repo)
WAKE_WORD_MODELS=models/Atlas.onnx
```

**Optional** variables commonly customised:

```bash
# Ollama model to use
OLLAMA_MODEL=llama3.2           # or llama3.1, mistral, gemma2:9b, etc.

# Transcription language
WHISPER_CPP_LANGUAGE=en         # en, fr, es, de, ...

# Phantom transcription filter threshold (0.0 to 1.0)
# Higher = less filtering. 0.6 is a good default.
WHISPER_NO_SPEECH_THRESHOLD=0.6

# MCP tool timeout in seconds
MCP_TOOL_TIMEOUT=10.0

# Inactivity duration before sleep mode (seconds)
SLEEP_TIMEOUT=180

# Speech synthesis speed (words/minute, empty = system default)
TTS_RATE=
```

---

## 4. Registering a user

Atlas identifies speakers by voice (ECAPA-TDNN). You need to register at least
one user so Atlas knows how to address you.

```bash
python scripts/register_user.py \
  --name "Roma" \
  --age 28 \
  --gender M \
  --profession "Developer" \
  --preferred-address "chief"
```

The script records **5 voice clips of 4 seconds** each (with pauses between).
Speak naturally during recording — no specific content required.

**Available options:**

| Option | Description |
|--------|-------------|
| `--name` | Full name (required) |
| `--age` | Age (optional) |
| `--gender` | Gender: M, F, or free text |
| `--profession` | Occupation (optional) |
| `--preferred-address` | How Atlas should address you |
| `--update` | Update profile without re-recording voice |
| `--re-record` | Re-record voice samples |

**Edit an existing profile:**

```bash
python scripts/edit_user.py --name "Roma" --profession "CTO"

# List all users:
python scripts/edit_user.py --list
```

---

## 5. Starting Atlas

### Prerequisites before starting

```bash
# 1. Ollama must be running
ollama serve &

# 2. Activate the Python environment
source .venv/bin/activate
```

### Standard start

```bash
python -m atlas.core.orchestrator
```

or via the installed entry point:

```bash
atlas
```

Atlas displays the health check result, then waits for the wake word.

**Say "Atlas"** (or the configured wake word) to start an interaction.

### Clean shutdown

`Ctrl+C` — Atlas closes the current session and exits cleanly.

---

## 6. Command-line options

### `--check` — Health check only

```bash
python -m atlas.core.orchestrator --check
```

Checks all dependencies and exits. Useful for diagnosing problems
without starting the full pipeline.

### `--text` — Text mode (debug / scripting)

```bash
# Single interaction
echo "What time is it?" | python -m atlas.core.orchestrator --text

# Interactive session
python -m atlas.core.orchestrator --text
# → Type questions, one per line, Ctrl+D to quit
```

Completely bypasses wake word and STT — ideal for testing prompts or debugging
MCP tools without a microphone.

### `--nothink` — Disable thinking tokens

```bash
python -m atlas.core.orchestrator --nothink
```

Disables Ollama's `<think>` tokens (if supported by the model). Reduces
latency at the cost of less structured reasoning.

---

## 7. Voice commands and memory

### Natural interaction

Atlas is designed for natural conversation. No special syntax — just speak:

> *"Atlas, what's the weather in London?"*  
> *"Atlas, note that the deployment is scheduled for Friday."*  
> *"Atlas, what do you know about Python?"*

### Available tools

| Tool | Typical triggers |
|------|----------------|
| `datetime` | "what time", "what day", "what's the date" |
| `geoposition` | "where am I", "my location" |
| `weather` | "weather", "what's the weather", "temperature" |
| `metrics` | "CPU", "RAM", "memory available", "system stats" |
| `wikipedia` | "what is", "who is", "explain", "tell me about" |
| `memory` | "note that", "remember", "what do you know about" |
| `inbox` | "read my inbox", "what's in my files" |

### Persistent memory

Atlas stores its notes in the Obsidian vault (`ATLAS_VAULT_PATH`). Sessions
are indexed in `Sessions.md`. Notes created via memory tools appear in
`Topics/`, `People/`, etc.

**Atlas can retrieve information mentioned in previous conversations** as long
as the corresponding note was created in the vault.

---

## 8. Utility scripts

### Download models

```bash
python scripts/download_models.py
```

### Index orphan sessions

If `Sessions/*.md` files don't appear in `Sessions.md`:

```bash
python scripts/index_sessions.py
# Or specify a vault:
python scripts/index_sessions.py --vault /path/to/vault
```

### Merge duplicate topics

After prolonged use, similar topics may accumulate
(e.g. `Python.md` and `Python_Programming.md`):

```bash
# Interactive review
python scripts/unify_topics.py

# Auto-merge high-similarity pairs (>= 80%)
python scripts/unify_topics.py --threshold 0.8 --auto
```

### Embed files into the vault

```bash
# Simple embedding (4096-char chunks)
python scripts/embed_memory.py /path/to/document.md

# Large-context embedding (100K-char windows, 50% overlap)
python scripts/embed_deep.py /path/to/large_document.md
```

---

## 9. FAQ

### Ollama not responding / "Connection refused"

```bash
# Check that Ollama is running
ollama list

# Start manually if needed
ollama serve &

# Run the health check
python -m atlas.core.orchestrator --check
```

### `whisper-cli` not found

```bash
# Check that whisper-cli is in $PATH
which whisper-cli

# If absent, recompile (see §2.4) or check:
ls /usr/local/bin/whisper-cli
```

If `whisper-cli` is in a non-standard directory, set in `.env`:

```bash
WHISPER_CPP_BIN=/path/to/whisper-cli
```

### Wake word not detected

1. Check that the microphone is authorised for Terminal in
   `System Settings → Privacy → Microphone`
2. Verify that `models/Atlas.onnx` exists: `ls -la models/Atlas.onnx`
3. Lower the detection threshold in `.env`: `WAKE_WORD_THRESHOLD=0.3`
4. Test in text mode to isolate the issue: `echo "test" | python -m atlas.core.orchestrator --text`

### Phantom transcriptions ("Thank you.", "Subtitles by...")

This is a known whisper.cpp bug — the model hallucinates on silence.

Atlas filters automatically via `no_speech_prob`. If phantoms persist,
lower the threshold (= filter more aggressively):

```bash
# In .env:
WHISPER_NO_SPEECH_THRESHOLD=0.4   # default: 0.6
```

Note: a threshold too low may discard legitimate low-confidence transcriptions
(distant voice, background noise).

### Atlas doesn't recognise my voice

1. Check that the user is registered:
   ```bash
   python scripts/edit_user.py --list
   ```

2. Re-record voice samples in a quiet environment:
   ```bash
   python scripts/register_user.py --name "Roma" --re-record
   ```

3. Adjust identification thresholds in `.env`:
   ```bash
   SPEAKER_ID_THRESHOLD=0.70     # Lower slightly (default: 0.75)
   SPEAKER_FALLBACK_MIN_SCORE=0.25  # Lower slightly (default: 0.30)
   ```

### How to update Atlas?

```bash
git pull origin DEV_AtlasV0.1
pip install -e .
python -m atlas.core.orchestrator --check
```

If the database schema has changed, delete and recreate:

```bash
rm atlas_users.db
# Re-register users
python scripts/register_user.py --name "Roma" ...
```

### Session not saved to the vault

1. Check that `ATLAS_VAULT_PATH` points to an existing directory with write permissions
2. Verify that `Sessions/` is created on first start
3. Use `scripts/index_sessions.py` to retroactively index existing files

### Atlas is slow to respond

Options to reduce latency:

- Use a lighter Ollama model: `OLLAMA_MODEL=llama3.2:1b`
- Disable thinking tokens: `python -m atlas.core.orchestrator --nothink`
- Reduce LLM context: `OLLAMA_NUM_CTX=2048`
- Use a smaller Whisper model: `ggml-tiny.bin` instead of `ggml-base.bin`
