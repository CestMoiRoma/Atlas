# Contributor Guide

Welcome to Atlas! This guide covers everything you need to contribute to the project:
development environment setup, code standards, and procedures for adding new tools or models.

> For internal architecture, see [01-internal-architecture.md](01-internal-architecture.md).  
> For user installation, see [03-user-manual.md](03-user-manual.md).

---

## Table of contents

1. [Development prerequisites](#1-development-prerequisites)
2. [Environment setup](#2-environment-setup)
3. [Code standards](#3-code-standards)
4. [Running tests](#4-running-tests)
5. [Adding an MCP tool](#5-adding-an-mcp-tool)
6. [Adding a downloadable model](#6-adding-a-downloadable-model)
7. [Contribution process](#7-contribution-process)
8. [Conventional Commits](#8-conventional-commits)

---

## 1. Development prerequisites

| Dependency | Minimum version | Notes |
|------------|----------------|-------|
| Python | 3.10 | Type unions `X \| Y`, `match/case` |
| macOS | 13 Ventura | `say`, CoreLocation, PortAudio |
| Ollama | latest | `ollama serve` must run in background |
| Git | 2.x | GPG signing recommended |

Dev Python tools installed via the `[dev]` group:

```
pytest          pytest-asyncio   ruff
mypy            httpx            python-dotenv
```

---

## 2. Environment setup

### 2.1 Clone and install

```bash
git clone https://github.com/CestMoiRoma/Atlas.git
cd Atlas
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2.2 Configure the environment

```bash
cp .env.example .env
# Edit .env — at minimum:
#   ATLAS_VAULT_PATH=/path/to/vault
#   WHISPER_CPP_MODEL=/path/to/ggml-base.bin
#   WAKE_WORD_MODELS=models/Atlas.onnx
```

### 2.3 Download heavy models

```bash
python scripts/download_models.py
```

This downloads SpeechBrain ECAPA-TDNN (~80 MB) and prints instructions for
Whisper GGML (~800 MB).

`models/Atlas.onnx` (97 KB, wake word) is already in the repo — no download needed.

### 2.4 Verify the installation

```bash
python -m atlas.core.orchestrator --check
```

Expected output (all green):

```
  ✓  Ollama            http://localhost:11434 — llama3.2 available
  ✓  whisper-cli       /usr/local/bin/whisper-cli
  ✓  Whisper model     /path/to/ggml-base.bin (142 MB)
  ✓  Wake word model   models/Atlas.onnx
  ✓  Database          atlas_users.db
  ✓  Vault             /path/to/vault
```

---

## 3. Code standards

### 3.1 SPDX header (mandatory)

Every `.py` file must start with:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
```

This line is checked by `ruff` in CI.

### 3.2 Strict typing

- All public functions are typed (parameters + return value)
- Use `X | Y` (PEP 604, Python 3.10+) rather than `Optional[X]` or `Union[X, Y]`
- Avoid `Any` except in documented exceptional cases

```python
# ✓ Correct
def greet(name: str, age: int | None = None) -> str: ...

# ✗ Avoid
def greet(name, age=None): ...
```

### 3.3 Docstrings (Google style)

```python
def embed(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Encode audio PCM to a L2-normalised embedding vector.

    Args:
        audio: 1-D float32 array of PCM samples.
        sample_rate: Sample rate in Hz (typically 16000).

    Returns:
        1-D float32 unit vector of shape (192,).

    Raises:
        RuntimeError: If the ECAPA encoder is not initialised.
    """
```

### 3.4 Ruff (linting + formatting)

```bash
# Check
ruff check atlas/ tests/ scripts/

# Format
ruff format atlas/ tests/ scripts/
```

Configuration is in `pyproject.toml`:

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "C4", "ANN"]
```

- **E/F** : pycodestyle/pyflakes errors and warnings
- **I** : import sorting (isort-compatible)
- **UP** : pyupgrade — modernise Python code
- **B** : bugbear — problematic patterns
- **C4** : idiomatic comprehensions
- **ANN** : missing annotations

### 3.5 Async conventions

- All I/O code is `async` — never block the event loop
- Use `asyncio.to_thread()` or `loop.run_in_executor()` for blocking operations
  (e.g. heavy file reads, numpy, sqlite when needed)
- Async tests use `@pytest.mark.asyncio` (configured as `auto` in `pyproject.toml`)

---

## 4. Running tests

### Unit tests (fast, no external dependencies)

```bash
pytest tests/unit/ -v
```

Unit tests mock all I/O dependencies (Ollama, sounddevice, whisper-cli, SQLite).
They must run without network access or audio hardware.

### Integration tests (require Ollama + models)

```bash
pytest tests/integration/ -v
```

### Full suite with coverage

```bash
pytest --cov=atlas --cov-report=term-missing
```

### Linter only (CI)

```bash
ruff check atlas/ tests/ scripts/
```

---

## 5. Adding an MCP tool

### Step 1 — Create the FastMCP server

```python
# atlas/tools/my_tool.py
# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/my_tool.py — Example MCP tool."""

from __future__ import annotations
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my_tool")


@mcp.tool()
async def my_tool_action(param: str) -> str:
    """Describe what this tool does.

    Args:
        param: Parameter description.

    Returns:
        Result as a string.
    """
    return f"Result for: {param}"


if __name__ == "__main__":
    mcp.run()
```

### Step 2 — Register in `TOOL_SERVERS`

```python
# atlas/core/mcp_client.py
TOOL_SERVERS: dict[str, str] = {
    # ... existing servers ...
    "my_tool": "atlas.tools.my_tool",   # ← Add here
}
```

### Step 3 — Declare prerequisites if needed

If your tool depends on another (e.g. vault read before write):

```python
TOOL_PREREQUISITES: dict[str, list[str]] = {
    # ... existing prerequisites ...
    "my_tool__my_tool_action": ["memory__memory_arbo"],   # if needed
}
```

### Step 4 — Add entry point in `pyproject.toml`

If the tool needs to be invoked directly (e.g. geoposition service):

```toml
[project.scripts]
atlas-my-tool = "atlas.tools.my_tool:mcp.run"
```

### Step 5 — Write a test

```python
# tests/unit/test_my_tool.py
import pytest
from atlas.tools.my_tool import my_tool_action

@pytest.mark.asyncio
async def test_my_tool_action():
    result = await my_tool_action("test")
    assert "test" in result
```

---

## 6. Adding a downloadable model

If your code requires a heavy model file (> 10 MB), it must be managed by
`scripts/download_models.py` and **excluded from git** (`.gitignore`).

### Step 1 — Define the `ModelSpec`

```python
# scripts/download_models.py

NEW_MODEL = ModelSpec(
    name="My Model v2",
    url="https://huggingface.co/org/model/resolve/main/model.bin",
    dest=Path("models/my_model_v2.bin"),
    sha256="abc123def456...",   # sha256sum of the expected file
    size_mb=250.0,
)
```

### Step 2 — Get the SHA-256

```bash
curl -L <url> -o /tmp/my_model.bin
sha256sum /tmp/my_model.bin
```

### Step 3 — Add to `.gitignore`

```gitignore
models/my_model_v2.bin
```

### Step 4 — Document in `.env.example`

```bash
# Path to My Model v2 (downloaded via scripts/download_models.py)
MY_MODEL_PATH=models/my_model_v2.bin
```

---

## 7. Contribution process

### Branches

```
main          ← Stable code, tagged
DEV_AtlasVx.x ← Active development
feature/*     ← New features
fix/*         ← Bug fixes
```

### Standard workflow

```bash
# 1. Create a branch
git checkout -b feature/my-tool

# 2. Develop + tests
pytest tests/unit/ -v
ruff check atlas/ tests/

# 3. Commit (GPG required for maintainers)
git add -p   # Hunk-by-hunk review
git commit -S -m "feat(tools): add my_tool server"

# 4. Open a Pull Request
gh pr create --base DEV_AtlasV0.1
```

### PR review

- At least one reviewer before merge
- CI must pass (ruff + pytest)
- Commits must follow Conventional Commits (see §8)

---

## 8. Conventional Commits

Format: `type(scope): description`

| Type | Usage |
|------|-------|
| `feat` | New feature |
| `fix` | Bug fix |
| `test` | Adding/modifying tests |
| `docs` | Documentation only |
| `chore` | Tooling, CI, dependencies |
| `refactor` | Rewrite without behaviour change |
| `perf` | Performance improvement |

**Common scopes:** `core`, `tools`, `db`, `config`, `scripts`, `tests`, `ci`

**Examples:**

```
feat(core): add no_speech_prob filter to STT transcription
fix(tools): handle empty Wikipedia response gracefully
test(config): add validation edge cases for float fields
chore: bump httpx to 0.28
docs(wiki): update architecture diagram for parallel dispatch
```

**Commit body** (for significant changes):

```
feat(core): add parallel tool dispatch via asyncio.gather

Independent MCP tool calls (no prerequisites) are now dispatched
concurrently. Dependent calls (memory_write after memory_arbo) remain
sequential as enforced by TOOL_PREREQUISITES.

Reduces average tool round latency from ~3s to ~1s for multi-tool turns.
```
