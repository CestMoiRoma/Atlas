# Contributing to Atlas

Thank you for your interest in contributing to Atlas!

Atlas is licensed under the **GNU Affero General Public License v3.0**. By contributing,
you agree that your contributions will be licensed under the same terms.

---

## Quick Start

```bash
git clone https://github.com/CestMoiRoma/Atlas.git
cd Atlas
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in ATLAS_VAULT_PATH, WHISPER_MODEL_PATH, WAKE_WORD_MODEL_PATHS
```

Full setup instructions: [docs/wiki/02-contributor-guide.md](docs/wiki/02-contributor-guide.md)

---

## Code Standards

- **SPDX header** in every `.py` file: `# SPDX-License-Identifier: AGPL-3.0-or-later`
- **Type annotations** on all public functions (Python 3.10+ `X | Y` syntax)
- **Docstrings** in Google style
- **Linter**: `ruff check atlas/ tests/ scripts/`
- **No blocking I/O** in async code — use `asyncio.to_thread()` where needed

---

## Running Tests

```bash
# Unit tests (no external deps)
pytest tests/unit/ -v

# Full suite
pytest --cov=atlas --cov-report=term-missing
```

---

## Submitting Changes

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run `ruff check` and `pytest tests/unit/ -v`
5. Commit using [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(tools): add calendar MCP server
   fix(stt): handle empty JSON output from whisper-cli
   test(config): add edge cases for float validation
   ```
6. Open a Pull Request against `DEV_AtlasV0.1`

---

## Adding an MCP Tool

See [docs/wiki/02-contributor-guide.md § 5](docs/wiki/02-contributor-guide.md#5-ajouter-un-outil-mcp)
for the full 5-step guide (server template → registration → prerequisites → entry point → test).

---

## Reporting Issues

Please include:
- macOS version
- Python version (`python3 --version`)
- Ollama model and version (`ollama --version`)
- Output of `python -m atlas.core.orchestrator --check`
- Relevant log output

---

## License

Atlas is AGPLv3. Contributions must be compatible.  
See [NOTICE](NOTICE) for third-party attributions.
