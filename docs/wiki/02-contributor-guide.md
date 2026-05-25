# Guide de Contribution

Bienvenue dans Atlas ! Ce guide couvre tout ce dont tu as besoin pour contribuer au projet :
setup de l'environnement de développement, normes de code, et procédures pour ajouter de
nouveaux outils ou modèles.

> Pour l'architecture interne, voir [01-internal-architecture.md](01-internal-architecture.md).  
> Pour l'installation utilisateur, voir [03-user-manual.md](03-user-manual.md).

---

## Table des matières

1. [Prérequis de développement](#1-prérequis-de-développement)
2. [Setup de l'environnement](#2-setup-de-lenvironnement)
3. [Normes de code](#3-normes-de-code)
4. [Lancer les tests](#4-lancer-les-tests)
5. [Ajouter un outil MCP](#5-ajouter-un-outil-mcp)
6. [Ajouter un modèle téléchargeable](#6-ajouter-un-modèle-téléchargeable)
7. [Processus de contribution](#7-processus-de-contribution)
8. [Conventional Commits](#8-conventional-commits)

---

## 1. Prérequis de développement

| Dépendance | Version minimale | Notes |
|-----------|-----------------|-------|
| Python | 3.10 | Type unions `X \| Y`, `match/case` |
| macOS | 13 Ventura | `say`, CoreLocation, PortAudio |
| Ollama | latest | `ollama serve` doit tourner en fond |
| Git | 2.x | GPG signing recommandé |

Outils Python de dev installés via le groupe `[dev]` :

```
pytest          pytest-asyncio   ruff
mypy            httpx            python-dotenv
```

---

## 2. Setup de l'environnement

### 2.1 Cloner et installer

```bash
git clone https://github.com/CestMoiRoma/Atlas.git
cd Atlas
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2.2 Configurer l'environnement

```bash
cp .env.example .env
# Éditer .env — au minimum :
#   ATLAS_VAULT_PATH=/chemin/vers/vault
#   WHISPER_MODEL_PATH=/chemin/vers/ggml-base.bin
#   WAKE_WORD_MODEL_PATHS=models/Atlas.onnx
```

### 2.3 Télécharger les modèles lourds

```bash
python scripts/download_models.py
```

Cela télécharge SpeechBrain ECAPA-TDNN (~80 MB) et affiche les instructions pour
Whisper GGML (~800 MB).

`models/Atlas.onnx` (97 KB, wakeword) est déjà dans le repo — aucun téléchargement
nécessaire.

### 2.4 Vérifier l'installation

```bash
python -m atlas.core.orchestrator --check
```

Sortie attendue (tout en vert) :

```
  ✓  Ollama            http://localhost:11434 — llama3.2 disponible
  ✓  whisper-cli       /usr/local/bin/whisper-cli
  ✓  Whisper model     /path/to/ggml-base.bin (142 MB)
  ✓  Wake word model   models/Atlas.onnx
  ✓  Database          atlas_users.db
  ✓  Vault             /path/to/vault
```

---

## 3. Normes de code

### 3.1 En-tête SPDX (obligatoire)

Chaque fichier `.py` doit commencer par :

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
```

Cette ligne est vérifiée par `ruff` en CI.

### 3.2 Typing strict

- Toutes les fonctions publiques sont typées (paramètres + retour)
- Utiliser `X | Y` (PEP 604, Python 3.10+) plutôt que `Optional[X]` ou `Union[X, Y]`
- Éviter `Any` sauf cas exceptionnel documenté

```python
# ✓ Correct
def greet(name: str, age: int | None = None) -> str: ...

# ✗ Éviter
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
# Vérifier
ruff check atlas/ tests/ scripts/

# Formater
ruff format atlas/ tests/ scripts/
```

La configuration est dans `pyproject.toml` :

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "C4", "ANN"]
```

- **E/F** : erreurs et avertissements pycodestyle/pyflakes
- **I** : import sorting (isort-compatible)
- **UP** : pyupgrade — moderniser le code Python
- **B** : bugbear — patterns problématiques
- **C4** : comprehensions idiomatiques
- **ANN** : annotations manquantes

### 3.5 Async conventions

- Tout code I/O est `async` — ne jamais bloquer l'event loop
- Utiliser `asyncio.to_thread()` ou `loop.run_in_executor()` pour les opérations
  bloquantes (ex: lecture fichier lourde, numpy, sqlite si nécessaire)
- Les tests async utilisent `@pytest.mark.asyncio` (configuré en `auto` dans `pyproject.toml`)

---

## 4. Lancer les tests

### Tests unitaires (rapides, pas de dépendances externes)

```bash
pytest tests/unit/ -v
```

Les tests unitaires mockent toutes les dépendances I/O (Ollama, sounddevice,
whisper-cli, SQLite). Ils doivent tourner sans réseau ni hardware audio.

### Tests d'intégration (nécessitent Ollama + modèles)

```bash
pytest tests/integration/ -v
```

### Suite complète avec couverture

```bash
pytest --cov=atlas --cov-report=term-missing
```

### Linter uniquement (CI)

```bash
ruff check atlas/ tests/ scripts/
```

---

## 5. Ajouter un outil MCP

### Étape 1 — Créer le serveur FastMCP

```python
# atlas/tools/mon_outil.py
# SPDX-License-Identifier: AGPL-3.0-or-later
"""atlas/tools/mon_outil.py — Exemple d'outil MCP."""

from __future__ import annotations
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mon_outil")


@mcp.tool()
async def mon_outil_action(param: str) -> str:
    """Décris ce que fait cet outil.

    Args:
        param: Description du paramètre.

    Returns:
        Résultat sous forme de chaîne.
    """
    return f"Résultat pour : {param}"


if __name__ == "__main__":
    mcp.run()
```

### Étape 2 — Enregistrer dans `TOOL_SERVERS`

```python
# atlas/core/mcp_client.py
TOOL_SERVERS: dict[str, str] = {
    # ... serveurs existants ...
    "mon_outil": "atlas.tools.mon_outil",   # ← Ajouter ici
}
```

### Étape 3 — Déclarer les prérequis si nécessaire

Si ton outil dépend d'un autre (ex: lecture vault avant écriture) :

```python
TOOL_PREREQUISITES: dict[str, list[str]] = {
    # ... prérequis existants ...
    "mon_outil__mon_outil_action": ["memory__memory_arbo"],   # si besoin
}
```

### Étape 4 — Ajouter l'entry point dans `pyproject.toml`

Si l'outil a besoin d'être invoqué directement (ex: service géoposition) :

```toml
[project.scripts]
atlas-mon-outil = "atlas.tools.mon_outil:mcp.run"
```

### Étape 5 — Écrire un test

```python
# tests/unit/test_mon_outil.py
import pytest
from atlas.tools.mon_outil import mon_outil_action

@pytest.mark.asyncio
async def test_mon_outil_action():
    result = await mon_outil_action("test")
    assert "test" in result
```

---

## 6. Ajouter un modèle téléchargeable

Si ton code nécessite un fichier modèle lourd (> 10 MB), il doit être géré par
`scripts/download_models.py` et **exclu de git** (`.gitignore`).

### Étape 1 — Définir le `ModelSpec`

```python
# scripts/download_models.py
from atlas.config import Config

NEW_MODEL = ModelSpec(
    name="Mon Modèle v2",
    url="https://huggingface.co/org/model/resolve/main/model.bin",
    dest=Path("models/mon_modele_v2.bin"),
    sha256="abc123def456...",   # sha256sum du fichier attendu
    size_mb=250.0,
)
```

### Étape 2 — Obtenir le SHA-256

```bash
curl -L <url> -o /tmp/mon_modele.bin
sha256sum /tmp/mon_modele.bin
```

### Étape 3 — Ajouter au `.gitignore`

```gitignore
models/mon_modele_v2.bin
```

### Étape 4 — Documenter dans `.env.example`

```bash
# Chemin vers Mon Modèle v2 (téléchargé via scripts/download_models.py)
MON_MODELE_PATH=models/mon_modele_v2.bin
```

---

## 7. Processus de contribution

### Branches

```
main          ← Code stable, taggué
DEV_AtlasVx.x ← Développement actif
feature/*     ← Nouvelles fonctionnalités
fix/*         ← Corrections de bugs
```

### Workflow standard

```bash
# 1. Créer une branche
git checkout -b feature/mon-outil

# 2. Développer + tests
pytest tests/unit/ -v
ruff check atlas/ tests/

# 3. Commiter (GPG requis pour les mainteneurs)
git add -p   # Revue hunk par hunk
git commit -S -m "feat(tools): add mon_outil server"

# 4. Ouvrir une Pull Request
gh pr create --base DEV_AtlasV0.1
```

### Revue de PR

- Au moins un reviewer avant merge
- CI doit passer (ruff + pytest)
- Les commits doivent suivre les Conventional Commits (voir §8)
- Pas de `Co-Authored-By: *` automatique

---

## 8. Conventional Commits

Format : `type(scope): description`

| Type | Usage |
|------|-------|
| `feat` | Nouvelle fonctionnalité |
| `fix` | Correction de bug |
| `test` | Ajout/modification de tests |
| `docs` | Documentation uniquement |
| `chore` | Outillage, CI, dépendances |
| `refactor` | Réécriture sans changement de comportement |
| `perf` | Amélioration de performance |

**Scopes courants :** `core`, `tools`, `db`, `config`, `scripts`, `tests`, `ci`

**Exemples :**

```
feat(core): add no_speech_prob filter to STT transcription
fix(tools): handle empty Wikipedia response gracefully
test(config): add validation edge cases for float fields
chore: bump httpx to 0.28
docs(wiki): update architecture diagram for parallel dispatch
```

**Corps de commit** (pour les changements significatifs) :

```
feat(core): add parallel tool dispatch via asyncio.gather

Independent MCP tool calls (no prerequisites) are now dispatched
concurrently. Dependent calls (memory_write after memory_arbo) remain
sequential as enforced by TOOL_PREREQUISITES.

Reduces average tool round latency from ~3s to ~1s for multi-tool turns.
```
