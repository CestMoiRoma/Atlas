# Architecture Interne d'Atlas

> Ce document décrit le fonctionnement interne d'Atlas à destination des contributeurs et des
> mainteneurs. Pour l'installation et l'usage, voir [03-user-manual.md](03-user-manual.md).

---

## Table des matières

1. [Vue d'ensemble — pipeline vocal](#1-vue-densemble--pipeline-vocal)
2. [Modules du package `atlas/`](#2-modules-du-package-atlas)
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
3. [Boucle multi-round d'outils](#3-boucle-multi-round-doutils)
4. [Serveurs MCP — atlas/tools/](#4-serveurs-mcp--atlastools)
5. [Système mémoire Obsidian](#5-système-mémoire-obsidian)
6. [Sleeping mode](#6-sleeping-mode)
7. [Référence Config](#7-référence-config)

---

## 1. Vue d'ensemble — pipeline vocal

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           ATLAS — Pipeline vocal                          │
└──────────────────────────────────────────────────────────────────────────┘

  Microphone
      │
      ▼
 ┌─────────────┐    keyword      ┌──────────────────────────┐
 │  WakeWord   │ ──────────────▶ │  STT  (whisper-cli)      │
 │  Listener   │  (Atlas.onnx)   │  VAD + no_speech_prob    │
 └─────────────┘                 └────────────┬─────────────┘
                                              │ texte brut
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
 │                       │ oui          ┌─────────────────┐ │
 │                       └────────────▶ │  MCPClient      │ │
 │                                      │  timeout+gather │ │
 │                                      └────────┬────────┘ │
 │                       ┌─────────────────────◀─┘          │
 │                       │ résultats outils                  │
 │                       ▼                                   │
 │                  boucle suivante (MAX_TOOL_ROUNDS)        │
 │                       │ réponse finale                    │
 └───────────────────────┼──────────────────────────────────┘
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

**8 étapes dans l'ordre :**

| # | Étape | Module |
|---|-------|--------|
| 1 | Détection du wakeword | `core/wake_word.py` |
| 2 | Enregistrement + VAD | `core/stt.py` |
| 3 | Transcription whisper | `core/stt.py` |
| 4 | Identification du locuteur | `core/speaker_id.py` |
| 5 | Construction du prompt système | `core/orchestrator.py` |
| 6 | Inférence LLM + boucle outils | `core/orchestrator.py` + `core/mcp_client.py` |
| 7 | Synthèse vocale | `core/tts.py` |
| 8 | Persistance session | `core/session.py` |

---

## 2. Modules du package `atlas/`

### `config.py`

Point d'entrée unique pour toute la configuration. Deux dataclasses :

```python
@dataclass(frozen=True)
class OllamaOptions:
    temperature: float | None = None
    num_ctx: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None

    def to_dict(self) -> dict: ...  # Retourne seulement les champs non-None

@dataclass(frozen=True)
class Config:
    @classmethod
    def from_env(cls) -> "Config": ...  # Lève ConfigError si variable manquante/invalide
```

`Config.from_env()` lit le fichier `.env` via `python-dotenv`, valide les types, et lève
`ConfigError` (sous-classe de `ValueError`) avec un message d'action si une variable
obligatoire est absente ou malformée.

**Avantages vs `os.getenv()` éparpillés :**
- Toutes les variables sont documentées en un seul endroit
- Injection directe dans les tests sans monkeypatching global
- Frozen → immuable en runtime, détecte les bugs de mutation accidentelle

---

### `db/user_db.py`

Couche SQLite WAL, schéma idempotent (créé au premier `init_db()`).

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
  embedding        BLOB   -- numpy float32, sérialisé via tobytes()/frombuffer()
```

**API publique :**

| Fonction | Description |
|----------|-------------|
| `init_db(path)` | Crée/ouvre la DB, active WAL, retourne `Connection` |
| `upsert_user(db, ...)` | INSERT … ON CONFLICT DO UPDATE — ne touche jamais l'embedding |
| `update_embedding(db, name, vec)` | BLOB uniquement — séparé pour éviter d'écraser par accident |
| `get_all_users(db)` | Retourne `list[User]` avec désérialisation BLOB → numpy |
| `get_user_by_name(db, name)` | Retourne `User | None` |

---

### `core/models.py`

Dataclasses légères sans dépendances externes.

```python
@dataclass
class User:
    id: int
    name: str
    user_tag: str
    preferred_address: str
    other_addresses: list[str]
    embedding: np.ndarray | None  # exclu du __repr__

    @property
    def is_guest(self) -> bool: ...
    @property
    def all_addresses(self) -> list[str]: ...  # preferred + other, dédupliqué

@dataclass(frozen=True)
class SpeakerMatch:
    user: User
    score: float        # cosine 0.0–1.0
    method: str         # "match" | "fallback" | "guest"

GUEST_USER: User  # Singleton invité (id=0)
```

---

### `core/audio_gate.py`

Verrou async pour éviter que la STT enregistre pendant que TTS parle.

```python
class AudioGate:
    def open(self) -> None         # event.set()
    def close(self) -> None        # event.clear()
    def is_open(self) -> bool
    async def wait_until_open(self) -> None

    @asynccontextmanager
    async def closed(self):
        # Ferme, yield, reouvre — garanti même en cas d'exception
```

**Singleton de module :** `gate: AudioGate = AudioGate()` — importé directement par STT et TTS.

**Pourquoi `asyncio.Event` et non `threading.Event` ?**  
Le pipeline entier est `async`. `asyncio.Event` évite tout blocage de l'event loop : `await
event.wait()` cède la main aux autres coroutines au lieu de bloquer un thread OS.

---

### `core/wake_word.py`

```python
class WakeWordListener:
    async def listen(self) -> AsyncIterator[str]:
        # Générateur async — yields le keyword détecté
        # Vérifie debounce (_last_fired) pour éviter déclenchements répétés
```

Utilise `livekit-wakeword` avec le modèle `models/Atlas.onnx` (97 KB, commité dans le repo).
Le seuil de confiance est configurable via `WAKE_WORD_THRESHOLD` (défaut : `0.5`).

---

### `core/stt.py`

**Enregistrement — VAD énergie :**

```
Chunks de 512 samples @ 16 kHz
         │
         ▼
  pre-buffer 5 chunks (garde le début de la parole)
         │
  RMS > threshold ? ──Non──▶ silence_count++
         │ Oui                      │
  is_speaking = True     silence_count > MAX_SILENCE ?
         │                          │ Oui
  accumule audio        stop + retourne PCM brut
         │
  MAX_DURATION atteinte ? ──Oui──▶ stop forcé
```

**Transcription — filtre `no_speech_prob` :**

```bash
whisper-cli --output-format json --no-timestamps -f audio.wav
```

Le JSON retourné contient `result[0].no_speech_prob`. Si cette valeur dépasse
`WHISPER_NO_SPEECH_THRESHOLD` (défaut : `0.6`), la transcription est ignorée
(retourne `""`). Cela corrige le bug documenté de *transcription fantôme* où
whisper hallucine "Merci." ou "Sous-titres réalisés par..." sur un silence.

---

### `core/tts.py`

```python
class TTS:
    async def speak(self, text: str) -> None:
        clean = _strip_markdown(text)
        async with gate.closed():          # Ferme le gate pendant la synth
            await _run_say(clean, ...)
```

**16 patterns Markdown strippés :**
blocs de code fencés, code inline, gras/italique (4 combinaisons), titres `#`,
listes `-`/`*`/`1.`, blockquotes `>`, liens `[text](url)`, wikilinks `[[...]]`,
balises HTML.

---

### `core/speaker_id.py`

**Identification en 3 étapes :**

```
audio PCM  →  ECAPA-TDNN  →  vecteur L2-normalisé (192 dims)
                                      │
                          score cosine vs chaque utilisateur
                                      │
         score > SPEAKER_MATCH_THRESHOLD (0.75) ? ──▶ method="match"
                │ Non
         score > SPEAKER_FALLBACK_THRESHOLD (0.55) ? ──▶ method="fallback" (utilisateur le plus probable)
                │ Non
         ──▶ GUEST_USER, method="guest"
```

**Auto-amélioration des embeddings :**
- Chaque match enregistre un WAV dans `user_voice_templates/<user_tag>/sample_N.wav`
- Pruning si `> MAX_VOICE_SAMPLES` (défaut : 10)
- `_recompute_embedding()` : moyenne des embeddings de tous les WAVs sauvegardés

**Sleeping mode :** `recompute_all_embeddings()` re-average tous les utilisateurs sur les 
nouveaux échantillons accumulés.

---

### `core/session.py`

```python
class SessionLog:
    path: Path         # Sessions/YYYY-MM-DD_HHMMSS.md
    turn_count: int
    speakers: set[str]

    def append_turn(self, speaker, user_text, tools_called, reply) -> None
    # Écriture immédiate (crash-safe — pas de buffer)

    def close(self) -> None
    # Footer stats + backlink [[Sessions]] + enregistrement dans Sessions.md
```

**Format d'un fichier de session :**

```markdown
---
type: session
date: 2025-01-15
started_at: 10:23:45
---

# Session 2025-01-15 10:23:45

## Tour 1 — Roma
**Utilisateur :** Quelle heure est-il ?
**Outils :** datetime__get_datetime
**Atlas :** Il est 10h23.

---
*Session fermée — 1 tour(s), durée ~42s*
[[Sessions]]
```

---

### `core/mcp_client.py`

**7 serveurs MCP enregistrés dans `TOOL_SERVERS` :**

| Nom | Module |
|-----|--------|
| `memory` | `atlas.tools.memory` |
| `datetime` | `atlas.tools.datetime_info` |
| `geoposition` | `atlas.tools.geoposition` |
| `weather` | `atlas.tools.weather` |
| `metrics` | `atlas.tools.metrics` |
| `wikipedia` | `atlas.tools.wikipedia` |
| `inbox` | `atlas.tools.inbox` |

**Prérequis d'outils** (`TOOL_PREREQUISITES`) :  
`memory_write`, `memory_patch`, `memory_link`, `memory_delete`, `memory_append`
nécessitent tous `memory__memory_arbo` au préalable. L'orchestrateur enforce cet
ordre avant de dispatcher.

**Timeout par outil :**

```python
result = await asyncio.wait_for(
    _dispatch_tool(server, name, args),
    timeout=config.mcp_tool_timeout   # défaut 10.0s
)
```

**Dispatch parallèle :**

```python
# Appels indépendants → asyncio.gather
results = await asyncio.gather(*tasks, return_exceptions=True)
```

---

### `core/health.py`

6 checks au démarrage, invoqués par `python -m atlas.core.orchestrator --check` :

| Check | Critique | Ce qui est vérifié |
|-------|----------|--------------------|
| Ollama | ✓ | HTTP GET `/` → 200 |
| whisper-cli | ✓ | `shutil.which("whisper-cli")` |
| Whisper model | ✓ | `WHISPER_MODEL_PATH` existe |
| Wake word models | ✓ | Tous les paths `WAKE_WORD_MODEL_PATHS` existent |
| Database | ✓ | `sqlite3.connect()` réussit |
| Vault | ✗ | `ATLAS_VAULT_PATH` existe (avertissement si absent) |

Sortie terminale colorée : `✓` vert / `⚠` jaune / `✗` rouge.  
`HealthCheckError` levée si au moins un check critique échoue.

---

### `core/orchestrator.py`

Point d'entrée principal du pipeline. Modes :

| Flag | Comportement |
|------|-------------|
| *(défaut)* | Wakeword + audio complet |
| `--text` | Stdin → bypass wakeword/STT, lit une ligne, idéal pour debug |
| `--check` | Lance health check puis quitte |
| `--nothink` | Désactive les tokens `<think>` d'Ollama (plus rapide) |

---

## 3. Boucle multi-round d'outils

```python
MAX_TOOL_ROUNDS = 6   # Cap anti-boucle infinie

for round_n in range(MAX_TOOL_ROUNDS):
    response = await ollama.chat(messages, tools=schemas)

    if response has tool_calls:
        # 1. Enforce prerequisites (memory_arbo si nécessaire)
        # 2. Dispatch indépendants en parallèle
        # 3. Ajouter résultats au contexte
        continue                            # → round suivant

    text = response.message.content

    if text.endswith("[SUITE]"):            # Sentinelle de continuation
        tts.speak(text.removesuffix("[SUITE]"))
        messages.append(user="continue")
        continue

    break  # Réponse finale → TTS → session log
```

**Sentinelle `[SUITE]` :** permet au LLM de produire une réponse longue en plusieurs
parties vocales sans dépasser le contexte. Cap à `_QUESTION_SENTINEL_CAP = 3` itérations
successives de suite sans tool calls pour éviter les boucles.

---

## 4. Serveurs MCP — `atlas/tools/`

Chaque outil est un serveur **FastMCP stdio** — spawné comme sous-processus par
`MCPClient`. Communication via le protocole MCP sur stdin/stdout.

| Fichier | Outils exposés |
|---------|---------------|
| `memory.py` | `memory_arbo`, `memory_read`, `memory_write`, `memory_patch`, `memory_link`, `memory_delete`, `memory_append` |
| `datetime_info.py` | `get_datetime` |
| `geoposition.py` | `get_location` |
| `weather.py` | `get_weather` |
| `metrics.py` | `get_system_metrics` |
| `wikipedia.py` | `search_wikipedia`, `get_wikipedia_article` |
| `inbox.py` | `read_inbox` |

**Protection traversée vault (`memory.py`) :**

```python
def _note_path(vault: Path, name: str) -> Path:
    target = (vault / name).resolve()
    if not target.is_relative_to(vault.resolve()):
        raise ValueError("Path traversal interdit")
    return target
```

---

## 5. Système mémoire Obsidian

**Structure du vault (`ATLAS_VAULT_PATH`) :**

```
atlas_memory/
├── Sessions.md          ← Hub index de toutes les sessions
├── Sessions/
│   ├── 2025-01-15_102345.md
│   └── 2025-01-16_090012.md
├── Topics/
│   ├── Python.md
│   └── Développement.md
├── People/
│   └── Roma.md
└── Notes/
    └── ...
```

**Auto-tagging :** `memory_write` injecte automatiquement `user_tag` dans le
frontmatter YAML de chaque note créée :

```yaml
---
type: note
user_tag: user_roma
date: 2025-01-15
---
```

**Wikilinks :** Les notes se référencent entre elles via `[[NomNote]]`. L'outil
`memory_link` ajoute un lien bidirectionnel. `memory_arbo` retourne l'arborescence
complète pour contextualiser les appels d'écriture.

**Sessions hub (`Sessions.md`) :**  
`SessionLog.close()` ajoute automatiquement `- [[Sessions/YYYY-MM-DD_HHMMSS]]` à
`Sessions.md`. `scripts/index_sessions.py` rétro-indexe les sessions orphelines.

---

## 6. Sleeping mode

**Déclencheur :** inactivité > `SLEEPING_TIMEOUT` secondes (défaut : 300s).

**Actions :**

1. TTS annonce la mise en veille (`"Je passe en mode veille."`)
2. `recompute_all_embeddings(config, db)` re-average tous les embeddings utilisateurs
   sur les nouveaux samples vocaux accumulés depuis la dernière veille
3. `SessionLog.close()` — ferme et archive la session en cours
4. Nouvelle `SessionLog()` créée — prête pour la session suivante
5. Retour à l'écoute du wakeword

Implémenté dans `_sleeping_mode_monitor()` comme tâche asyncio de fond — ne bloque
pas le pipeline principal.

---

## 7. Référence Config

| Champ | Variable `.env` | Défaut | Obligatoire |
|-------|----------------|--------|-------------|
| `ollama_host` | `OLLAMA_HOST` | `http://localhost:11434` | Non |
| `ollama_model` | `OLLAMA_MODEL` | `llama3.2` | Non |
| `whisper_bin` | `WHISPER_BIN` | `whisper-cli` | Non |
| `whisper_model_path` | `WHISPER_MODEL_PATH` | — | **Oui** |
| `whisper_language` | `WHISPER_LANGUAGE` | `fr` | Non |
| `whisper_no_speech_threshold` | `WHISPER_NO_SPEECH_THRESHOLD` | `0.6` | Non |
| `wake_word_models` | `WAKE_WORD_MODEL_PATHS` | — | **Oui** |
| `wake_word_threshold` | `WAKE_WORD_THRESHOLD` | `0.5` | Non |
| `wake_word_debounce` | `WAKE_WORD_DEBOUNCE` | `2.0` | Non |
| `vault_path` | `ATLAS_VAULT_PATH` | — | **Oui** |
| `speaker_db_path` | `SPEAKER_DB_PATH` | `./atlas_users.db` | Non |
| `voice_templates_dir` | `VOICE_TEMPLATES_DIR` | `./user_voice_templates` | Non |
| `speaker_match_threshold` | `SPEAKER_MATCH_THRESHOLD` | `0.75` | Non |
| `speaker_fallback_threshold` | `SPEAKER_FALLBACK_THRESHOLD` | `0.55` | Non |
| `max_voice_samples` | `MAX_VOICE_SAMPLES` | `10` | Non |
| `mcp_tool_timeout` | `MCP_TOOL_TIMEOUT` | `10.0` | Non |
| `sleeping_timeout` | `SLEEPING_TIMEOUT` | `300` | Non |
| `tts_rate` | `TTS_RATE` | *(say défaut)* | Non |
| `claude_api_key` | `CLAUDE_API_KEY` | `""` | Non |
| `ollama_options` | `OLLAMA_TEMPERATURE`, `OLLAMA_NUM_CTX`, … | `None` each | Non |

`ollama_options_dict()` retourne uniquement les options non-`None` — seuls ces champs
sont envoyés à l'API Ollama pour ne pas écraser les valeurs du modèle.
