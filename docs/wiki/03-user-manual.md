# Manuel Utilisateur Atlas

Atlas est un assistant vocal IA local-first pour macOS — entièrement hors ligne
après la configuration initiale. Ce guide couvre l'installation, la configuration,
et l'utilisation quotidienne.

---

## Table des matières

1. [Prérequis système](#1-prérequis-système)
2. [Installation](#2-installation)
3. [Configuration `.env`](#3-configuration-env)
4. [Inscription d'un utilisateur](#4-inscription-dun-utilisateur)
5. [Démarrer Atlas](#5-démarrer-atlas)
6. [Options de la ligne de commande](#6-options-de-la-ligne-de-commande)
7. [Commandes vocales et mémoire](#7-commandes-vocales-et-mémoire)
8. [Scripts utilitaires](#8-scripts-utilitaires)
9. [FAQ](#9-faq)

---

## 1. Prérequis système

| Composant | Requis | Notes |
|-----------|--------|-------|
| macOS | 13 Ventura ou supérieur | Requis pour `say`, CoreLocation, PortAudio |
| Python | 3.10+ | Disponible via Homebrew (`brew install python@3.12`) |
| Ollama | Dernière version | [ollama.ai](https://ollama.ai) — tourne en arrière-plan |
| whisper.cpp | `whisper-cli` dans `$PATH` | Voir §2.4 |
| PortAudio | — | `brew install portaudio` — requis par `sounddevice` |
| Disk | ~1.5 GB minimum | Modèles Whisper (~800 MB) + SpeechBrain (~80 MB) |
| RAM | 8 GB recommandé | Pour Ollama + modèle Whisper en mémoire simultanément |

---

## 2. Installation

### 2.1 Installer les dépendances système

```bash
# Homebrew requis — https://brew.sh
brew install python@3.12 portaudio
```

### 2.2 Installer Ollama et télécharger un modèle LLM

```bash
# Installer Ollama (téléchargeur macOS sur ollama.ai)
# Puis démarrer le serveur :
ollama serve &

# Télécharger le modèle de langage (ex: llama3.2 ~2 GB)
ollama pull llama3.2
```

### 2.3 Installer Atlas

```bash
git clone https://github.com/CestMoiRoma/Atlas.git
cd Atlas
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2.4 Installer whisper.cpp

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
cmake -B build && cmake --build build --config Release -j
# Copier l'exécutable dans $PATH :
cp build/bin/whisper-cli /usr/local/bin/whisper-cli
```

Télécharger un modèle Whisper GGML (ex: `base` pour un bon équilibre vitesse/qualité) :

```bash
# Depuis la racine de whisper.cpp :
bash models/download-ggml-model.sh base
# → Fichier : models/ggml-base.bin (~142 MB)
```

### 2.5 Télécharger les modèles Atlas

```bash
# Depuis la racine du repo Atlas, avec .venv activé :
python scripts/download_models.py
```

Télécharge automatiquement SpeechBrain ECAPA-TDNN (~80 MB) dans
`models/spkrec-ecapa-voxceleb/`.

---

## 3. Configuration `.env`

```bash
cp .env.example .env
```

Éditer `.env` — variables **obligatoires** :

```bash
# Vault Obsidian — dossier Markdown où Atlas stocke sa mémoire
ATLAS_VAULT_PATH=/Users/toi/Documents/atlas_memory

# Modèle Whisper GGML (chemin absolu recommandé)
WHISPER_MODEL_PATH=/Users/toi/whisper.cpp/models/ggml-base.bin

# Modèle wakeword (déjà dans le repo)
WAKE_WORD_MODEL_PATHS=models/Atlas.onnx
```

Variables **optionnelles** fréquemment personnalisées :

```bash
# Modèle Ollama à utiliser
OLLAMA_MODEL=llama3.2           # ou llama3.1, mistral, gemma2:9b, etc.

# Langue de transcription Whisper
WHISPER_LANGUAGE=fr             # fr, en, es, de, ...

# Seuil de filtrage des transcriptions fantômes (0.0 à 1.0)
# Plus élevé = moins filtrant. 0.6 est un bon défaut.
WHISPER_NO_SPEECH_THRESHOLD=0.6

# Timeout des outils MCP en secondes
MCP_TOOL_TIMEOUT=10.0

# Durée d'inactivité avant mise en veille (secondes)
SLEEPING_TIMEOUT=300

# Vitesse de synthèse vocale (mots/minute, vide = défaut système)
TTS_RATE=
```

---

## 4. Inscription d'un utilisateur

Atlas identifie les locuteurs par leur voix (ECAPA-TDNN). Il faut enregistrer
au moins un utilisateur pour qu'Atlas sache comment s'adresser à toi.

```bash
python scripts/register_user.py \
  --name "Roma" \
  --age 28 \
  --gender M \
  --profession "Développeur" \
  --preferred-address "chef"
```

Le script enregistre **5 extraits vocaux de 4 secondes** chacun (des pauses entre chaque).
Parle normalement pendant l'enregistrement — aucun contenu spécifique n'est requis.

**Options disponibles :**

| Option | Description |
|--------|-------------|
| `--name` | Nom complet (obligatoire) |
| `--age` | Âge (optionnel) |
| `--gender` | Genre : M, F, ou libre |
| `--profession` | Profession (optionnel) |
| `--preferred-address` | Comment Atlas doit t'appeler |
| `--update` | Met à jour le profil sans ré-enregistrer la voix |
| `--re-record` | Re-enregistre les samples vocaux |

**Modifier un profil existant :**

```bash
python scripts/edit_user.py --name "Roma" --profession "CTO"

# Lister tous les utilisateurs :
python scripts/edit_user.py --list
```

---

## 5. Démarrer Atlas

### Prérequis avant démarrage

```bash
# 1. Ollama doit tourner
ollama serve &

# 2. Activer l'environnement Python
source .venv/bin/activate
```

### Démarrage standard

```bash
python -m atlas.core.orchestrator
```

ou via l'entry point installé :

```bash
atlas
```

Atlas affiche le résultat du health check, puis attend le wakeword.

**Dire "Atlas"** (ou le wakeword configuré) pour démarrer une interaction.

### Arrêt propre

`Ctrl+C` — Atlas ferme la session en cours et quitte proprement.

---

## 6. Options de la ligne de commande

### `--check` — Health check uniquement

```bash
python -m atlas.core.orchestrator --check
```

Vérifie toutes les dépendances et quitte. Utile pour diagnostiquer des problèmes
sans démarrer le pipeline complet.

### `--text` — Mode texte (debug / scripting)

```bash
# Interaction simple
echo "Quelle heure est-il ?" | python -m atlas.core.orchestrator --text

# Session interactive
python -m atlas.core.orchestrator --text
# → Taper les questions, une par ligne, Ctrl+D pour quitter
```

Bypass total du wakeword et de la STT — idéal pour tester des prompts ou déboguer
les outils MCP sans microphone.

### `--nothink` — Désactiver les tokens de réflexion

```bash
python -m atlas.core.orchestrator --nothink
```

Désactive les tokens `<think>` d'Ollama (si supportés par le modèle). Réduit la
latence au détriment d'un raisonnement moins structuré.

---

## 7. Commandes vocales et mémoire

### Interaction naturelle

Atlas est conçu pour le français. Pas de syntaxe spéciale — parle normalement :

> *"Atlas, c'est quoi la météo à Lyon ?"*  
> *"Atlas, note que le déploiement est prévu pour vendredi."*  
> *"Atlas, qu'est-ce que tu sais sur Python ?"*

### Outils disponibles

| Outil | Exemples de déclencheurs |
|-------|-------------------------|
| `datetime` | "quelle heure", "quel jour", "quelle date" |
| `geoposition` | "où suis-je", "ma localisation" |
| `weather` | "météo", "temps qu'il fait", "température" |
| `metrics` | "CPU", "mémoire disponible", "charge système" |
| `wikipedia` | "c'est quoi", "définition de", "parle-moi de" |
| `memory` | "note que", "souviens-toi", "qu'est-ce que tu sais sur" |
| `inbox` | "lis mon inbox", "qu'est-ce qu'il y a dans" |

### Mémoire persistante

Atlas stocke ses notes dans le vault Obsidian (`ATLAS_VAULT_PATH`). Les sessions
sont indexées dans `Sessions.md`. Les notes créées via la mémoire apparaissent
dans `Topics/`, `People/`, etc.

**Atlas peut retrouver des informations dites lors de conversations précédentes**
tant que la note correspondante a été créée dans le vault.

---

## 8. Scripts utilitaires

### Télécharger les modèles

```bash
python scripts/download_models.py
```

### Indexer des sessions orphelines

Si des fichiers `Sessions/*.md` n'apparaissent pas dans `Sessions.md` :

```bash
python scripts/index_sessions.py
# Ou spécifier un vault :
python scripts/index_sessions.py --vault /chemin/vers/vault
```

### Fusionner des topics en double

Après une longue utilisation, des topics similaires peuvent s'accumuler
(ex: `Python.md` et `Programmation_Python.md`) :

```bash
# Revue interactive
python scripts/unify_topics.py

# Fusion automatique des paires à haute similarité (>= 80%)
python scripts/unify_topics.py --threshold 0.8 --auto
```

### Embedding de fichiers dans le vault

```bash
# Embedding simple (chunks de 4096 chars)
python scripts/embed_memory.py /chemin/vers/document.md

# Embedding large-context (fenêtres de 100K chars, 50% overlap)
python scripts/embed_deep.py /chemin/vers/document_long.md
```

---

## 9. FAQ

### Ollama ne répond pas / "Connection refused"

```bash
# Vérifier qu'Ollama tourne
ollama list

# Démarrer manuellement si nécessaire
ollama serve &

# Vérifier le health check
python -m atlas.core.orchestrator --check
```

### `whisper-cli` introuvable

```bash
# Vérifier que whisper-cli est dans $PATH
which whisper-cli

# Si absent, recompiler (voir §2.4) ou vérifier :
ls /usr/local/bin/whisper-cli
```

Si `whisper-cli` est dans un répertoire non standard, définir dans `.env` :

```bash
WHISPER_BIN=/chemin/vers/whisper-cli
```

### Le wakeword n'est pas détecté

1. Vérifier que le microphone est autorisé pour le Terminal dans
   `Réglages Système → Confidentialité → Microphone`
2. S'assurer que `models/Atlas.onnx` existe : `ls -la models/Atlas.onnx`
3. Baisser le seuil de détection dans `.env` : `WAKE_WORD_THRESHOLD=0.3`
4. Tester en mode texte pour isoler le problème : `echo "test" | python -m atlas.core.orchestrator --text`

### Transcriptions fantômes ("Merci.", "Sous-titres réalisés par...")

Ce bug est connu de whisper.cpp — le modèle hallucine sur les silences.

Atlas filtre automatiquement via `no_speech_prob`. Si les fantômes persistent,
baisser le seuil (= filtrer plus agressivement) :

```bash
# Dans .env :
WHISPER_NO_SPEECH_THRESHOLD=0.4   # défaut : 0.6
```

Note : un seuil trop bas peut éliminer des transcriptions légitimes à faible
confiance (voix lointaine, bruit de fond).

### Atlas n'identifie pas ma voix

1. Vérifier que l'utilisateur est bien inscrit :
   ```bash
   python scripts/edit_user.py --list
   ```

2. Re-enregistrer les samples vocaux dans un environnement calme :
   ```bash
   python scripts/register_user.py --name "Roma" --re-record
   ```

3. Vérifier les seuils d'identification dans `.env` :
   ```bash
   SPEAKER_MATCH_THRESHOLD=0.70     # Baisser légèrement (défaut: 0.75)
   SPEAKER_FALLBACK_THRESHOLD=0.50  # Baisser légèrement (défaut: 0.55)
   ```

### Comment mettre à jour Atlas ?

```bash
git pull origin DEV_AtlasV0.1
pip install -e .
python -m atlas.core.orchestrator --check
```

Si le schéma de la base de données a changé, supprimer et recréer :

```bash
rm atlas_users.db
# Ré-inscrire les utilisateurs
python scripts/register_user.py --name "Roma" ...
```

### La session ne s'enregistre pas dans le vault

1. Vérifier que `ATLAS_VAULT_PATH` pointe vers un répertoire existant avec permissions
   d'écriture
2. Vérifier que `Sessions/` se crée bien au premier démarrage
3. Utiliser `scripts/index_sessions.py` pour rétro-indexer les fichiers existants

### Atlas est trop lent à répondre

Options pour réduire la latence :

- Utiliser un modèle Ollama plus léger : `OLLAMA_MODEL=llama3.2:1b`
- Désactiver les tokens de réflexion : `python -m atlas.core.orchestrator --nothink`
- Réduire le contexte LLM : `OLLAMA_NUM_CTX=2048`
- Utiliser un modèle Whisper plus petit : `ggml-tiny.bin` au lieu de `ggml-base.bin`
