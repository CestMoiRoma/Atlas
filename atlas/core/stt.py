# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/stt.py
=================
Speech-to-text module — energy-based VAD recorder + whisper.cpp transcription.

Pipeline
--------
1. **Record** — ``record_utterance()`` captures audio from the microphone using
   an energy-based Voice Activity Detector.  Returns a ``numpy`` float32 array
   at 16 kHz.  Returns an empty array if no speech was detected within the
   timeout.

2. **Transcribe** — ``transcribe()`` calls ``whisper-cli`` on the recorded
   audio, parses the JSON output, and applies the ``no_speech_prob`` filter to
   silently discard near-silence hallucinations.

Energy VAD
----------
The VAD uses a simple RMS threshold:

* A rolling pre-speech buffer keeps the last ``PRE_SPEECH_CHUNKS`` audio
  chunks so the first syllable is never clipped.
* Recording starts when the RMS of a chunk exceeds ``config.vad_energy_threshold``.
* Recording stops when ``config.vad_silence_after`` seconds of consecutive
  silence are observed after speech started.
* A hard cap of ``config.vad_max_duration`` seconds prevents unbounded recording.

no_speech_prob filter
---------------------
whisper-cli is invoked with ``--output-format json``.  The resulting JSON
contains a ``no_speech_prob`` field per segment.  Any transcription where the
first segment's ``no_speech_prob`` exceeds ``config.whisper_no_speech_threshold``
is discarded silently — this eliminates the phantom "Thank you." hallucinations that
whisper produces on near-silence inputs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from atlas.config import Config
from atlas.core.audio_gate import gate

logger = logging.getLogger(__name__)

# Number of audio chunks held in the pre-speech buffer.
# At 16 kHz / 512 samples per chunk this is ~160 ms of audio kept before VAD
# triggers — enough to capture the leading consonant of any wake word response.
_PRE_SPEECH_CHUNKS = 5
_CHUNK_SAMPLES = 512


class STT:
    """Speech-to-text engine wrapping VAD recording and whisper.cpp.

    Args:
        config: Atlas runtime configuration.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config

    # ── Recording ─────────────────────────────────────────────────────────────

    async def record_utterance(self) -> np.ndarray:
        """Record a single utterance from the microphone using energy VAD.

        Runs the blocking sounddevice capture in a thread executor so the
        event loop stays responsive.

        Returns:
            Float32 numpy array at ``config.audio_sample_rate`` Hz, or an
            empty array (``size == 0``) if no speech was detected in time.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._record_sync)

    def _record_sync(self) -> np.ndarray:
        """Blocking VAD recorder — called from a thread pool."""
        cfg = self._cfg
        rate = cfg.audio_sample_rate
        device = cfg.audio_input_device if cfg.audio_input_device >= 0 else None

        pre_buffer: list[np.ndarray] = []   # rolling pre-speech buffer
        speech_chunks: list[np.ndarray] = []
        speaking = False
        silence_count = 0

        # Derived counts
        chunks_per_sec = rate / _CHUNK_SAMPLES
        silence_chunks = int(cfg.vad_silence_after * chunks_per_sec)
        max_chunks = int(cfg.vad_max_duration * chunks_per_sec)
        timeout_chunks = int(cfg.vad_speech_timeout * chunks_per_sec)
        pre_speech_size = _PRE_SPEECH_CHUNKS

        chunks_waited = 0

        with sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            blocksize=_CHUNK_SAMPLES,
            device=device,
        ) as stream:
            while True:
                chunk, _ = stream.read(_CHUNK_SAMPLES)
                chunk = chunk[:, 0]  # mono

                # Gate: discard chunks captured while TTS is playing
                if not gate.is_open():
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)) * 32767)

                if not speaking:
                    pre_buffer.append(chunk.copy())
                    if len(pre_buffer) > pre_speech_size:
                        pre_buffer.pop(0)

                    if rms >= cfg.vad_energy_threshold:
                        speaking = True
                        speech_chunks.extend(pre_buffer)
                        pre_buffer.clear()
                        logger.debug("VAD: speech start  rms=%.0f", rms)
                    else:
                        chunks_waited += 1
                        if chunks_waited >= timeout_chunks:
                            logger.debug("VAD: speech timeout — no speech detected")
                            return np.array([], dtype=np.float32)
                else:
                    speech_chunks.append(chunk.copy())

                    if rms < cfg.vad_energy_threshold:
                        silence_count += 1
                        if silence_count >= silence_chunks:
                            logger.debug("VAD: speech end  chunks=%d", len(speech_chunks))
                            break
                    else:
                        silence_count = 0

                    if len(speech_chunks) >= max_chunks:
                        logger.warning("VAD: max duration reached — truncating")
                        break

        if not speech_chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(speech_chunks)
        logger.debug("VAD: recorded %.2f s", len(audio) / rate)
        return audio

    # ── Transcription ─────────────────────────────────────────────────────────

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 audio array using whisper.cpp.

        Saves the audio to a temporary WAV file, invokes ``whisper-cli`` with
        JSON output, and applies the ``no_speech_prob`` filter.

        Args:
            audio: Float32 numpy array at ``config.audio_sample_rate`` Hz.

        Returns:
            Transcribed text, stripped of leading/trailing whitespace.
            Empty string if:

            * ``audio`` is empty,
            * ``no_speech_prob`` exceeds the configured threshold,
            * or the whisper subprocess fails.
        """
        if audio.size == 0:
            return ""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Blocking whisper.cpp call — called from a thread pool."""
        cfg = self._cfg

        if not shutil.which(cfg.whisper_bin):
            logger.error("whisper-cli not found in PATH: %r", cfg.whisper_bin)
            return ""

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "utterance.wav"
            sf.write(str(wav_path), audio, cfg.audio_sample_rate, subtype="PCM_16")

            cmd = [
                cfg.whisper_bin,
                "--model", str(cfg.whisper_model),
                "--language", cfg.whisper_language,
                "--output-format", "json",
                "--no-prints",
                "--file", str(wav_path),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                logger.error("whisper-cli timed out after 60 s")
                return ""
            except Exception as exc:
                logger.error("whisper-cli failed to start: %s", exc)
                return ""

            if result.returncode != 0:
                logger.warning("whisper-cli exited %d: %s", result.returncode, result.stderr[:200])
                return ""

            # whisper writes the JSON file next to the input file
            json_path = wav_path.with_suffix(".json")
            if not json_path.exists():
                # Fall back: try parsing stdout directly
                raw = result.stdout.strip()
            else:
                raw = json_path.read_text(encoding="utf-8")

            return self._parse_whisper_json(raw)

    def _parse_whisper_json(self, raw: str) -> str:
        """Parse whisper JSON output and apply the no_speech_prob filter.

        Args:
            raw: Raw JSON string from whisper-cli.

        Returns:
            Transcribed text, or empty string if filtered out.
        """
        if not raw:
            return ""

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # whisper sometimes prints the text directly without JSON
            logger.debug("whisper output is not JSON — using raw text")
            return raw.strip()

        segments: list[dict] = data.get("transcription", data.get("segments", []))
        if not segments:
            return ""

        # Apply no_speech_prob filter on the first segment
        first = segments[0]
        no_speech = float(first.get("no_speech_prob", 0.0))
        if no_speech > self._cfg.whisper_no_speech_threshold:
            logger.debug(
                "Transcription discarded — no_speech_prob=%.3f > threshold=%.3f",
                no_speech, self._cfg.whisper_no_speech_threshold,
            )
            return ""

        text = " ".join(
            seg.get("text", "").strip()
            for seg in segments
            if seg.get("text", "").strip()
        )
        logger.debug("Transcription: %r  (no_speech_prob=%.3f)", text, no_speech)
        return text.strip()
