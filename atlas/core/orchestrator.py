# SPDX-License-Identifier: AGPL-3.0-or-later
"""
atlas/core/orchestrator.py
==========================
Main async control loop for the Atlas voice assistant.

Pipeline — one conversation turn
---------------------------------
::

    1. Wake word detected      WakeWordListener.listen()
            ↓
    2. Record utterance        STT.record_utterance()
            ↓
    3. [parallel]
       Identify speaker        SpeakerIdentifier.identify()
       Transcribe              STT.transcribe()
            ↓
    4. Build system prompt     _build_system_prompt()
            ↓
    5. LLM inference           ollama.AsyncClient.chat()
            ↓
    6. Multi-round tool loop   MCPClient.call_tool() / call_tools_parallel()
            ↓
    7. TTS playback            TTS.speak()
            ↓
    8. Session log             SessionLog.append_turn()

Multi-round tool loop
---------------------
The model may request multiple rounds of tool calls before producing a final
text reply.  Each round: call Ollama → if tool calls are present, dispatch
them (respecting prerequisite ordering) and loop.  Capped at
``config.max_tool_rounds`` to prevent infinite loops.

[SUITE] continuation sentinel
------------------------------
The model may append ``[SUITE]`` to signal "I want to speak an acknowledgement
and then act autonomously".  Atlas strips ``[SUITE]`` from TTS, speaks the
text, then fires a free tool-call turn.  If ``[SUITE]`` appears on a pure
question (no prior sentence ending in ``.`` or ``!``), the model is gently
nudged with a tight cap (3 rounds) to complete its intended actions.

Sleeping mode
-------------
A background task monitors inactivity.  After ``config.sleep_timeout`` seconds
of silence it closes the current session log, re-averages all voice embeddings,
and opens a fresh session for the next conversation.

--text debug mode
-----------------
When ``--text`` is passed on the CLI, wake word detection and audio recording
are bypassed entirely.  Each line typed on stdin is processed as a turn.
Useful for testing prompt changes, tool routing, and memory behaviour without
a microphone.

--check mode
------------
Runs only the startup health check and exits — no main loop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import os
import random
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from ollama import AsyncClient, Message  # type: ignore[import]

from atlas.config import Config, ConfigError
from atlas.core.health import HealthCheckError, run_health_check
from atlas.core.mcp_client import MCPClient, TOOL_PREREQUISITES
from atlas.core.models import GUEST_USER, User
from atlas.core.session import SessionLog
from atlas.core.speaker_id import SpeakerIdentifier, recompute_all_embeddings, warm_up
from atlas.core.stt import STT
from atlas.core.tts import TTS
from atlas.core.wake_word import WakeWordListener
from atlas.db.user_db import init_db

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_CONTINUE_SENTINEL = "[SUITE]"
_QUESTION_SENTINEL_CAP = 3

# Default wake acknowledgements — overridden at runtime by config.wake_ack_phrases
_WAKE_ACK_DEFAULT = [
    "Yes?",
    "Listening.",
    "Yes, go ahead.",
    "Here.",
    "I'm listening.",
]

# Strips <think>...</think> reasoning blocks emitted by models like Gemma 4.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Detects a "pure question" — only used to guard against [SUITE] on bare questions.
_IS_PURE_QUESTION = re.compile(r"^[^.!]*\?\s*$", re.DOTALL)


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_file_logging(log_path: str) -> None:
    """Attach a rotating DEBUG file handler to the root logger."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        p, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-35s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    for h in root.handlers:
        if h.level == logging.NOTSET:
            h.setLevel(logging.INFO)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    logger.info("File logging → %s  (DEBUG+, rotating 10 MB × 3)", p.resolve())


# ── System prompt ─────────────────────────────────────────────────────────────

_MEMORY_GRAPH = """
=== OBSIDIAN MEMORY — KNOWLEDGE GRAPH ===

You have a structured Obsidian vault as your long-term memory.
You MUST use it VERY frequently — both for reading and writing.

── MEMORY TOOLS ──
  memory__memory_read             read a note
  memory__memory_write            create/overwrite a note  ← user_tag REQUIRED
  memory__memory_append           append content to the end of a note
  memory__memory_patch_section    replace the content of an existing ## section
  memory__memory_link             create a bidirectional wikilink between two notes
  memory__memory_search           search note names and content
  memory__memory_delete           permanently delete a note
  memory__memory_arbo             view the vault directory tree (or a subfolder)

── DIRECTORY STRUCTURE ──
  Users/{Name}.md          user hub node
  Topics/{Subject}.md      thematic node (work, project X, hobby…)
  Memories/{date} - {Name} - {subject}.md   memory node (event, info, exchange)
  Sessions/{date}_{time}.md                 automatic journal of each session

── MANDATORY RULES ──
BEFORE any memory operation → Call memory__memory_arbo to see what exists.
BEFORE each reply about the user or their projects → memory__memory_search then memory__memory_read.
AFTER each exchange with a new fact → Create a Memories/ node with memory__memory_write (user_tag required).
LINKS → Every Memories/ node must wikilink [[Name]] AND [[Topic(s)]] via memory__memory_link.
=== END MEMORY ===
"""


def _load_memory_graph(config: Config) -> str:
    """Return the memory graph prompt block.

    If ``config.memory_graph_file`` points to an existing file its content is
    returned verbatim, letting users customise the Obsidian memory instructions
    without touching source code.  Otherwise the built-in ``_MEMORY_GRAPH``
    constant is returned.
    """
    if config.memory_graph_file:
        p = Path(config.memory_graph_file)
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.warning("Could not read MEMORY_GRAPH_FILE %s: %s", p, exc)
    return _MEMORY_GRAPH


def _build_system_prompt(user: User, config: Config, think: bool | None = None) -> str:
    """Build a per-turn system prompt tailored to the identified speaker."""
    voice_rules = (
        "You are Atlas, a local AI voice assistant. "
        f"Always respond in {config.response_language}, concisely and naturally for spoken delivery. "
        "FORBIDDEN: markdown, tables, bullet lists, asterisks, hashes, "
        "list dashes, code blocks, or any other visual formatting — "
        "you speak, you do not write."
    )
    if config.voice_rules_extra:
        voice_rules += " " + config.voice_rules_extra

    think_hint = ""
    if think is not False and config.think_depth:
        _depth_map = {
            "short":    "Reason very briefly — a few sentences are enough.",
            "moderate": "Reason concisely — stay focused on the essentials.",
            "deep":     "Reason in depth if the question warrants it.",
        }
        hint_text = _depth_map.get(config.think_depth, config.think_depth)
        think_hint = f"\nREASONING DEPTH: {hint_text}"

    action_rules = (
        "EXECUTION ORDER — two valid options:\n"
        "• Option 1 (preferred): call tools directly, "
        "then report the result to the user in the same message.\n"
        "• Option 2 (if you need to speak before acting): announce your INTENTION only, "
        "then add [SUITE] at the very end of your message. "
        "[SUITE] will NOT be spoken aloud — it automatically triggers "
        "a new turn where you can call the tools.\n"
        "FORBIDDEN: promise or confirm an action without calling the tools "
        "AND without [SUITE]."
    )

    tool_triggers = (
        "You MUST call the following tools as soon as the topic is mentioned:\n"
        "• POSITION: 'where am I', 'what city', 'my location' → get_current_place\n"
        "• LOCAL WEATHER: 'what's the weather', 'weather' → get_local_weather\n"
        "• CITY WEATHER: weather + city name → get_city_weather\n"
        "• TIME/DATE: 'what time', 'what day' → get_datetime\n"
        "• MAC METRICS: 'cpu', 'ram', 'system stats' → get_mac_metrics\n"
        "• WIKIPEDIA: 'what is', 'who is', 'explain' → wikipedia_search then wikipedia_summary\n"
        "• INBOX: 'read my inbox', 'my files' → inbox_list then inbox_read\n"
        "NEVER answer from memory for these topics."
    )

    if user.is_guest:
        identity = (
            "The current user is unknown (guest). "
            "Welcome them warmly. "
            "If you learn their name, create a Memories/ node with tag user_unknown."
        )
        return f"{voice_rules}{think_hint}\n\n{action_rules}\n\n{tool_triggers}\n\n{_load_memory_graph(config)}\n\n{identity}"

    profile_parts = [p for p in [
        f"{user.age} years old" if user.age else "",
        user.gender or "",
        user.profession or "",
    ] if p]
    profile = ", ".join(profile_parts)

    if len(user.all_addresses) > 1:
        addr_instr = (
            f"Alternate naturally between these nicknames: "
            f"{', '.join(repr(a) for a in user.all_addresses)}."
        )
    else:
        addr_instr = f"Call them '{user.preferred_address}'."

    identity = (
        f"You are talking with {user.name}"
        f"{' (' + profile + ')' if profile else ''}. "
        f"{addr_instr} "
        f"Their user node is Users/{user.name}.md. "
        f"Their Obsidian tag is '{user.user_tag}'. "
        f"EVERY call to memory__memory_write MUST include user_tag='{user.user_tag}'. "
        f"Start the session by checking Users/{user.name}.md."
    )

    return f"{voice_rules}{think_hint}\n\n{action_rules}\n\n{tool_triggers}\n\n{_load_memory_graph(config)}\n\n{identity}"


# ── Turn logic ────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip <think> blocks and non-printable characters."""
    text = _THINK_BLOCK.sub("", text or "").strip()
    return "".join(c for c in text if c.isprintable())


async def _run_turn(
    *,
    ollama: AsyncClient,
    mcp: MCPClient,
    tts: TTS,
    config: Config,
    conversation: list[Message],
    user: User,
    user_text: str,
    session_log: SessionLog,
    think: bool | None,
    pending_audio: list[Any],
    stt: STT,
) -> bool:
    """Execute one complete LLM → tool loop → TTS cycle.

    Args:
        conversation: Mutable conversation history (system prompt at index 0).
        user:         Identified speaker for this turn.
        user_text:    Transcribed utterance.
        pending_audio: Shared list for background audio tasks (question sentinel).
        stt:          STT instance for background recording.

    Returns:
        True if Atlas ended with a question (skip wake word on the next call).
    """
    ollama_options = config.ollama_options_dict() or None
    tools_this_turn: list[str] = []
    tools_since_speech: set[str] = set()

    working = list(conversation)
    tool_round = 0
    had_tool_calls = False
    q_sentinel_rounds = 0
    question_sentinel_spoken = ""
    question_sentinel_capped = False
    reply_override: str | None = None

    while True:
        response = await ollama.chat(
            model=config.ollama_model,
            messages=working,
            tools=conversation[0]._tools if hasattr(conversation[0], "_tools") else None,
            think=think,
            options=ollama_options,
        )
        msg: Message = response.message

        logger.debug(
            "RAW_RESPONSE round=%d content=%r thinking=%r tool_calls=%r",
            tool_round,
            msg.content,
            getattr(msg, "thinking", None),
            [tc.function.name for tc in msg.tool_calls] if msg.tool_calls else None,
        )

        # ── Tool calls ────────────────────────────────────────────────────────
        if msg.tool_calls:
            had_tool_calls = True
            tool_round += 1

            if tool_round > config.max_tool_rounds:
                logger.warning("Tool loop cap (%d) reached — aborting turn", config.max_tool_rounds)
                await tts.speak(config.atlas_loop_message)
                return False

            logger.info("Tool round %d/%d", tool_round, config.max_tool_rounds)
            working.append(msg)

            tool_results: list[str] = []
            blocked_tool: str | None = None
            missing_prereqs: list[str] = []

            for tc in msg.tool_calls:
                tc_name = tc.function.name
                unmet = [
                    p for p in TOOL_PREREQUISITES.get(tc_name, [])
                    if p not in tools_since_speech
                ]
                if unmet:
                    blocked_tool = tc_name
                    missing_prereqs = unmet
                    break

                import json as _json  # noqa: PLC0415
                raw_args = tc.function.arguments
                args = _json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})

                # Always stamp memory_write with the real user_tag
                if tc_name == "memory__memory_write" and not user.is_guest:
                    args["user_tag"] = user.user_tag

                result = await mcp.call_tool(tc_name, args)
                tool_results.append(result)
                tools_this_turn.append(tc_name)
                tools_since_speech.add(tc_name)

            if tool_results:
                working.append(Message(role="tool", content="\n\n".join(tool_results)))

            if blocked_tool:
                missing_str = " et ".join(f"`{p}`" for p in missing_prereqs)
                logger.warning(
                    "Prerequisite violation: %r requires %s — injecting nudge",
                    blocked_tool, missing_str,
                )
                working.append(Message(
                    role="user",
                    content=(
                        f"⛔ INCORRECT TOOL ORDER — you requested `{blocked_tool}` "
                        f"without calling {missing_str} first.\n"
                        f"Call {missing_str} now, then retry `{blocked_tool}`."
                    ),
                ))
            continue

        # ── [SUITE] continuation sentinel ─────────────────────────────────────
        raw_text = _clean(msg.content)
        if _CONTINUE_SENTINEL in raw_text:
            speak_text = raw_text.replace(_CONTINUE_SENTINEL, "").strip()

            # Pure-question guard
            if _IS_PURE_QUESTION.match(speak_text):
                q_sentinel_rounds += 1
                had_tool_calls = True
                tool_round += 1

                if q_sentinel_rounds > _QUESTION_SENTINEL_CAP:
                    logger.warning("Question-sentinel cap reached — treating as final reply")
                    question_sentinel_capped = True
                    reply_override = speak_text
                    break

                if speak_text:
                    await tts.speak(speak_text)
                    tools_since_speech.clear()
                    if not question_sentinel_spoken:
                        question_sentinel_spoken = speak_text

                if pending_audio is not None and not pending_audio:
                    pending_audio.append(asyncio.create_task(stt.record_utterance()))

                working.append(Message(role="assistant", content=speak_text or "…"))
                working.append(Message(
                    role="user",
                    content=(
                        f"⚠️ FORCED CONTINUATION ({q_sentinel_rounds}/{_QUESTION_SENTINEL_CAP}) — "
                        "you just asked a question AND added [SUITE]. "
                        "If actions remain, execute them NOW. "
                        "Otherwise reply directly WITHOUT [SUITE]."
                    ),
                ))
                continue

            # Normal sentinel
            had_tool_calls = True
            tool_round += 1

            if tool_round > config.max_tool_rounds:
                logger.warning("Continuation loop cap reached — aborting")
                await tts.speak(config.atlas_loop_message)
                return False

            if speak_text:
                await tts.speak(speak_text)
                tools_since_speech.clear()

            working.append(Message(role="assistant", content=speak_text or "…"))
            working.append(Message(
                role="user",
                content=(
                    f"[AUTOMATIC CONTINUATION — round {tool_round}/{config.max_tool_rounds}] "
                    "Execute the planned actions now, or reply directly if everything is done."
                ),
            ))
            continue

        # ── Text reply — exit loop ────────────────────────────────────────────
        break

    reply_text = reply_override if reply_override is not None else _clean(msg.content)

    # Validation guards
    if reply_text.lstrip().startswith("<"):
        logger.warning("Model returned HTML — discarding")
        await tts.speak(config.atlas_error_message)
        return False

    if not reply_text.strip():
        if had_tool_calls:
            working.append(Message(role="user", content="Summarise the result for the user now."))
            resp2 = await ollama.chat(
                model=config.ollama_model, messages=working,
                tools=[], think=think, options=ollama_options,
            )
            reply_text = _clean(resp2.message.content)
            msg = resp2.message
        if not reply_text.strip():
            logger.warning("Empty response — discarding")
            return False

    # Question-sentinel drop guard
    if q_sentinel_rounds > 0 and not question_sentinel_capped:
        if question_sentinel_spoken:
            conversation.append(Message(role="assistant", content=question_sentinel_spoken))
        return True

    conversation.append(msg)
    logger.info("Atlas: %r", reply_text[:120])

    try:
        session_log.append_turn(
            speaker=user.name,
            user_text=user_text,
            tools_called=tools_this_turn,
            reply=reply_text,
        )
    except Exception as exc:
        logger.warning("Session log write failed: %s", exc)

    await tts.speak(reply_text)
    return True


# ── Sleeping mode monitor ─────────────────────────────────────────────────────

async def _sleeping_mode_monitor(
    config: Config,
    db_conn: Any,
    get_last_activity: Any,
    session_log_ref: list[SessionLog],
    tts: TTS,
) -> None:
    """Background task — re-averages embeddings and rotates session after inactivity."""
    last_ran_for: float = 0.0

    while True:
        await asyncio.sleep(30)
        last_active = get_last_activity()
        idle = time.monotonic() - last_active

        if idle >= config.sleep_timeout and last_ran_for < last_active:
            logger.info("💤 Sleeping mode — %.0f s idle", idle)
            try:
                await tts.speak(config.atlas_sleep_message)
            except Exception as exc:
                logger.warning("💤 Sleep message TTS failed: %s", exc)
            try:
                session_log_ref[0].close()
                session_log_ref[0] = SessionLog(config)
                logger.info("💤 Session rotated")
            except Exception as exc:
                logger.warning("💤 Session rotation failed: %s", exc)

            try:
                await recompute_all_embeddings(config, db_conn)
            except Exception as exc:
                logger.warning("💤 Embedding re-average failed: %s", exc)

            last_ran_for = last_active
            logger.info("💤 Sleeping mode complete — standby")


# ── Main loop ─────────────────────────────────────────────────────────────────

async def _main_loop(config: Config, text_mode: bool = False) -> None:
    """Infinite pipeline loop — runs until SIGINT / SIGTERM."""
    think: bool | None = False if config.nothink else None

    logger.info(
        "Atlas starting — model=%s  host=%s  think=%s  text_mode=%s",
        config.ollama_model, config.ollama_host,
        "disabled" if config.nothink else "model-default",
        text_mode,
    )

    # Initialise subsystems
    ollama_client = AsyncClient(host=config.ollama_host)
    db_conn = init_db(config.speaker_db_path)
    mcp = MCPClient(config)
    stt_engine = STT(config)
    tts_engine = TTS(config)
    wake_word_listener = WakeWordListener(config)
    speaker_identifier = SpeakerIdentifier(config, db_conn)

    # Pre-load SpeechBrain
    await warm_up(config)

    # Discover all MCP tool schemas once at startup
    all_tools = await mcp.discover_all_schemas()
    logger.info("Total MCP tools available: %d", len(all_tools))

    # Conversation history — system prompt at index 0
    conversation: list[Message] = [
        Message(role="system", content=_build_system_prompt(GUEST_USER, config, think=think))
    ]
    # Attach tool schemas to the system message so _run_turn can pass them to Ollama
    conversation[0]._tools = all_tools  # type: ignore[attr-defined]

    last_user_state: list[User] = [GUEST_USER]
    session_log_ref: list[SessionLog] = [SessionLog(config)]
    last_activity: list[float] = [time.monotonic()]
    pending_audio: list[Any] = []

    sleep_task = asyncio.create_task(
        _sleeping_mode_monitor(config, db_conn, lambda: last_activity[0], session_log_ref, tts_engine)
    )

    bypass_wake_word = False

    try:
        while True:
            try:
                # 1. Wake word (or stdin in text mode)
                if text_mode:
                    line = await asyncio.get_running_loop().run_in_executor(
                        None, sys.stdin.readline
                    )
                    user_text = line.strip()
                    if not user_text:
                        break
                    audio = None
                    user = GUEST_USER
                else:
                    if bypass_wake_word:
                        logger.info("Listening for follow-up…")
                    else:
                        async for _ in wake_word_listener.listen():
                            break
                        await tts_engine.speak(random.choice(config.wake_ack_phrases))

                    # 2. Record utterance
                    if pending_audio:
                        audio = await pending_audio.pop(0)
                    else:
                        audio = await stt_engine.record_utterance()

                    import numpy as _np  # noqa: PLC0415
                    if audio is None or (_np.ndarray and isinstance(audio, _np.ndarray) and audio.size == 0):
                        bypass_wake_word = False
                        continue

                    # 3. Parallel: identify speaker + transcribe
                    fallback = last_user_state[0] if not last_user_state[0].is_guest else None
                    match, user_text = await asyncio.gather(
                        speaker_identifier.identify(audio, fallback_user=fallback),
                        stt_engine.transcribe(audio),
                    )
                    user = match.user

                    if not last_user_state[0].is_guest or not user.is_guest:
                        if not user.is_guest:
                            last_user_state[0] = user

                    if not user_text:
                        logger.warning("Empty transcription — ignoring")
                        bypass_wake_word = False
                        continue

                    logger.info("Speaker: %s | Said: %r", user.name, user_text)

                # 4. Refresh system prompt for this speaker
                system_msg = Message(
                    role="system",
                    content=_build_system_prompt(user, config, think=think),
                )
                system_msg._tools = all_tools  # type: ignore[attr-defined]
                if conversation and conversation[0].role == "system":
                    conversation[0] = system_msg
                else:
                    conversation.insert(0, system_msg)

                conversation.append(Message(role="user", content=user_text))

                # 5–8. LLM → tools → TTS
                bypass_wake_word = await _run_turn(
                    ollama=ollama_client,
                    mcp=mcp,
                    tts=tts_engine,
                    config=config,
                    conversation=conversation,
                    user=user,
                    user_text=user_text,
                    session_log=session_log_ref[0],
                    think=think,
                    pending_audio=pending_audio,
                    stt=stt_engine,
                )

                if bypass_wake_word:
                    last_activity[0] = time.monotonic()
                elif pending_audio:
                    for task in pending_audio:
                        task.cancel()
                    pending_audio.clear()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Error in main loop: %s", exc, exc_info=True)
                for task in pending_audio:
                    task.cancel()
                pending_audio.clear()
                bypass_wake_word = False
                await asyncio.sleep(1.0)

    finally:
        sleep_task.cancel()
        try:
            await sleep_task
        except asyncio.CancelledError:
            pass
        try:
            session_log_ref[0].close()
        except Exception as exc:
            logger.warning("Session log close failed: %s", exc)
        for task in pending_audio:
            task.cancel()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point — ``atlas`` console script."""
    parser = argparse.ArgumentParser(
        description="Atlas — local-first AI voice assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nothink", action="store_true", default=False,
        help="Disable chain-of-thought tokens (faster responses). "
             "Can also be set via NOTHINK=true in .env.",
    )
    parser.add_argument(
        "--text", action="store_true", default=False,
        help="Text debug mode — bypass wake word and STT, read turns from stdin.",
    )
    parser.add_argument(
        "--check", action="store_true", default=False,
        help="Run startup health check and exit.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"\n[Atlas] Configuration error:\n  {exc}\n", file=sys.stderr)
        sys.exit(1)

    # Override nothink from CLI flag
    if args.nothink:
        import dataclasses  # noqa: PLC0415
        config = dataclasses.replace(config, nothink=True)

    if config.log_file:
        _setup_file_logging(config.log_file)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run() -> None:
        try:
            await run_health_check(config)
        except HealthCheckError as exc:
            print(f"\n[Atlas] {exc}\n", file=sys.stderr)
            sys.exit(1)

        if args.check:
            print("[Atlas] All checks passed.")
            return

        main_task = asyncio.ensure_future(_main_loop(config, text_mode=args.text))

        def _shutdown(sig: signal.Signals) -> None:
            logger.info("Received %s — shutting down", sig.name)
            main_task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown, sig)

        try:
            await main_task
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Atlas stopped.")

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
