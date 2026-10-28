"""Step 4 voice evaluation on the real ConversationAgent.

Each clip runs:

    Gnani STT (en-IN) -> ConversationAgent.respond(..., input_modality="voice")
    -> Gnani TTS (voice Kaveri, model timbre-v2.5) -> outputs/step4_voice/audio/

The live run requires RUN_LIVE=1. Importing this module or running the tests
never constructs a client and never makes a Gemini or Gnani call.

Usage (PowerShell):

    $env:RUN_LIVE = "1"
    .\\venv\\Scripts\\python.exe eval\\run_voice.py

A thin recorder wraps src.rag.answer_question only to capture the is_asr flag
and retrieved chunk IDs; every call delegates to the real function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CLIP_DIR = _REPO_ROOT / "voice_clips"
OUT_DIR = _REPO_ROOT / "outputs" / "step4_voice"
AUDIO_DIR = OUT_DIR / "audio"
TRANSCRIPTS_PATH = OUT_DIR / "transcripts.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"
TICKETS_PATH = OUT_DIR / "tickets_created.csv"
COMMAND = r".\venv\Scripts\python.exe eval\run_voice.py"

SOURCE_FILES = [
    "src/conversation.py",
    "src/rag.py",
    "src/retrieval.py",
    "src/llm.py",
    "src/order_tools.py",
]

MIN_GEMINI_INTERVAL_SECONDS = 4.5
QUOTA_RETRY_SECONDS = 65
MAX_QUOTA_RETRIES = 3

# The six clips. Clips 01 and 03 replay prior turns as text so the clip lands on
# the intended fixed-flow step. Clips 04-06 are single-turn policy probes whose
# expected decisions are copied from the Step 2 frozen outputs named in
# expected_source. Clip 02 has no valid dataset row and is not compared.
CLIPS: List[Dict[str, Any]] = [
    {
        "name": "01_pincode_verify",
        "tts_language": "en-IN",
        "setup_turns": [
            {"text": "Hello, I need help with an order.", "input_modality": "text"},
            {"text": "My order number is 69520867.", "input_modality": "text"},
        ],
        "expected_decision": "answer",
        "expected_source": "IDV-2 verification of order 69520867 with clip pincode 160017",
        "checks": {"order_id": "69520867", "verified": True},
    },
    {
        "name": "02_damaged_compensation",
        "tts_language": "hinglish",
        "setup_turns": [],
        "expected_decision": None,
        "expected_source": "no valid row; PC015/PC016 T4 do not match this damaged-compensation demand",
    },
    {
        "name": "03_human_agent",
        "tts_language": "en-IN",
        "setup_turns": [
            {"text": "Hi, I need some help with an order.", "input_modality": "text"},
            {"text": "The order number is 6649 2475.", "input_modality": "text"},
            {"text": "The last four digits are 6073.", "input_modality": "text"},
        ],
        "expected_decision": "escalate",
        "expected_source": "conversations.csv PC006 T1-T3 replayed as text; clip is T4 (human request)",
    },
    {
        "name": "04_hinglish_return",
        "tts_language": "hinglish",
        "setup_turns": [],
        "expected_decision": "answer",
        "expected_source": "outputs/step2_frozen_holdout_text.csv PQ0014 (single turn)",
    },
    {
        "name": "05_sku_warranty",
        "tts_language": "hinglish",
        "setup_turns": [],
        "expected_decision": "answer",
        "expected_source": "outputs/step2_frozen_dev_asr.csv PQ0002 (single turn)",
    },
    {
        "name": "06_credit_refund",
        "tts_language": "hinglish",
        "setup_turns": [],
        "expected_decision": "answer",
        "expected_source": "outputs/step2_frozen_holdout_text.csv PQ0008 (single turn)",
    },
]


def sha256_file(rel_path: str) -> str:
    path = _REPO_ROOT / rel_path
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_agent(retriever: Any, state: Any = None) -> Any:
    """Build the real ConversationAgent with the production voice defaults."""
    from src.conversation import ConversationAgent

    return ConversationAgent(
        state=state,
        answer_adapter=None,
        retriever=retriever,
        top_k=10,
        enable_rewrite=False,
        input_modality="voice",
    )


def run_agent_turn(
    agent: Any,
    transcript: str,
    input_modality: str = "voice",
) -> Dict[str, Any]:
    """Single agent call; voice turns must go through input_modality='voice'."""
    return agent.respond(transcript, input_modality=input_modality)


def _redact(text: str) -> str:
    from src import llm

    key = getattr(llm, "_API_KEY", None)
    if key:
        return str(text).replace(key, "[REDACTED]")
    return str(text)


def _install_gemini_throttle(counters: Dict[str, int]) -> Any:
    """Throttle Gemini to 4.5s between calls and bound 429 retries."""
    import src.rag as rag
    from src.llm import QuotaExceededError

    real_complete = rag.complete_with_meta
    last_call = [0.0]

    def throttled(system: str, user: str) -> Any:
        for attempt in range(MAX_QUOTA_RETRIES + 1):
            wait = MIN_GEMINI_INTERVAL_SECONDS - (time.monotonic() - last_call[0])
            if wait > 0:
                time.sleep(wait)
            try:
                result = real_complete(system, user)
                last_call[0] = time.monotonic()
                counters["gemini"] += 1
                return result
            except QuotaExceededError:
                last_call[0] = time.monotonic()
                counters["gemini_quota_errors"] += 1
                if attempt >= MAX_QUOTA_RETRIES:
                    raise
                time.sleep(QUOTA_RETRY_SECONDS)
        raise RuntimeError("unreachable Gemini retry state")

    rag.complete_with_meta = throttled
    return real_complete


def _install_answer_recorder(trace: List[Dict[str, Any]]) -> Any:
    """Wrap answer_question to capture is_asr and retrieved IDs; delegate all."""
    import src.rag as rag

    real_answer_question = rag.answer_question

    def recording_answer_question(**kwargs: Any) -> Dict[str, Any]:
        result = real_answer_question(**kwargs)
        trace.append(
            {
                "is_asr": kwargs.get("is_asr"),
                "retrieved_ids": [
                    str(chunk_id)
                    for chunk_id, _score in result.get("retrieved_items", [])
                ],
            }
        )
        return result

    rag.answer_question = recording_answer_question
    return real_answer_question


def _install_gnani_counters(counters: Dict[str, int]) -> None:
    """Count actual Gnani HTTP attempts, including internal retries."""
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient

    real_stt = GnaniSTTClient.transcribe
    real_tts = GnaniTTSClient.synthesize

    def counted_stt(self: Any, *args: Any, **kwargs: Any) -> Any:
        counters["gnani_stt"] += 1
        return real_stt(self, *args, **kwargs)

    def counted_tts(self: Any, *args: Any, **kwargs: Any) -> Any:
        counters["gnani_tts"] += 1
        return real_tts(self, *args, **kwargs)

    GnaniSTTClient.transcribe = counted_stt  # type: ignore[method-assign]
    GnaniTTSClient.synthesize = counted_tts  # type: ignore[method-assign]


def _finalise_wav_header(path: Path) -> None:
    """Patch placeholder RIFF/data sizes written by the Gnani streaming client.

    Gnani returns 0xFFFFFFFF in both size fields; the PCM payload is complete.
    Rewriting only the container sizes makes the duration readable and keeps
    the audio bytes untouched.
    """
    raw = bytearray(path.read_bytes())
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return

    pos = 12
    data_size_offset: Optional[int] = None
    data_start: Optional[int] = None
    while pos + 8 <= len(raw):
        chunk_id = bytes(raw[pos : pos + 4])
        chunk_size = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        if chunk_id == b"data":
            data_size_offset = pos + 4
            data_start = pos + 8
            break
        if chunk_size == 0xFFFFFFFF:
            break
        pos += 8 + chunk_size + (chunk_size & 1)

    if data_size_offset is None or data_start is None:
        return

    actual_data = len(raw) - data_start
    raw[data_size_offset : data_size_offset + 4] = actual_data.to_bytes(4, "little")
    raw[4:8] = (len(raw) - 8).to_bytes(4, "little")
    path.write_bytes(bytes(raw))


def process_clip(
    clip: Dict[str, Any],
    retriever: Any,
    trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one clip end to end and return its transcript record."""
    from src import voice
    from src.voice import get_audio_duration_seconds

    clip_path = CLIP_DIR / f"{clip['name']}.aac"
    tts_path = AUDIO_DIR / f"{clip['name']}.wav"
    if tts_path.exists():
        tts_path.unlink()

    record: Dict[str, Any] = {
        "clip": clip_path.name,
        "input_modality": "voice",
        "stt_transcript": "",
        "decision": "error",
        "reply": "",
        "used_sections": [],
        "is_asr_seen_by_answer_question": None,
        "retrieved_chunk_ids": [],
        "rag_called": False,
        "expected_decision": clip["expected_decision"],
        "expected_source": clip["expected_source"],
        "decision_correct": None,
        "parsed_order_id": None,
        "verified": None,
        "state_checks_passed": None,
        "tts_file": None,
        "tts_duration_seconds": None,
        "tts_ok": False,
        "stt_ok": False,
        "error": None,
    }

    started = time.perf_counter()

    # 1. Gnani STT, en-IN.
    try:
        transcript = voice.transcribe(clip_path, language_code="en-IN")
        record["stt_transcript"] = transcript or ""
        record["stt_ok"] = bool((transcript or "").strip())
        if not record["stt_ok"]:
            record["error"] = "STT returned empty transcript"
    except Exception as exc:  # noqa: BLE001 - record the real failure
        record["error"] = _redact(f"STT failed: {type(exc).__name__}: {exc}")

    transcript = record["stt_transcript"].strip()

    # 2. Setup turns (text) and then the real agent on the voice transcript.
    agent_state: Dict[str, Any] = {}
    if transcript and not record["error"]:
        agent = build_agent(retriever)
        try:
            for setup in clip.get("setup_turns", []):
                run_agent_turn(
                    agent,
                    setup["text"],
                    input_modality=setup["input_modality"],
                )
            trace.clear()
            result = run_agent_turn(agent, transcript, input_modality="voice")
            agent_state = result.get("state") or {}
            record["decision"] = str(result.get("decision", ""))
            record["reply"] = str(result.get("reply", ""))
            record["used_sections"] = list(result.get("used_sections", []) or [])
        except Exception as exc:  # noqa: BLE001 - record the real failure
            record["decision"] = "error"
            record["error"] = _redact(
                f"Agent error: {type(exc).__name__}: {exc}"
            )

        if trace:
            record["rag_called"] = True
            record["is_asr_seen_by_answer_question"] = trace[-1]["is_asr"]
            record["retrieved_chunk_ids"] = trace[-1]["retrieved_ids"]

    # 3. Gnani TTS: Kaveri, timbre-v2.5, Hinglish -> auto.
    reply = record["reply"].strip()
    if reply:
        try:
            spoken = voice.normalise_for_speech(reply)
            voice.speak(
                spoken,
                language_code=clip["tts_language"],
                out_path=tts_path,
            )
            if tts_path.exists() and tts_path.stat().st_size > 0:
                _finalise_wav_header(tts_path)
                record["tts_file"] = str(
                    tts_path.relative_to(_REPO_ROOT)
                ).replace("\\", "/")
                record["tts_duration_seconds"] = get_audio_duration_seconds(
                    tts_path
                )
                record["tts_ok"] = True
            else:
                raise RuntimeError("TTS produced no audio")
        except Exception as exc:  # noqa: BLE001 - record the real failure
            message = f"TTS failed: {type(exc).__name__}: {exc}"
            record["error"] = (
                f"{record['error']}; {message}" if record["error"] else message
            )

    expected = clip["expected_decision"]
    if expected is None:
        record["decision_correct"] = None
    else:
        record["decision_correct"] = record["decision"] == expected

    checks = clip.get("checks") or {}
    if checks:
        record["parsed_order_id"] = agent_state.get("order_id")
        record["verified"] = agent_state.get("verified")
        record["state_checks_passed"] = bool(
            record["parsed_order_id"] == checks.get("order_id")
            and record["verified"] is checks.get("verified")
        )

    record["clip_pass"] = bool(
        record["tts_ok"]
        and record["error"] is None
        and (record["decision_correct"] is not False)
        and (record["state_checks_passed"] is not False)
    )
    record["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    return record


def run_live() -> int:
    if os.getenv("RUN_LIVE") != "1":
        raise RuntimeError("run_live requires RUN_LIVE=1")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Redirect the dataset mock tool's ticket log away from the dataset.
    import src.order_tools  # noqa: F401 - puts DATASET_DIR on sys.path
    import tools.tools as dataset_tools

    dataset_tools._TICKETS = str(TICKETS_PATH)
    if TICKETS_PATH.exists():
        TICKETS_PATH.unlink()

    from src.data import DATASET_DIR
    from src.retrieval import HybridRetriever

    started_at = now_iso()
    source_hashes = {rel: sha256_file(rel) for rel in SOURCE_FILES}

    counters: Dict[str, int] = {
        "gemini": 0,
        "gemini_quota_errors": 0,
        "gnani_stt": 0,
        "gnani_tts": 0,
    }
    trace: List[Dict[str, Any]] = []

    real_complete = _install_gemini_throttle(counters)
    real_answer_question = _install_answer_recorder(trace)
    _install_gnani_counters(counters)

    shared_retriever = HybridRetriever(use_query_rewrite=False)
    records: List[Dict[str, Any]] = []

    try:
        for clip in CLIPS:
            print(f"Processing {clip['name']}...", flush=True)
            records.append(process_clip(clip, shared_retriever, trace))
    finally:
        import src.rag as rag

        rag.answer_question = real_answer_question
        rag.complete_with_meta = real_complete

    finished_at = now_iso()

    with TRANSCRIPTS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    tools_dir = Path(DATASET_DIR) / "tools"
    dataset_tool_files = (
        sorted(p.name for p in tools_dir.iterdir() if p.is_file())
        if tools_dir.exists()
        else []
    )

    stt_errors = sum(1 for r in records if not r["stt_ok"])
    tts_errors = sum(1 for r in records if not r["tts_ok"])
    decision_errors = sum(
        1 for r in records if r["decision_correct"] is False
    )
    state_check_errors = sum(
        1 for r in records if r["state_checks_passed"] is False
    )
    passes = sum(1 for r in records if r["clip_pass"])

    manifest = {
        "start_time": started_at,
        "end_time": finished_at,
        "command": COMMAND,
        "run_live_gate": "RUN_LIVE=1",
        "source_hashes": source_hashes,
        "production_defaults": {
            "agent": "src.conversation.ConversationAgent",
            "answer_adapter": "default rag_answer_adapter",
            "retriever": "one shared HybridRetriever(use_query_rewrite=False)",
            "top_k": 10,
            "enable_rewrite": False,
            "input_modality": "voice",
            "stt_language": "en-IN",
            "tts_voice": "Kaveri",
            "tts_model": "timbre-v2.5",
            "hinglish_tts_language": "auto",
        },
        "api_calls_by_service": {
            "gemini": counters["gemini"],
            "gemini_quota_errors": counters["gemini_quota_errors"],
            "gnani_stt": counters["gnani_stt"],
            "gnani_tts": counters["gnani_tts"],
            "gnani_total": counters["gnani_stt"] + counters["gnani_tts"],
        },
        "rate_limiting": {
            "min_gemini_interval_seconds": MIN_GEMINI_INTERVAL_SECONDS,
            "quota_retry_seconds": QUOTA_RETRY_SECONDS,
            "max_quota_retries": MAX_QUOTA_RETRIES,
        },
        "clips": [
            {
                "name": clip["name"],
                "tts_language_requested": clip["tts_language"],
                "setup_turns": clip.get("setup_turns", []),
                "expected_decision": clip["expected_decision"],
                "expected_source": clip["expected_source"],
                "checks": clip.get("checks"),
            }
            for clip in CLIPS
        ],
        "counts": {
            "clips": len(records),
            "pass": passes,
            "stt_errors": stt_errors,
            "tts_errors": tts_errors,
            "decision_errors": decision_errors,
            "state_check_errors": state_check_errors,
        },
        "dataset_tools_folder": {
            "path": str(tools_dir),
            "files": dataset_tool_files,
            "only_tools_py_and_init": dataset_tool_files
            == ["__init__.py", "tools.py"],
        },
        "ticket_log_redirected_to": str(
            TICKETS_PATH.relative_to(_REPO_ROOT)
        ).replace("\\", "/"),
        "instrumentation": (
            "src.rag.answer_question wrapped only to record is_asr and "
            "retrieved_items; all calls delegated to the real function."
        ),
    }

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\nSTEP 4 VOICE SUMMARY")
    for record in records:
        print(
            f"{record['clip']} | stt={'ok' if record['stt_ok'] else 'fail'} | "
            f"decision={record['decision']} | "
            f"expected={record['expected_decision']} | "
            f"state_checks={record['state_checks_passed']} | "
            f"tts={'ok' if record['tts_ok'] else 'fail'} | "
            f"pass={record['clip_pass']}"
        )
    print(
        f"counts: pass={passes}/{len(records)} stt_errors={stt_errors} "
        f"tts_errors={tts_errors} decision_errors={decision_errors} "
        f"state_check_errors={state_check_errors}"
    )
    print(
        f"api: gemini={counters['gemini']} "
        f"gnani_stt={counters['gnani_stt']} gnani_tts={counters['gnani_tts']}"
    )
    print(f"dataset tools/: {dataset_tool_files}")
    print(f"wrote {TRANSCRIPTS_PATH}")
    print(f"wrote {MANIFEST_PATH}")
    return 0 if passes == len(records) else 1


def main() -> int:
    if os.getenv("RUN_LIVE") != "1":
        print(
            "SKIP: eval/run_voice.py requires RUN_LIVE=1; "
            "no ConversationAgent constructed and no Gemini or Gnani call made."
        )
        return 0

    parser = argparse.ArgumentParser(
        description="Run the six Step 4 voice clips on the real agent"
    )
    parser.parse_args()
    return run_live()


if __name__ == "__main__":
    raise SystemExit(main())
