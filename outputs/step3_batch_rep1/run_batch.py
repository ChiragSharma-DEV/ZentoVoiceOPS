"""Step 3 96-turn batch runner.

Runs every practice conversation with the real ConversationAgent, the real
default rag_answer_adapter, text modality, and one shared HybridRetriever.
A thin recorder wraps src.rag.answer_question only to capture retrieved chunk
IDs and API-call deltas; all answer calls delegate to the real function.

The dataset ticket log is redirected to outputs/step3_batch/tickets_created.csv
so the batch does not write into the dataset folder.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

import src.order_tools  # noqa: F401 - ensures DATASET_DIR is on sys.path
import tools.tools as dataset_tools
from src.conversation import ConversationAgent
from src.data import DATASET_DIR
from src.retrieval import HybridRetriever
from src.scoring import score_row
from eval.score_step3 import match_tool_calls
import src.llm as llm
from src.llm import QuotaExceededError
import src.rag as rag

OUT_DIR = REPO_ROOT / "outputs" / "step3_batch_rep1"
TRANSCRIPTS_PATH = OUT_DIR / "transcripts.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"
TICKETS_PATH = OUT_DIR / "tickets_created.csv"
COMMAND = r".\venv\Scripts\python.exe outputs\step3_batch_rep1\run_batch.py"
MIN_API_INTERVAL_SECONDS = 4.5
QUOTA_RETRY_SECONDS = 65
MAX_QUOTA_RETRIES = 5

SOURCE_FILES = [
    "src/conversation.py",
    "src/rag.py",
    "src/retrieval.py",
    "src/llm.py",
    "src/order_tools.py",
]
SUPPORTING_FILES = [
    "src/scoring.py",
    "src/data.py",
    "eval/score_step3.py",
    "tests/test_phase3c_offline.py",
    "outputs/step3_batch_rep1/run_batch.py",
]


def sha256_file(rel_path: str) -> str:
    path = REPO_ROOT / rel_path
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def redact(text: str) -> str:
    """Remove the API key from any error text before printing or writing it."""
    key = getattr(llm, "_API_KEY", None)
    if key:
        return str(text).replace(key, "[REDACTED]")
    return str(text)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Redirect the mock tool's append-only ticket log away from the dataset.
    dataset_tools._TICKETS = str(TICKETS_PATH)
    if TICKETS_PATH.exists():
        TICKETS_PATH.unlink()

    started_at = now_iso()
    api_start = rag.API_CALL_COUNT
    source_hashes = {rel: sha256_file(rel) for rel in SOURCE_FILES}
    supporting_hashes = {rel: sha256_file(rel) for rel in SUPPORTING_FILES}

    collected_chunks: List[List[str]] = []
    quota_retry_events: List[Dict[str, Any]] = []
    last_attempt_at = [0.0]
    real_answer_question = rag.answer_question

    def recording_answer_question(**kwargs: Any) -> Dict[str, Any]:
        for attempt in range(MAX_QUOTA_RETRIES + 1):
            wait_seconds = MIN_API_INTERVAL_SECONDS - (
                time.monotonic() - last_attempt_at[0]
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            try:
                result = real_answer_question(**kwargs)
                last_attempt_at[0] = time.monotonic()
                collected_chunks.append(
                    [
                        str(chunk_id)
                        for chunk_id, _score in result.get("retrieved_items", [])
                    ]
                )
                return result
            except QuotaExceededError as exc:
                last_attempt_at[0] = time.monotonic()
                quota_retry_events.append(
                    {"attempt": attempt + 1, "error": redact(str(exc))}
                )
                if attempt >= MAX_QUOTA_RETRIES:
                    raise
                time.sleep(QUOTA_RETRY_SECONDS)
        raise RuntimeError("unreachable quota retry state")

    rag.answer_question = recording_answer_question

    conversations_path = DATASET_DIR / "practice" / "conversations.csv"
    gt_df = pd.read_csv(conversations_path, dtype=str, keep_default_na=False)
    conversation_ids = list(dict.fromkeys(gt_df["conversation_id"].tolist()))

    shared_retriever = HybridRetriever(use_query_rewrite=False)

    transcripts: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    exceptions: List[Dict[str, Any]] = []
    scenario_summary: Dict[str, Dict[str, int]] = {}
    abort_reason: str | None = None

    for cid in conversation_ids:
        if abort_reason is not None:
            break

        agent = ConversationAgent(retriever=shared_retriever)
        rows = gt_df[gt_df["conversation_id"] == cid].copy()
        rows["turn_num"] = rows["turn"].astype(int)
        rows = rows.sort_values("turn_num")

        passed = 0
        failed = 0

        for _, row in rows.iterrows():
            turn = int(row["turn"])
            input_text = row["customer_message"]
            expected_decision = row["expected_decision"]
            must_include = row.get("must_include", "")
            must_not_include = row.get("must_not_include", "")
            expected_tool_calls = row.get("expected_tool_calls", "")

            collected_chunks.clear()
            api_before = rag.API_CALL_COUNT
            error: str | None = None

            try:
                result = agent.respond(input_text, input_modality="text")
            except Exception as exc:  # noqa: BLE001 - record the real failure
                error = redact(f"{type(exc).__name__}: {exc}")
                result = {
                    "decision": "ERROR",
                    "reply": "",
                    "used_sections": [],
                    "tool_calls": [],
                    "state": agent.state.to_dict(),
                }
                exceptions.append(
                    {"scenario": cid, "turn": turn, "error": error}
                )
                if "QuotaExceeded" in type(exc).__name__ or "429" in error:
                    abort_reason = error

            api_after = rag.API_CALL_COUNT
            api_calls = api_after - api_before

            decision = str(result.get("decision", "ERROR"))
            reply = str(result.get("reply", ""))
            used_sections = list(result.get("used_sections", []) or [])
            tool_calls = list(result.get("tool_calls", []) or [])
            retrieved_chunk_ids = (
                list(collected_chunks[-1]) if collected_chunks else []
            )

            gold = {
                "id": f"{cid}_T{turn}",
                "question": input_text,
                "acceptable_decisions": expected_decision,
                "must_include": must_include,
                "must_not_include": must_not_include,
            }
            scored = score_row(gold, {"decision": decision, "reply": reply})
            decision_correct = bool(scored["decision_correct"])
            must_include_pass = bool(scored["must_include_pass"])
            must_not_include_violations = list(
                scored.get("must_not_include_violations", [])
            )
            must_not_pass = len(must_not_include_violations) == 0
            tool_ok, tool_error = match_tool_calls(expected_tool_calls, tool_calls)

            turn_pass = (
                error is None
                and decision_correct
                and must_include_pass
                and must_not_pass
                and tool_ok
            )

            record = {
                "scenario": cid,
                "turn": turn,
                "input": input_text,
                "decision": decision,
                "reply": reply,
                "used_sections": used_sections,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "must_include_pass": must_include_pass,
                "must_not_pass": must_not_pass,
                "api_calls": api_calls,
                "decision_correct": decision_correct,
                "tool_calls_correct": tool_ok,
                "tool_call_error": tool_error,
                "missing_facts": list(scored.get("missing_facts", [])),
                "must_not_include_violations": must_not_include_violations,
                "turn_pass": turn_pass,
                "error": error,
            }
            transcripts.append(record)

            if turn_pass:
                passed += 1
            else:
                failed += 1
                failures.append(
                    {
                        "scenario": cid,
                        "turn": turn,
                        "input": input_text,
                        "decision": decision,
                        "expected_decision": expected_decision,
                        "reply": reply,
                        "used_sections": used_sections,
                        "decision_correct": decision_correct,
                        "must_include_pass": must_include_pass,
                        "missing_facts": record["missing_facts"],
                        "must_not_pass": must_not_pass,
                        "must_not_include_violations": must_not_include_violations,
                        "tool_calls_correct": tool_ok,
                        "tool_call_error": tool_error,
                        "error": error,
                    }
                )

            if abort_reason is not None:
                break

        scenario_summary[cid] = {
            "turns": int(len(rows)),
            "passed": passed,
            "failed": failed,
        }

    rag.answer_question = real_answer_question
    finished_at = now_iso()
    total_api_calls = rag.API_CALL_COUNT - api_start

    with TRANSCRIPTS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in transcripts:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    failure_counts = {
        "decision_mismatch": sum(
            1 for record in transcripts if not record["decision_correct"]
        ),
        "missing_must_include": sum(
            1 for record in transcripts if not record["must_include_pass"]
        ),
        "must_not_violation": sum(
            1 for record in transcripts if not record["must_not_pass"]
        ),
        "tool_call_mismatch": sum(
            1 for record in transcripts if not record["tool_calls_correct"]
        ),
        "exception": sum(1 for record in transcripts if record["error"] is not None),
    }

    total_turns = len(transcripts)
    total_passed = sum(item["passed"] for item in scenario_summary.values())
    total_failed = sum(item["failed"] for item in scenario_summary.values())

    manifest = {
        "start_time": started_at,
        "end_time": finished_at,
        "command": COMMAND,
        "total_api_calls": total_api_calls,
        "api_call_counter_start": api_start,
        "api_call_counter_end": rag.API_CALL_COUNT,
        "production_defaults": {
            "top_k": 10,
            "enable_rewrite": False,
            "use_query_rewrite": False,
            "input_modality": "text",
            "adapter": "default rag_answer_adapter",
            "retriever": "one shared HybridRetriever(use_query_rewrite=False)",
        },
        "source_hashes": source_hashes,
        "supporting_hashes": supporting_hashes,
        "instrumentation": (
            "src.rag.answer_question wrapped only to record retrieved_items; "
            "all calls delegated to the real function."
        ),
        "rate_limiting": {
            "min_api_interval_seconds": MIN_API_INTERVAL_SECONDS,
            "quota_retry_seconds": QUOTA_RETRY_SECONDS,
            "max_quota_retries_per_call": MAX_QUOTA_RETRIES,
            "quota_retries_observed": len(quota_retry_events),
        },
        "dataset_ticket_log_redirected_to": str(
            TICKETS_PATH.relative_to(REPO_ROOT)
        ),
        "scenario_count": len(scenario_summary),
        "scenarios": scenario_summary,
        "total_turns": total_turns,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "failure_counts_by_type": failure_counts,
        "failures": failures,
        "exceptions": exceptions,
        "aborted": abort_reason is not None,
        "abort_reason": abort_reason,
    }

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("=" * 72)
    print("STEP 3 96-TURN BATCH SUMMARY")
    print("=" * 72)
    for cid in conversation_ids:
        if cid not in scenario_summary:
            print(f"{cid}: NOT RUN (batch aborted)")
            continue
        item = scenario_summary[cid]
        print(
            f"{cid}: {item['passed']}/{item['turns']} passed "
            f"({item['failed']} failed)"
        )
    print(f"TOTAL: {total_passed}/{total_turns} passed ({total_failed} failed)")
    print(
        "FAILURE COUNTS BY TYPE: "
        + json.dumps(failure_counts, sort_keys=True)
    )
    print(f"TOTAL API CALLS: {total_api_calls}")
    print(f"QUOTA RETRIES OBSERVED: {len(quota_retry_events)}")
    print(f"EXCEPTIONS: {len(exceptions)}")
    for exc in exceptions:
        print(f"  EXCEPTION {exc['scenario']} T{exc['turn']}: {exc['error']}")
    if abort_reason is not None:
        print(f"BATCH ABORTED: {abort_reason}")

    if failures:
        print("\nFAILING TURNS:")
        for fail in failures:
            print(
                f"- {fail['scenario']} T{fail['turn']}: "
                f"decision={fail['decision']} "
                f"(expected {fail['expected_decision']}), "
                f"decision_correct={fail['decision_correct']}, "
                f"must_include_pass={fail['must_include_pass']}, "
                f"must_not_pass={fail['must_not_pass']}, "
                f"tool_calls_correct={fail['tool_calls_correct']}"
            )
            print(f"  REPLY: {fail['reply']}")
            if fail["error"]:
                print(f"  ERROR: {fail['error']}")
    else:
        print("\nFAILING TURNS: none")

    print(f"\nWROTE: {TRANSCRIPTS_PATH}")
    print(f"WROTE: {MANIFEST_PATH}")
    return 0 if total_failed == 0 and not exceptions and abort_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
