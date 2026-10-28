"""Step 3 12-turn smoke runner for Phase 3C scenarios.

Uses the real ConversationAgent, the real default rag_answer_adapter, and one
shared HybridRetriever. A thin recorder wraps src.rag.answer_question only to
capture retrieved chunk IDs and API-call deltas; it delegates to the real
function and does not stub the answer.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.conversation import ConversationAgent
from src.data import DATASET_DIR
from src.retrieval import HybridRetriever
from src.scoring import score_row
from eval.score_step3 import match_tool_calls
import src.llm as llm
import src.rag as rag

OUT_DIR = REPO_ROOT / "outputs" / "step3_smoke"
TRANSCRIPTS_PATH = OUT_DIR / "transcripts.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"

SCENARIOS = ["PC021", "PC015", "PC019"]
COMMAND = r".\venv\Scripts\python.exe outputs\step3_smoke\run_smoke.py"

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
    "outputs/step3_smoke/run_smoke.py",
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

    started_at = now_iso()
    api_start = rag.API_CALL_COUNT
    source_hashes = {rel: sha256_file(rel) for rel in SOURCE_FILES}
    supporting_hashes = {rel: sha256_file(rel) for rel in SUPPORTING_FILES}

    # Recorder: delegates to the real answer_question, captures retrieved IDs.
    collected_chunks: List[List[str]] = []
    real_answer_question = rag.answer_question

    def recording_answer_question(**kwargs: Any) -> Dict[str, Any]:
        result = real_answer_question(**kwargs)
        collected_chunks.append(
            [str(chunk_id) for chunk_id, _score in result.get("retrieved_items", [])]
        )
        return result

    rag.answer_question = recording_answer_question

    conversations_path = DATASET_DIR / "practice" / "conversations.csv"
    gt_df = pd.read_csv(conversations_path, dtype=str, keep_default_na=False)

    shared_retriever = HybridRetriever(use_query_rewrite=False)

    transcripts: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    exceptions: List[Dict[str, Any]] = []
    scenario_summary: Dict[str, Dict[str, int]] = {}

    for cid in SCENARIOS:
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
        "scenarios": scenario_summary,
        "total_turns": total_turns,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "failures": failures,
        "exceptions": exceptions,
    }

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("=" * 72)
    print("STEP 3 SMOKE SUMMARY (12 PHASE 3C TURNS, TEXT)")
    print("=" * 72)
    for cid in SCENARIOS:
        item = scenario_summary[cid]
        print(
            f"{cid}: {item['passed']}/{item['turns']} passed "
            f"({item['failed']} failed)"
        )
    print(f"TOTAL: {total_passed}/{total_turns} passed ({total_failed} failed)")
    print(f"TOTAL API CALLS: {total_api_calls}")
    print(f"EXCEPTIONS: {len(exceptions)}")
    for exc in exceptions:
        print(f"  EXCEPTION {exc['scenario']} T{exc['turn']}: {exc['error']}")

    if failures:
        print("\nFAILING TURNS:")
        for fail in failures:
            print(
                f"- {fail['scenario']} T{fail['turn']}: "
                f"decision={fail['decision']} "
                f"(expected {fail['expected_decision']}), "
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
    return 0 if total_failed == 0 and not exceptions else 1


if __name__ == "__main__":
    raise SystemExit(main())
