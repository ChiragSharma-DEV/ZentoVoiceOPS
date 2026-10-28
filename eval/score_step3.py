"""Scoring module for VoiceOps 2026 Step 3 multi-turn conversations.

Evaluates turn-by-turn and conversation-level accuracy:
- decision_correct: decision equals expected_decision
- must_include & must_not_include: uses score_row from src.scoring
- tool_calls_correct: matches expected_tool_calls by tool name and arguments
A conversation passes only if every turn passes.
Saves outputs/step3_scores.json.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from src.data import DATASET_DIR
from src.scoring import score_row, check_must_include, check_must_not_include

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval.score_step3")


def parse_tool_call_spec(spec: str) -> Optional[Dict[str, Any]]:
    """Parse string representation of expected tool call.

    e.g. 'lookup_order(order_id=89782837)'
    or   'lookup_order(order_id=89782837, verification_value=4533)'
    or   'create_ticket()'
    """
    spec_clean = str(spec or "").strip()
    if not spec_clean or spec_clean.lower() in ("nan", "none", ""):
        return None

    m = re.match(r"^(\w+)\((.*)\)$", spec_clean)
    if not m:
        return None

    name = m.group(1).strip()
    args_str = m.group(2).strip()
    args: Dict[str, str] = {}

    if args_str:
        # Split on comma
        parts = [p.strip() for p in args_str.split(",") if p.strip()]
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                args[k.strip()] = v.strip().strip("'\"")

    return {"name": name, "arguments": args}


def match_tool_calls(
    expected_spec: Optional[str],
    calls_made: Any,
) -> Tuple[bool, str]:
    """Check if tool calls made match expected tool call specification."""
    expected = parse_tool_call_spec(expected_spec)

    # Normalize calls_made to a list of dicts
    if isinstance(calls_made, str):
        try:
            calls_list = json.loads(calls_made)
        except Exception:
            calls_list = []
    elif isinstance(calls_made, list):
        calls_list = calls_made
    else:
        calls_list = []

    # Case 1: No tool call expected
    if expected is None:
        if len(calls_list) == 0:
            return True, ""
        return False, f"Expected no tool calls, but made {len(calls_list)}"

    # Case 2: Tool call expected
    if len(calls_list) == 0:
        return False, f"Expected {expected['name']}({expected['arguments']}), but made none"

    # Find matching call
    exp_name = expected["name"]
    exp_args = expected["arguments"]

    matched = False
    for call in calls_list:
        if call.get("name") != exp_name:
            continue
        call_args = call.get("arguments", {})

        # Convert tool-call arguments to strings on both sides before comparing (order_id and verification_value)
        # to ensure type neutrality across integer and string representations (e.g. order_id 77461439 as int vs
        # string "77461439", and preserving leading zeros in verification_value like "0783").
        args_match = True
        for k, v in exp_args.items():
            call_v = str(call_args.get(k, "")).strip().strip("'\"")
            exp_v = str(v).strip().strip("'\"")
            if call_v != exp_v:
                args_match = False
                break

        if args_match:
            matched = True
            break

    if matched:
        return True, ""
    return False, f"No tool call matched expected {exp_name}({exp_args})"


def evaluate_step3(
    results_csv: str | Path = "outputs/step3_results.csv",
    ground_truth_csv: str | Path | None = None,
    output_json: str | Path = "outputs/step3_scores.json",
) -> Dict[str, Any]:
    """Score step3 results against ground truth."""
    res_path = Path(results_csv)
    if not res_path.is_absolute():
        res_path = _REPO_ROOT / res_path
    if not res_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {res_path}")

    gt_path = Path(ground_truth_csv) if ground_truth_csv else DATASET_DIR / "practice/conversations.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {gt_path}")

    res_df = pd.read_csv(res_path, dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(gt_path, dtype=str, keep_default_na=False)

    scored_turns: List[Dict[str, Any]] = []
    conversations_summary: List[Dict[str, Any]] = []

    # Align by (conversation_id, turn)
    res_indexed = res_df.set_index(["conversation_id", "turn"])

    for conv_id, group in gt_df.groupby("conversation_id"):
        conv_turns_passed = 0
        total_conv_turns = len(group)
        conv_turn_details: List[Dict[str, Any]] = []

        for _, gt_row in group.iterrows():
            turn_key = (conv_id, gt_row["turn"])
            if turn_key not in res_indexed.index:
                # Missing turn in results
                turn_eval = {
                    "conversation_id": conv_id,
                    "turn": gt_row["turn"],
                    "decision_correct": False,
                    "must_include_pass": False,
                    "must_not_include_pass": False,
                    "tool_calls_correct": False,
                    "turn_passed": False,
                    "reason": "Missing from results",
                }
                scored_turns.append(turn_eval)
                conv_turn_details.append(turn_eval)
                continue

            res_row = res_indexed.loc[turn_key]
            if isinstance(res_row, pd.DataFrame):
                res_row = res_row.iloc[0]

            expected_decision = gt_row.get("expected_decision", "").strip().lower()
            actual_decision = str(res_row.get("decision", "")).strip().lower()
            decision_correct = (actual_decision == expected_decision)

            reply = str(res_row.get("reply", ""))
            must_inc_pass, missing_facts = check_must_include(gt_row.get("must_include", ""), reply)
            violations = check_must_not_include(gt_row.get("must_not_include", ""), reply)
            must_not_inc_pass = (len(violations) == 0)

            expected_tools = gt_row.get("expected_tool_calls", "")
            actual_tools = res_row.get("tool_calls", "[]")
            tool_calls_correct, tool_err = match_tool_calls(expected_tools, actual_tools)

            turn_passed = bool(
                decision_correct and must_inc_pass and must_not_inc_pass and tool_calls_correct
            )
            if turn_passed:
                conv_turns_passed += 1

            turn_eval = {
                "conversation_id": conv_id,
                "turn": gt_row["turn"],
                "decision_correct": decision_correct,
                "expected_decision": expected_decision,
                "actual_decision": actual_decision,
                "must_include_pass": must_inc_pass,
                "missing_facts": missing_facts,
                "must_not_include_pass": must_not_inc_pass,
                "violations": violations,
                "tool_calls_correct": tool_calls_correct,
                "tool_call_error": tool_err,
                "turn_passed": turn_passed,
            }
            scored_turns.append(turn_eval)
            conv_turn_details.append(turn_eval)

        conv_passed = (conv_turns_passed == total_conv_turns)
        conversations_summary.append({
            "conversation_id": conv_id,
            "passed": conv_passed,
            "turns_passed": conv_turns_passed,
            "total_turns": total_conv_turns,
            "turn_accuracy": round(conv_turns_passed / total_conv_turns, 4) if total_conv_turns else 0.0,
        })

    total_turns = len(scored_turns)
    total_turns_passed = sum(1 for t in scored_turns if t["turn_passed"])
    total_conversations = len(conversations_summary)
    total_conversations_passed = sum(1 for c in conversations_summary if c["passed"])

    summary_scores = {
        "total_conversations": total_conversations,
        "conversations_passed": total_conversations_passed,
        "conversation_pass_rate": round(total_conversations_passed / total_conversations, 4) if total_conversations else 0.0,
        "total_turns": total_turns,
        "turns_passed": total_turns_passed,
        "turn_pass_rate": round(total_turns_passed / total_turns, 4) if total_turns else 0.0,
        "conversations": conversations_summary,
        "turns": scored_turns,
    }

    out_file = Path(output_json)
    if not out_file.is_absolute():
        out_file = _REPO_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary_scores, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Evaluation finished: %d/%d conversations passed (%.1f%%), %d/%d turns passed (%.1f%%)",
        total_conversations_passed,
        total_conversations,
        summary_scores["conversation_pass_rate"] * 100,
        total_turns_passed,
        total_turns,
        summary_scores["turn_pass_rate"] * 100,
    )
    return summary_scores


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score Step 3 conversations")
    parser.add_argument("--results", default="outputs/step3_results.csv", help="Path to results CSV")
    parser.add_argument("--ground-truth", default=None, help="Path to practice/conversations.csv")
    parser.add_argument("--output", default="outputs/step3_scores.json", help="Path to output JSON")
    args = parser.parse_args()

    evaluate_step3(
        results_csv=args.results,
        ground_truth_csv=args.ground_truth,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
