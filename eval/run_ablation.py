"""Ablation evaluation runner for VoiceOps Phase 2E.

Runs DEV-only (26 rows) ablation experiments across feature switches:
Text runs:
  A: all OFF (baseline rerun)
  B: fix 1 only (FIX1_CATEGORY_EXPANSION)
  C: fix 4 only (FIX4_WARRANTY_PROMPT)
  D: fix 6 only (FIX6_TIMELINE_RANGE)
  E: all ON
ASR runs (from asr_text):
  F: all OFF
  G: fix 5 only (FIX5_ASR_NORMALISE)
  H: all ON
Optional:
  B2: fix 1 with SKU guard (if B exhibits SKU loss)

Adheres strictly to Gemini rate pacing (4.3s delay) and halts immediately on QuotaExceededError.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import QUESTIONS_DF
import src.retrieval as ret_module
from src.retrieval import HybridRetriever
import src.rag as rag_module
from src.rag import answer_question
from src.llm import QuotaExceededError
from src.scoring import score_row, compute_metrics
from eval.score_step2 import compute_retrieval_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval.run_ablation")


def get_dev_rows(df: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Extract sorted DEV rows (odd 1-indexed / even 0-indexed positions) from questions df."""
    df_sorted = df.sort_values(by="id").reset_index(drop=True)
    gold_by_id = {row["id"]: row.to_dict() for _, row in df_sorted.iterrows()}
    dev_rows = []
    for idx, row in df_sorted.iterrows():
        if (idx + 1) % 2 == 1:  # DEV row
            dev_rows.append(row.to_dict())
    return dev_rows, gold_by_id


def set_switches(
    fix1: bool = False,
    fix1_guard: bool = False,
    fix4: bool = False,
    fix5: bool = False,
    fix6: bool = False,
) -> Dict[str, bool]:
    """Configure module-level feature switches."""
    ret_module.FIX1_CATEGORY_EXPANSION = fix1
    ret_module.FIX1_SKU_GUARD = fix1_guard
    ret_module.FIX5_ASR_NORMALISE = fix5
    rag_module.FIX4_WARRANTY_PROMPT = fix4
    rag_module.FIX6_TIMELINE_RANGE = fix6

    config = {
        "FIX1_CATEGORY_EXPANSION": fix1,
        "FIX1_SKU_GUARD": fix1_guard,
        "FIX4_WARRANTY_PROMPT": fix4,
        "FIX5_ASR_NORMALISE": fix5,
        "FIX6_TIMELINE_RANGE": fix6,
    }
    return config


def score_run_rows(
    rows: List[Dict[str, Any]],
    gold_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute score_row, retrieval metrics, and 'not in documents' count."""
    eval_details = [score_row(gold_by_id[r["id"]], r) for r in rows if r["id"] in gold_by_id]
    metrics = compute_metrics(eval_details)
    metrics.update(compute_retrieval_metrics(rows, gold_by_id))

    not_in_docs = []
    for r in rows:
        reply_l = str(r.get("reply", "")).lower()
        if any(p in reply_l for p in [
            "not in our documents",
            "cannot find that in our documents",
            "not covered in our documents",
            "documents do not cover",
        ]):
            not_in_docs.append(r["id"])

    metrics["not_in_docs_count"] = len(not_in_docs)
    metrics["not_in_docs_ids"] = not_in_docs
    metrics["eval_details"] = eval_details
    return metrics


def execute_ablation_run(
    run_id: str,
    input_column: str,
    config: Dict[str, bool],
    dev_rows: List[Dict[str, Any]],
    gold_by_id: Dict[str, Dict[str, Any]],
    retriever: HybridRetriever,
    out_dir: Path,
) -> Dict[str, Any]:
    """Execute a single 26-row DEV ablation run."""
    out_file = out_dir / f"{run_id}.csv"
    print("\n" + "=" * 80)
    print(f"STARTING RUN {run_id} | Input: '{input_column}' | Config: {config}")
    print("=" * 80)

    # Set switches
    set_switches(
        fix1=config.get("FIX1_CATEGORY_EXPANSION", False),
        fix1_guard=config.get("FIX1_SKU_GUARD", False),
        fix4=config.get("FIX4_WARRANTY_PROMPT", False),
        fix5=config.get("FIX5_ASR_NORMALISE", False),
        fix6=config.get("FIX6_TIMELINE_RANGE", False),
    )

    fieldnames = [
        "id",
        "split",
        "question",
        "decision",
        "reply",
        "used_sections",
        "retrieved_top10",
        "raw_output",
        "finish_reason",
        "truncated",
    ]

    results: List[Dict[str, Any]] = []

    # Check for existing completed rows
    existing: Dict[str, Dict[str, Any]] = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("id") and r.get("decision") and r.get("decision") != "unparsed":
                        existing[r["id"]] = r
        except Exception:
            existing = {}

    if not out_file.exists() or out_file.stat().st_size == 0:
        with open(out_file, "w", encoding="utf-8", newline="") as f:
            f.write(f"# CONFIG: {json.dumps(config)}\n")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for count, row in enumerate(dev_rows, 1):
        qid = row["id"]
        if qid in existing:
            results.append(existing[qid])
            print(f"[{count}/26] {qid} already completed, resuming...", flush=True)
            continue

        query_text = row[input_column]
        print(f"[{count}/26] Answering {qid} from '{input_column}'...", flush=True)

        try:
            rag_res = answer_question(
                question=query_text,
                retriever=retriever,
                top_k=10,
                enable_rewrite=False,
            )
        except QuotaExceededError as qe:
            print(f"CRITICAL: QuotaExceededError on {qid}: {qe}", flush=True)
            print("HALTING ABLATION IMMEDIATELY PER INSTRUCTIONS.", flush=True)
            raise
        except Exception as e:
            print(f"Error answering {qid}: {e}", flush=True)
            rag_res = {
                "decision": "unparsed",
                "reply": "",
                "used_sections": [],
                "retrieved_items": [],
                "raw_output": f"ERROR: {e}",
                "finish_reason": "ERROR",
                "truncated": False,
            }

        row_res = {
            "id": qid,
            "split": "DEV",
            "question": query_text,
            "decision": rag_res["decision"],
            "reply": rag_res["reply"],
            "used_sections": json.dumps(rag_res["used_sections"], ensure_ascii=False),
            "retrieved_top10": json.dumps(rag_res["retrieved_items"], ensure_ascii=False),
            "raw_output": rag_res.get("raw_output", ""),
            "finish_reason": rag_res.get("finish_reason", "STOP"),
            "truncated": rag_res.get("truncated", False),
        }
        results.append(row_res)

        # Append row to CSV
        with open(out_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_res)

        print(f"   -> decision={row_res['decision']}, reply_len={len(row_res['reply'])}, used={rag_res['used_sections']}", flush=True)
        # Sequential pacing to avoid RPM limits
        time.sleep(4.3)

    metrics = score_run_rows(results, gold_by_id)
    metrics["run_id"] = run_id
    metrics["config"] = config
    metrics["input_column"] = input_column
    metrics["rows"] = results
    return metrics


def find_flips(
    baseline_run: Dict[str, Any],
    candidate_run: Dict[str, Any],
    gold_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compare candidate run against baseline run row by row for flips."""
    b_rows = {r["id"]: r for r in baseline_run["rows"]}
    c_rows = {r["id"]: r for r in candidate_run["rows"]}

    flips = []
    for qid in sorted(b_rows.keys()):
        b_r = b_rows[qid]
        c_r = c_rows[qid]
        gold = gold_by_id[qid]

        b_scored = score_row(gold, b_r)
        c_scored = score_row(gold, c_r)

        dec_flipped = b_scored["decision_correct"] != c_scored["decision_correct"]
        must_flipped = b_scored["must_include_pass"] != c_scored["must_include_pass"]

        if dec_flipped or must_flipped:
            # Determine reason
            reasons = []
            if dec_flipped:
                reasons.append(
                    f"Decision changed: {b_scored['decision']} ({'OK' if b_scored['decision_correct'] else 'FAIL'}) -> "
                    f"{c_scored['decision']} ({'OK' if c_scored['decision_correct'] else 'FAIL'})"
                )
            if must_flipped:
                reasons.append(
                    f"Must-include changed: {'PASS' if b_scored['must_include_pass'] else 'FAIL'} (missing {b_scored['missing_facts']}) -> "
                    f"{'PASS' if c_scored['must_include_pass'] else 'FAIL'} (missing {c_scored['missing_facts']})"
                )

            flips.append({
                "id": qid,
                "question": gold["question"],
                "baseline_decision": b_scored["decision"],
                "candidate_decision": c_scored["decision"],
                "baseline_reply": b_r["reply"],
                "candidate_reply": c_r["reply"],
                "baseline_must_inc": b_scored["must_include_pass"],
                "candidate_must_inc": c_scored["must_include_pass"],
                "reason": " ; ".join(reasons),
            })
    return flips


def main() -> None:
    dev_rows, gold_by_id = get_dev_rows(QUESTIONS_DF)
    out_dir = PROJECT_ROOT / "outputs" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing shared HybridRetriever...")
    retriever = HybridRetriever(use_query_rewrite=False)

    runs_plan = [
        # Text runs
        ("A", "question", {"FIX1_CATEGORY_EXPANSION": False, "FIX4_WARRANTY_PROMPT": False, "FIX5_ASR_NORMALISE": False, "FIX6_TIMELINE_RANGE": False}),
        ("B", "question", {"FIX1_CATEGORY_EXPANSION": True, "FIX4_WARRANTY_PROMPT": False, "FIX5_ASR_NORMALISE": False, "FIX6_TIMELINE_RANGE": False}),
        ("C", "question", {"FIX1_CATEGORY_EXPANSION": False, "FIX4_WARRANTY_PROMPT": True, "FIX5_ASR_NORMALISE": False, "FIX6_TIMELINE_RANGE": False}),
        ("D", "question", {"FIX1_CATEGORY_EXPANSION": False, "FIX4_WARRANTY_PROMPT": False, "FIX5_ASR_NORMALISE": False, "FIX6_TIMELINE_RANGE": True}),
        ("E", "question", {"FIX1_CATEGORY_EXPANSION": True, "FIX4_WARRANTY_PROMPT": True, "FIX5_ASR_NORMALISE": False, "FIX6_TIMELINE_RANGE": True}),
        # ASR runs
        ("F", "asr_text", {"FIX1_CATEGORY_EXPANSION": False, "FIX4_WARRANTY_PROMPT": False, "FIX5_ASR_NORMALISE": False, "FIX6_TIMELINE_RANGE": False}),
        ("G", "asr_text", {"FIX1_CATEGORY_EXPANSION": False, "FIX4_WARRANTY_PROMPT": False, "FIX5_ASR_NORMALISE": True, "FIX6_TIMELINE_RANGE": False}),
        ("H", "asr_text", {"FIX1_CATEGORY_EXPANSION": True, "FIX4_WARRANTY_PROMPT": True, "FIX5_ASR_NORMALISE": True, "FIX6_TIMELINE_RANGE": True}),
    ]

    all_metrics: Dict[str, Dict[str, Any]] = {}

    for run_id, input_col, cfg in runs_plan:
        res = execute_ablation_run(
            run_id=run_id,
            input_column=input_col,
            config=cfg,
            dev_rows=dev_rows,
            gold_by_id=gold_by_id,
            retriever=retriever,
            out_dir=out_dir,
        )
        all_metrics[run_id] = res

    # Check if B shows SKU loss compared to A
    sku_rate_A = all_metrics["A"]["sku_retrieval_rate_pct"]
    sku_rate_B = all_metrics["B"]["sku_retrieval_rate_pct"]
    print(f"\nSKU Retrieval Rate: Run A = {sku_rate_A}%, Run B = {sku_rate_B}%")
    if sku_rate_B < sku_rate_A:
        print("Run B shows SKU loss! Executing Run B2 (Fix 1 with SKU guard)...")
        cfg_b2 = {"FIX1_CATEGORY_EXPANSION": True, "FIX1_SKU_GUARD": True, "FIX4_WARRANTY_PROMPT": False, "FIX5_ASR_NORMALISE": False, "FIX6_TIMELINE_RANGE": False}
        res_b2 = execute_ablation_run(
            run_id="B2",
            input_column="question",
            config=cfg_b2,
            dev_rows=dev_rows,
            gold_by_id=gold_by_id,
            retriever=retriever,
            out_dir=out_dir,
        )
        all_metrics["B2"] = res_b2
    else:
        print("Run B does NOT show SKU loss relative to Run A.")

    # Print Summary Table
    print("\n" + "=" * 105)
    print("PHASE 2E ABLATION SUMMARY TABLE (DEV ONLY, n=26)")
    print("=" * 105)
    header = f"{'Run':<5} | {'Input':<9} | {'Decision Correct':<16} | {'Must-Include':<14} | {'Recall@5':<10} | {'Recall@10':<10} | {'SKU Rate':<10} | {'Not In Docs':<11} | {'n':<3}"
    print(header)
    print("-" * len(header))

    runs_to_print = ["A", "B"]
    if "B2" in all_metrics:
        runs_to_print.append("B2")
    runs_to_print.extend(["C", "D", "E", "F", "G", "H"])

    for rid in runs_to_print:
        m = all_metrics[rid]
        dec_str = f"{m['decision_correct']}/26 ({m['decision_correct_pct']}%)"
        must_str = f"{m['must_include_pass']}/26 ({m['must_include_pass_pct']}%)"
        r5_str = f"{m['recall_at_5_pct']}%"
        r10_str = f"{m['recall_at_10_pct']}%"
        sku_str = f"{m['sku_retrieval_rate_pct']}%"
        not_doc_str = str(m['not_in_docs_count'])
        inp_col = m["input_column"]
        print(f"{rid:<5} | {inp_col:<9} | {dec_str:<16} | {must_str:<14} | {r5_str:<10} | {r10_str:<10} | {sku_str:<10} | {not_doc_str:<11} | 26 ")

    print("=" * 105)

    # Flip Analysis
    print("\n" + "=" * 80)
    print("FLIP ANALYSIS (Changes in Decision or Must-Include)")
    print("=" * 80)

    comparisons = [
        ("B", "A"),
        ("C", "A"),
        ("D", "A"),
        ("G", "F"),
        ("H", "F"),
    ]
    if "B2" in all_metrics:
        comparisons.insert(1, ("B2", "A"))

    for cand_id, base_id in comparisons:
        flips = find_flips(all_metrics[base_id], all_metrics[cand_id], gold_by_id)
        print(f"\n--- {cand_id} vs {base_id} (Total Flips: {len(flips)}) ---")
        if not flips:
            print("  Zero decision or must-include flips.")
            continue
        for f in flips:
            print(f"Row ID        : {f['id']}")
            print(f"Question      : {f['question']}")
            print(f"Reason        : {f['reason']}")
            print(f"Baseline ({base_id})  : \"{f['baseline_reply']}\"")
            print(f"Candidate ({cand_id}): \"{f['candidate_reply']}\"")
            print("-" * 60)


if __name__ == "__main__":
    main()
