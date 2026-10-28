"""Evaluation and scoring script for Step 1."""

import csv
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import QUESTIONS_DF
from src.scoring import score_row, compute_metrics


def run_scoring(results_csv: str = "outputs/step1_results.csv",
                output_json: str = "outputs/step1_scores.json"):
    results_path = PROJECT_ROOT / results_csv
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}. Run 'python run.py step1' first.")
        sys.exit(1)

    # Read step 1 results
    results_by_id = {}
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results_by_id[row["id"]] = row

    # Ground truth indexed by id
    gold_by_id = {row["id"]: row.to_dict() for _, row in QUESTIONS_DF.iterrows()}

    # Confirm multi-value acceptable_decisions row handling
    multi_val_rows = [r for r in gold_by_id.values() if " ; " in r.get("acceptable_decisions", "")]
    if multi_val_rows:
        sample_row = multi_val_rows[0]
        options = [x.strip() for x in sample_row["acceptable_decisions"].split(" ; ")]
        print(f"[Verification] Confirmed multi-value row {sample_row['id']}: {options}")

    eval_details = []
    for qid, gold in gold_by_id.items():
        if qid not in results_by_id:
            print(f"Warning: Question ID {qid} missing from results.")
            continue
        res = results_by_id[qid]
        scored = score_row(gold, res)
        eval_details.append(scored)

    metrics = compute_metrics(eval_details)

    # Save outputs/step1_scores.json
    out_json_path = PROJECT_ROOT / output_json
    os.makedirs(out_json_path.parent, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "details": eval_details
        }, f, indent=2, ensure_ascii=False)

    print(f"\nStep 1 evaluation results saved to {out_json_path}\n")

    # Print a clean, formatted table of every metric
    print("=" * 60)
    print(f"{'METRIC':<35} | {'VALUE':<10} | {'TOTAL / PCT':<10}")
    print("-" * 60)
    print(f"{'Total Rows Evaluated':<35} | {metrics['rows_run']:<10} | {'100%':<10}")
    print(f"{'Decision Correct':<35} | {metrics['decision_correct']:<10} | {metrics['decision_correct_pct']}%")
    print(f"{'Must Include Passed':<35} | {metrics['must_include_pass']:<10} | {metrics['must_include_pass_pct']}%")
    print(f"{'Must Not Include Violations':<35} | {metrics['must_not_include_violations']:<10} | {'-':<10}")
    print(f"{'Answerable=False Flags':<35} | {metrics['answerable_false_flags']:<10} | {'-':<10}")
    print(f"{'Unparsed LLM Responses':<35} | {metrics['unparsed']:<10} | {'-':<10}")
    print(f"{'Truncated Rows':<35} | {metrics['truncated']:<10} | {'-':<10}")
    print("=" * 60)


if __name__ == "__main__":
    run_scoring()
