"""Score a frozen Step 2 run against practice questions."""
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import QUESTIONS_DF
from src.scoring import score_row, compute_metrics

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval/score_frozen_run.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    df_sorted = QUESTIONS_DF.sort_values(by="id").reset_index(drop=True)
    gold_by_id = {row["id"]: row.to_dict() for _, row in df_sorted.iterrows()}

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    eval_details = []
    decision_failures = []
    must_include_failures = []

    for r in rows:
        qid = r["id"]
        gold = gold_by_id[qid]
        scored = score_row(gold, r)
        eval_details.append(scored)
        if not scored["decision_correct"]:
            decision_failures.append((qid, r["decision"], gold["acceptable_decisions"]))
        if not scored["must_include_pass"]:
            must_include_failures.append((qid, scored.get("missing_facts", [])))

    metrics = compute_metrics(eval_details)

    print(f"=== SCORE RESULTS FOR {csv_path.name} (n={len(rows)}) ===")
    print(f"Decision Correct: {metrics['decision_correct']}/{len(rows)} ({metrics['decision_correct_pct']}%)")
    print(f"Must Include Pass: {metrics['must_include_pass']}/{len(rows)} ({metrics['must_include_pass_pct']}%)")
    print(f"Decision Failures ({len(decision_failures)}):")
    for qid, got, exp in decision_failures:
        print(f"  [{qid}] Got: '{got}' | Acceptable: '{exp}'")
    print(f"Must Include Failures ({len(must_include_failures)}):")
    for qid, missing in must_include_failures:
        print(f"  [{qid}] Missing facts: {missing}")
    print(f"Unparsed: {metrics['unparsed']}, Truncated: {metrics['truncated']}")

if __name__ == "__main__":
    main()
