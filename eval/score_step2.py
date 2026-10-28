"""Scoring script for Step 2 RAG evaluation."""

from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import QUESTIONS_DF
from src.scoring import score_row, compute_metrics


def parse_retrieved_top10(raw_val: str) -> List[Tuple[str, float]]:
    """Parse JSON or string representations of retrieved top 10 items."""
    if not raw_val:
        return []
    try:
        data = json.loads(raw_val)
        if isinstance(data, list):
            res = []
            for item in data:
                if isinstance(item, list) and len(item) >= 2:
                    res.append((str(item[0]), float(item[1])))
                elif isinstance(item, tuple) and len(item) >= 2:
                    res.append((str(item[0]), float(item[1])))
                elif isinstance(item, str):
                    res.append((item, 1.0))
            return res
    except Exception:
        pass
    return []


def compute_retrieval_metrics(
    eval_rows: List[Dict[str, Any]],
    gold_by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute retrieval metrics (recall@5, recall@10, hard_negative_hit_rate@5) on answerable=True rows."""
    r5_list = []
    r10_list = []
    hn_hit_list = []
    sku_entries_total = 0
    sku_entries_found_in_top10 = 0
    answerable_count = 0

    for r in eval_rows:
        qid = r["id"]
        gold = gold_by_id.get(qid, {})
        is_answerable = str(gold.get("answerable", "True")).strip().lower() not in ("false", "0")
        if not is_answerable:
            continue

        answerable_count += 1
        gold_evidence_raw = gold.get("gold_evidence", "")
        gold_items = [g.strip() for g in gold_evidence_raw.split(" ; ") if g.strip()]
        if not gold_items:
            continue

        retrieved_items = parse_retrieved_top10(r.get("retrieved_top10", ""))
        retrieved_ids = [item[0] for item in retrieved_items]
        top5_ids = retrieved_ids[:5]
        top10_ids = retrieved_ids[:10]

        # Check SKU entries separately
        sku_golds = [g for g in gold_items if g.startswith("products.csv:")]
        for sg in sku_golds:
            sku_entries_total += 1
            if sg in top10_ids:
                sku_entries_found_in_top10 += 1

        # Recall@5 and Recall@10
        hits_5 = sum(1 for g in gold_items if g in top5_ids)
        hits_10 = sum(1 for g in gold_items if g in top10_ids)
        r5_list.append(hits_5 / len(gold_items))
        r10_list.append(hits_10 / len(gold_items))

        # Hard negative hit rate @ 5 (evaluated ONLY on rows that have hard_negatives)
        hn_raw = gold.get("hard_negatives", "")
        hn_items = [h.strip() for h in hn_raw.split(" ; ") if h.strip()]
        if hn_items:
            hn_hits = sum(1 for h in hn_items if h in top5_ids)
            hn_hit_list.append(hn_hits / len(hn_items))

    macro_r5 = (sum(r5_list) / len(r5_list) * 100) if r5_list else 0.0
    macro_r10 = (sum(r10_list) / len(r10_list) * 100) if r10_list else 0.0
    macro_hn = (sum(hn_hit_list) / len(hn_hit_list) * 100) if hn_hit_list else 0.0

    return {
        "answerable_count": answerable_count,
        "recall_at_5_pct": round(macro_r5, 2),
        "recall_at_10_pct": round(macro_r10, 2),
        "hard_negative_n": len(hn_hit_list),
        "hard_negative_hit_rate_at_5_pct": round(macro_hn, 2),
        "sku_gold_entries_total": sku_entries_total,
        "sku_gold_entries_retrieved": sku_entries_found_in_top10,
        "sku_retrieval_rate_pct": round((sku_entries_found_in_top10 / sku_entries_total * 100), 2) if sku_entries_total else 100.0,
    }


def score_results_file(
    results_path: Path,
    gold_by_id: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Score a step2 results CSV file directly using src.scoring.score_row (no regex recovery)."""
    rows = []
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    eval_details = []
    for r in rows:
        qid = r["id"]
        if qid not in gold_by_id:
            continue
        gold = gold_by_id[qid]

        # Score directly via src.scoring.score_row - no regex recovery
        scored = score_row(gold, r)
        scored["split"] = r.get("split", "")
        scored["retrieved_top10"] = r.get("retrieved_top10", "")
        scored["finish_reason"] = r.get("finish_reason", "STOP")
        scored["truncated"] = str(r.get("truncated", "False")).lower() == "true"
        eval_details.append(scored)

    metrics = compute_metrics(eval_details)
    retrieval_metrics = compute_retrieval_metrics(rows, gold_by_id)
    metrics.update(retrieval_metrics)

    return metrics, eval_details, rows


def run_scoring(
    step2_csv: str = "outputs/step2_results.csv",
    step1_scores_json: str = "outputs/step1_scores.json",
    output_scores_json: str = "outputs/step2_scores.json",
    comparison_md: str = "outputs/step1_vs_step2.md",
    asr_results_csv: str = "outputs/step2_asr_results.csv",
):
    step2_path = PROJECT_ROOT / step2_csv
    if not step2_path.exists():
        print(f"Error: Step 2 results not found at {step2_path}. Run eval/run_step2.py first.")
        sys.exit(1)

    # Sort questions by id
    df_sorted = QUESTIONS_DF.sort_values(by="id").reset_index(drop=True)
    gold_by_id = {row["id"]: row.to_dict() for _, row in df_sorted.iterrows()}

    # Score Step 2 All
    metrics_all, eval_all, raw_rows_all = score_results_file(step2_path, gold_by_id)

    # Generate outputs/step2_recall_misses.csv
    recall_misses = []
    for r in raw_rows_all:
        qid = r["id"]
        gold = gold_by_id.get(qid, {})
        is_answerable = str(gold.get("answerable", "True")).strip().lower() not in ("false", "0")
        if not is_answerable:
            continue

        gold_evidence_raw = gold.get("gold_evidence", "")
        gold_items = [g.strip() for g in gold_evidence_raw.split(" ; ") if g.strip()]
        if not gold_items:
            continue

        retrieved_items = parse_retrieved_top10(r.get("retrieved_top10", ""))
        top5_ids = [item[0] for item in retrieved_items[:5]]
        missing_golds = [g for g in gold_items if g not in top5_ids]

        if missing_golds:
            top5_formatted = str([(item[0], item[1]) for item in retrieved_items[:5]])
            recall_misses.append({
                "id": qid,
                "question": gold["question"],
                "missing_gold_ids": " ; ".join(missing_golds),
                "top5_retrieved": top5_formatted,
            })

    misses_file = PROJECT_ROOT / "outputs" / "step2_recall_misses.csv"
    with open(misses_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "missing_gold_ids", "top5_retrieved"])
        writer.writeheader()
        writer.writerows(recall_misses)
    print(f"Saved {len(recall_misses)} recall misses to {misses_file}")

    # Score DEV and HOLDOUT subsets
    # Odd positions 1, 3, 5... (0, 2, 4 in 0-indexed) = DEV
    dev_ids = set(df_sorted.iloc[0::2]["id"])
    holdout_ids = set(df_sorted.iloc[1::2]["id"])

    eval_dev = [d for d in eval_all if d["id"] in dev_ids]
    eval_holdout = [d for d in eval_all if d["id"] in holdout_ids]
    raw_dev = [r for r in raw_rows_all if r["id"] in dev_ids]
    raw_holdout = [r for r in raw_rows_all if r["id"] in holdout_ids]

    metrics_dev = compute_metrics(eval_dev)
    metrics_dev.update(compute_retrieval_metrics(raw_dev, gold_by_id))

    metrics_holdout = compute_metrics(eval_holdout)
    metrics_holdout.update(compute_retrieval_metrics(raw_holdout, gold_by_id))

    # Score ASR if available
    asr_metrics = None
    asr_dev_metrics = None
    asr_holdout_metrics = None
    asr_path = PROJECT_ROOT / asr_results_csv
    if asr_path.exists():
        asr_m, asr_eval, asr_raw = score_results_file(asr_path, gold_by_id)
        asr_metrics = asr_m

        asr_eval_dev = [d for d in asr_eval if d["id"] in dev_ids]
        asr_eval_holdout = [d for d in asr_eval if d["id"] in holdout_ids]
        asr_raw_dev = [r for r in asr_raw if r["id"] in dev_ids]
        asr_raw_holdout = [r for r in asr_raw if r["id"] in holdout_ids]

        asr_dev_metrics = compute_metrics(asr_eval_dev)
        asr_dev_metrics.update(compute_retrieval_metrics(asr_raw_dev, gold_by_id))

        asr_holdout_metrics = compute_metrics(asr_eval_holdout)
        asr_holdout_metrics.update(compute_retrieval_metrics(asr_raw_holdout, gold_by_id))

    # Load Step 1 scores if available
    step1_path = PROJECT_ROOT / step1_scores_json
    step1_metrics = {}
    if step1_path.exists():
        with open(step1_path, "r", encoding="utf-8") as f:
            step1_data = json.load(f)
            step1_metrics = step1_data.get("metrics", {})
    else:
        print("[Notice] outputs/step1_scores.json not found yet.")

    # Save outputs/step2_scores.json
    out_json = PROJECT_ROOT / output_scores_json
    os.makedirs(out_json.parent, exist_ok=True)
    score_payload = {
        "step2_all": metrics_all,
        "step2_holdout": metrics_holdout,
        "step2_dev": metrics_dev,
        "step2_asr": asr_metrics,
        "step2_asr_dev": asr_dev_metrics,
        "step2_asr_holdout": asr_holdout_metrics,
        "step1": step1_metrics,
        "details_all": eval_all,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(score_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved Step 2 scores to {out_json}")

    # Build comparison Markdown table with explicit n for each row
    s1_dec = step1_metrics.get("decision_correct", "N/A")
    s1_dec_pct = f"{step1_metrics.get('decision_correct_pct', 0.0)}%" if "decision_correct_pct" in step1_metrics else "N/A"
    s1_must_inc = step1_metrics.get("must_include_pass", "N/A")
    s1_must_inc_pct = f"{step1_metrics.get('must_include_pass_pct', 0.0)}%" if "must_include_pass_pct" in step1_metrics else "N/A"
    s1_viol = step1_metrics.get("must_not_include_violations", "N/A")
    s1_unparsed = step1_metrics.get("unparsed", "N/A")
    s1_trunc = step1_metrics.get("truncated", "N/A")

    n_all = metrics_all["rows_run"]
    n_holdout = metrics_holdout["rows_run"]
    n_ans_all = metrics_all.get("answerable_count", 47)
    n_ans_holdout = metrics_holdout.get("answerable_count", 24)
    n_hn_all = metrics_all.get("hard_negative_n", 5)
    n_hn_holdout = metrics_holdout.get("hard_negative_n", 4)

    md_lines = [
        "# VoiceOps 2026: Step 1 vs Step 2 Evaluation Results\n",
        "| Metric | Step 1 (n=52) | Step 2 all (n=52) | Step 2 holdout (n=26) |",
        "| :--- | :--- | :--- | :--- |",
        f"| decision_correct | {s1_dec}/{n_all} ({s1_dec_pct}) | {metrics_all['decision_correct']}/{n_all} ({metrics_all['decision_correct_pct']}%) | {metrics_holdout['decision_correct']}/{n_holdout} ({metrics_holdout['decision_correct_pct']}%) |",
        f"| must_include_pass | {s1_must_inc}/{n_all} ({s1_must_inc_pct}) | {metrics_all['must_include_pass']}/{n_all} ({metrics_all['must_include_pass_pct']}%) | {metrics_holdout['must_include_pass']}/{n_holdout} ({metrics_holdout['must_include_pass_pct']}%) |",
        f"| must_not_include_violations | {s1_viol} | {metrics_all['must_not_include_violations']} | {metrics_holdout['must_not_include_violations']} |",
        f"| unparsed | {s1_unparsed}/{n_all} | {metrics_all['unparsed']}/{n_all} | {metrics_holdout['unparsed']}/{n_holdout} |",
        f"| truncated | {s1_trunc}/{n_all} | {metrics_all['truncated']}/{n_all} | {metrics_holdout['truncated']}/{n_holdout} |",
        f"| recall@5 (answerable) | N/A | {metrics_all['recall_at_5_pct']}% (n={n_ans_all}) | {metrics_holdout['recall_at_5_pct']}% (n={n_ans_holdout}) |",
        f"| recall@10 (answerable) | N/A | {metrics_all['recall_at_10_pct']}% (n={n_ans_all}) | {metrics_holdout['recall_at_10_pct']}% (n={n_ans_holdout}) |",
        f"| hard_negative_hit_rate@5 | N/A | {metrics_all['hard_negative_hit_rate_at_5_pct']}% (n={n_hn_all}) | {metrics_holdout['hard_negative_hit_rate_at_5_pct']}% (n={n_hn_holdout}) |",
    ]

    if asr_metrics:
        n_asr = asr_metrics["rows_run"]
        n_ans_asr = asr_metrics.get("answerable_count", 47)
        n_hn_asr = asr_metrics.get("hard_negative_n", 5)
        md_lines.append("")
        md_lines.append("## Step 2 ASR (Speech-to-Text Text Input) Results\n")
        md_lines.append("| Metric | Step 2 ASR (n=52) |")
        md_lines.append("| :--- | :--- |")
        md_lines.append(f"| decision_correct | {asr_metrics['decision_correct']}/{n_asr} ({asr_metrics['decision_correct_pct']}%) |")
        md_lines.append(f"| must_include_pass | {asr_metrics['must_include_pass']}/{n_asr} ({asr_metrics['must_include_pass_pct']}%) |")
        md_lines.append(f"| must_not_include_violations | {asr_metrics['must_not_include_violations']} |")
        md_lines.append(f"| unparsed | {asr_metrics['unparsed']}/{n_asr} |")
        md_lines.append(f"| truncated | {asr_metrics['truncated']}/{n_asr} |")
        md_lines.append(f"| recall@5 | {asr_metrics['recall_at_5_pct']}% (n={n_ans_asr}) |")
        md_lines.append(f"| recall@10 | {asr_metrics['recall_at_10_pct']}% (n={n_ans_asr}) |")
        md_lines.append(f"| hard_negative_hit_rate@5 | {asr_metrics['hard_negative_hit_rate_at_5_pct']}% (n={n_hn_asr}) |")
        md_lines.append(f"| sku_retrieval_rate | {asr_metrics['sku_retrieval_rate_pct']}% |")

    out_md = PROJECT_ROOT / comparison_md
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved comparison Markdown to {out_md}")

    # Print summary
    print("\n" + "=" * 75)
    print(f"{'METRIC':<30} | {'STEP 1 (n=52)':<15} | {'STEP 2 ALL (n=52)':<18} | {'STEP 2 HOLDOUT (n=26)':<18}")
    print("-" * 75)
    print(f"{'Decision Correct':<30} | {str(s1_dec):<15} | {metrics_all['decision_correct_pct']}% ({metrics_all['decision_correct']}/{n_all}) | {metrics_holdout['decision_correct_pct']}% ({metrics_holdout['decision_correct']}/{n_holdout})")
    print(f"{'Must Include Pass':<30} | {str(s1_must_inc):<15} | {metrics_all['must_include_pass_pct']}% ({metrics_all['must_include_pass']}/{n_all}) | {metrics_holdout['must_include_pass_pct']}% ({metrics_holdout['must_include_pass']}/{n_holdout})")
    print(f"{'Must Not Violations':<30} | {str(s1_viol):<15} | {metrics_all['must_not_include_violations']:<18} | {metrics_holdout['must_not_include_violations']:<18}")
    print(f"{'Unparsed Responses':<30} | {str(s1_unparsed):<15} | {metrics_all['unparsed']:<18} | {metrics_holdout['unparsed']:<18}")
    print(f"{'Truncated Responses':<30} | {str(s1_trunc):<15} | {metrics_all['truncated']:<18} | {metrics_holdout['truncated']:<18}")
    print(f"{'Recall@5':<30} | {'-':<15} | {metrics_all['recall_at_5_pct']}% (n={n_ans_all})    | {metrics_holdout['recall_at_5_pct']}% (n={n_ans_holdout})")
    print(f"{'Recall@10':<30} | {'-':<15} | {metrics_all['recall_at_10_pct']}% (n={n_ans_all})    | {metrics_holdout['recall_at_10_pct']}% (n={n_ans_holdout})")
    print(f"{'Hard Negative Hit@5':<30} | {'-':<15} | {metrics_all['hard_negative_hit_rate_at_5_pct']}% (n={n_hn_all})      | {metrics_holdout['hard_negative_hit_rate_at_5_pct']}% (n={n_hn_holdout})")
    print("=" * 75 + "\n")

    return score_payload

    return score_payload


if __name__ == "__main__":
    run_scoring()
