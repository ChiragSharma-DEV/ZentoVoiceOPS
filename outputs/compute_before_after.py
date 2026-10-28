"""Compute exact before and after metrics for DEV and HOLDOUT, Text and ASR."""
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import QUESTIONS_DF
from src.scoring import score_row, compute_metrics
from eval.score_step2 import compute_retrieval_metrics

df_sorted = QUESTIONS_DF.sort_values(by="id").reset_index(drop=True)
gold_by_id = {row["id"]: row.to_dict() for _, row in df_sorted.iterrows()}
dev_ids = set(df_sorted.iloc[0::2]["id"])
holdout_ids = set(df_sorted.iloc[1::2]["id"])

def score_file(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    
    eval_all = [score_row(gold_by_id[r["id"]], r) for r in rows if r["id"] in gold_by_id]
    
    # Split
    eval_dev = [d for d in eval_all if d["id"] in dev_ids]
    eval_holdout = [d for d in eval_all if d["id"] in holdout_ids]
    raw_dev = [r for r in rows if r["id"] in dev_ids]
    raw_holdout = [r for r in rows if r["id"] in holdout_ids]

    m_all = compute_metrics(eval_all)
    m_all.update(compute_retrieval_metrics(rows, gold_by_id))

    m_dev = compute_metrics(eval_dev)
    m_dev.update(compute_retrieval_metrics(raw_dev, gold_by_id))

    m_holdout = compute_metrics(eval_holdout)
    m_holdout.update(compute_retrieval_metrics(raw_holdout, gold_by_id))

    # Count "not in documents"
    not_in_docs_dev = []
    not_in_docs_holdout = []
    for r in rows:
        reply = r.get("reply", "").lower()
        if any(phrase in reply for phrase in ["not in our documents", "cannot find that in our documents", "not covered in our documents", "documents do not cover"]):
            if r["id"] in dev_ids:
                not_in_docs_dev.append(r["id"])
            else:
                not_in_docs_holdout.append(r["id"])

    return {
        "all": m_all,
        "dev": m_dev,
        "holdout": m_holdout,
        "not_in_docs_dev": not_in_docs_dev,
        "not_in_docs_holdout": not_in_docs_holdout,
    }

print("Computing pre-fix scores...")
# ASR pre-fix
asr_before = score_file(PROJECT_ROOT / "outputs" / "step2_asr_results_pre_fix.csv")

# ASR post-fix
print("Computing post-fix scores...")
asr_after = score_file(PROJECT_ROOT / "outputs" / "step2_asr_results.csv")
text_after = score_file(PROJECT_ROOT / "outputs" / "step2_results.csv")

# For text before, we have step2_scores.json recorded from Phase 2B
with open(PROJECT_ROOT / "outputs" / "step2_scores.json", "r", encoding="utf-8") as f:
    s2_scores = json.load(f)

print("\n--- RESULTS SUMMARY ---")
print("ASR BEFORE (DEV):")
print(f"  Decision correct: {asr_before['dev']['decision_correct']}/26 ({asr_before['dev']['decision_correct_pct']}%)")
print(f"  Must include pass: {asr_before['dev']['must_include_pass']}/26 ({asr_before['dev']['must_include_pass_pct']}%)")
print(f"  Recall@5: {asr_before['dev']['recall_at_5_pct']}%")
print(f"  Recall@10: {asr_before['dev']['recall_at_10_pct']}%")
print(f"  Not in docs: {len(asr_before['not_in_docs_dev'])} {asr_before['not_in_docs_dev']}")

print("\nASR AFTER (DEV):")
print(f"  Decision correct: {asr_after['dev']['decision_correct']}/26 ({asr_after['dev']['decision_correct_pct']}%)")
print(f"  Must include pass: {asr_after['dev']['must_include_pass']}/26 ({asr_after['dev']['must_include_pass_pct']}%)")
print(f"  Recall@5: {asr_after['dev']['recall_at_5_pct']}%")
print(f"  Recall@10: {asr_after['dev']['recall_at_10_pct']}%")
print(f"  Not in docs: {len(asr_after['not_in_docs_dev'])} {asr_after['not_in_docs_dev']}")

print("\nASR BEFORE (HOLDOUT):")
print(f"  Decision correct: {asr_before['holdout']['decision_correct']}/26 ({asr_before['holdout']['decision_correct_pct']}%)")
print(f"  Must include pass: {asr_before['holdout']['must_include_pass']}/26 ({asr_before['holdout']['must_include_pass_pct']}%)")
print(f"  Recall@5: {asr_before['holdout']['recall_at_5_pct']}%")
print(f"  Recall@10: {asr_before['holdout']['recall_at_10_pct']}%")
print(f"  Not in docs: {len(asr_before['not_in_docs_holdout'])} {asr_before['not_in_docs_holdout']}")

print("\nASR AFTER (HOLDOUT):")
print(f"  Decision correct: {asr_after['holdout']['decision_correct']}/26 ({asr_after['holdout']['decision_correct_pct']}%)")
print(f"  Must include pass: {asr_after['holdout']['must_include_pass']}/26 ({asr_after['holdout']['must_include_pass_pct']}%)")
print(f"  Recall@5: {asr_after['holdout']['recall_at_5_pct']}%")
print(f"  Recall@10: {asr_after['holdout']['recall_at_10_pct']}%")
print(f"  Not in docs: {len(asr_after['not_in_docs_holdout'])} {asr_after['not_in_docs_holdout']}")

print("\nTEXT AFTER (DEV):")
print(f"  Decision correct: {text_after['dev']['decision_correct']}/26 ({text_after['dev']['decision_correct_pct']}%)")
print(f"  Must include pass: {text_after['dev']['must_include_pass']}/26 ({text_after['dev']['must_include_pass_pct']}%)")
print(f"  Recall@5: {text_after['dev']['recall_at_5_pct']}%")
print(f"  Recall@10: {text_after['dev']['recall_at_10_pct']}%")
print(f"  Not in docs: {len(text_after['not_in_docs_dev'])} {text_after['not_in_docs_dev']}")

print("\nTEXT AFTER (HOLDOUT):")
print(f"  Decision correct: {text_after['holdout']['decision_correct']}/26 ({text_after['holdout']['decision_correct_pct']}%)")
print(f"  Must include pass: {text_after['holdout']['must_include_pass']}/26 ({text_after['holdout']['must_include_pass_pct']}%)")
print(f"  Recall@5: {text_after['holdout']['recall_at_5_pct']}%")
print(f"  Recall@10: {text_after['holdout']['recall_at_10_pct']}%")
print(f"  Not in docs: {len(text_after['not_in_docs_holdout'])} {text_after['not_in_docs_holdout']}")
