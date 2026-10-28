"""Inspect Run A, B, and noise floor against saved 2B replies."""

from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data import QUESTIONS_DF
from src.scoring import score_row, compute_metrics
from eval.score_step2 import compute_retrieval_metrics

gold_by_id = {r['id']: r.to_dict() for _, r in QUESTIONS_DF.iterrows()}

print("=" * 80)
print("ABLATION RUN METRICS")
print("=" * 80)

for name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
    csv_file = PROJECT_ROOT / f"outputs/ablation/{name}.csv"
    if not csv_file.exists():
        continue
    df = pd.read_csv(csv_file, comment='#')
    rows = [r.to_dict() for _, r in df.iterrows()]
    scored = [score_row(gold_by_id[r['id']], r) for r in rows]
    m = compute_metrics(scored)
    m.update(compute_retrieval_metrics(rows, gold_by_id))
    not_in_docs = [r['id'] for r in rows if any(p in str(r.get('reply','')).lower() for p in ['not in our documents', 'cannot find that in our documents', 'not covered in our documents', 'documents do not cover'])]
    print(f"=== RUN {name} ===")
    print(f"  Rows total       : {len(rows)}")
    print(f"  Decision correct : {m['decision_correct']}/26 ({m['decision_correct_pct']}%)")
    print(f"  Must include pass: {m['must_include_pass']}/26 ({m['must_include_pass_pct']}%)")
    print(f"  Recall@5         : {m['recall_at_5_pct']}%")
    print(f"  Recall@10        : {m['recall_at_10_pct']}%")
    print(f"  SKU rate         : {m['sku_retrieval_rate_pct']}%")
    print(f"  Not in docs count: {len(not_in_docs)} {not_in_docs}")

# Step 2: Compare Run A with saved 2B replies
print("\n" + "=" * 80)
print("NOISE FLOOR: RUN A (all OFF) VS SAVED 2B REPLIES")
print("=" * 80)
df_a = pd.read_csv(PROJECT_ROOT / "outputs/ablation/A.csv", comment='#')
df_2b = pd.read_csv(PROJECT_ROOT / "outputs/step2_results.csv")
dev_2b = df_2b[df_2b['split'] == 'DEV'].reset_index(drop=True)

diffs = []
for i in range(len(df_a)):
    r_a = df_a.iloc[i]
    r_2b = dev_2b[dev_2b['id'] == r_a['id']].iloc[0]
    if r_a['reply'].strip() != r_2b['reply'].strip():
        diffs.append((r_a['id'], r_2b['reply'], r_a['reply']))

print(f"Total DEV replies differing from saved 2B replies: {len(diffs)} / 26")
print(f"Differing row IDs: {[d[0] for d in diffs]}")

# Flip analysis: B vs A
print("\n" + "=" * 80)
print("FLIPS: RUN B (Fix 1 only) VS RUN A (Baseline)")
print("=" * 80)
df_b = pd.read_csv(PROJECT_ROOT / "outputs/ablation/B.csv", comment='#')
b_flips = []
for i in range(len(df_a)):
    r_a = df_a.iloc[i]
    r_b = df_b[df_b['id'] == r_a['id']].iloc[0]
    gold = gold_by_id[r_a['id']]
    s_a = score_row(gold, r_a.to_dict())
    s_b = score_row(gold, r_b.to_dict())
    dec_flip = s_a['decision_correct'] != s_b['decision_correct']
    must_flip = s_a['must_include_pass'] != s_b['must_include_pass']
    if dec_flip or must_flip:
        reasons = []
        if dec_flip:
            reasons.append(f"Decision: {s_a['decision']} ({s_a['decision_correct']}) -> {s_b['decision']} ({s_b['decision_correct']})")
        if must_flip:
            reasons.append(f"Must-include: {s_a['must_include_pass']} (missing {s_a['missing_facts']}) -> {s_b['must_include_pass']} (missing {s_b['missing_facts']})")
        b_flips.append({
            "id": r_a['id'],
            "question": gold['question'],
            "reason": " ; ".join(reasons),
            "reply_A": r_a['reply'],
            "reply_B": r_b['reply'],
        })

print(f"Total B vs A flips: {len(b_flips)}")
for f in b_flips:
    print(f"Row ID   : {f['id']}")
    print(f"Question : {f['question']}")
    print(f"Reason   : {f['reason']}")
    print(f"Reply A  : \"{f['reply_A']}\"")
    print(f"Reply B  : \"{f['reply_B']}\"")
    print("-" * 60)
