"""Compare ASR pre-fix (2B) vs post-fix (2D) to identify which rows flipped and why."""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import QUESTIONS_DF
from src.scoring import score_row

gold_by_id = {r['id']: r.to_dict() for _, r in QUESTIONS_DF.iterrows()}

df_before = pd.read_csv(PROJECT_ROOT / 'outputs/step2_asr_results_pre_fix.csv')
df_after = pd.read_csv(PROJECT_ROOT / 'outputs/step2_asr_results.csv')

print("=" * 80)
print("ASR BEFORE (2B) VS AFTER (2D) ROW-BY-ROW COMPARISON")
print("=" * 80)

flips_down = []
flips_up = []

for _, r_b in df_before.iterrows():
    qid = r_b['id']
    r_a = df_after[df_after['id'] == qid].iloc[0]
    gold = gold_by_id[qid]
    s_b = score_row(gold, r_b.to_dict())
    s_a = score_row(gold, r_a.to_dict())

    b_pass = s_b['decision_correct'] and s_b['must_include_pass']
    a_pass = s_a['decision_correct'] and s_a['must_include_pass']

    if b_pass and not a_pass:
        flips_down.append((qid, r_b['split'], s_b, s_a, r_b['reply'], r_a['reply'], gold['question']))
    elif not b_pass and a_pass:
        flips_up.append((qid, r_b['split'], s_b, s_a, r_b['reply'], r_a['reply'], gold['question']))

print(f"Total Negative Flips (PASS in 2B -> FAIL in 2D): {len(flips_down)}")
for qid, split, s_b, s_a, rep_b, rep_a, q in flips_down:
    print(f"\n[{split}] Row ID: {qid}")
    print(f"Question : {q}")
    print(f"Before (2B): decision={s_b['decision']} ({s_b['decision_correct']}), must_inc={s_b['must_include_pass']} (missing: {s_b['missing_facts']})")
    print(f"After  (2D): decision={s_a['decision']} ({s_a['decision_correct']}), must_inc={s_a['must_include_pass']} (missing: {s_a['missing_facts']})")
    print(f"Before Reply: \"{rep_b}\"")
    print(f"After Reply : \"{rep_a}\"")

print(f"\nTotal Positive Flips (FAIL in 2B -> PASS in 2D): {len(flips_up)}")
for qid, split, s_b, s_a, rep_b, rep_a, q in flips_up:
    print(f"\n[{split}] Row ID: {qid}")
    print(f"Question : {q}")
    print(f"Before (2B): decision={s_b['decision']} ({s_b['decision_correct']}), must_inc={s_b['must_include_pass']} (missing: {s_b['missing_facts']})")
    print(f"After  (2D): decision={s_a['decision']} ({s_a['decision_correct']}), must_inc={s_a['must_include_pass']} (missing: {s_a['missing_facts']})")
