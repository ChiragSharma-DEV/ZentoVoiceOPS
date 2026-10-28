"""Test ASR text normalization on practice questions."""

from __future__ import annotations
import os
import re
import sys
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from src.data import DATASET_DIR
from src.asr_normalise import normalise_asr_text, SKU_CANDIDATE_PATTERN, ORDER_ID_CANDIDATE_PATTERN

df = pd.read_csv(Path(DATASET_DIR) / "practice" / "questions.csv")

tokens_checked = 0
tokens_recovered = 0
false_matches = []
recovered_list = []

for idx, r in df.iterrows():
    qid = r["id"]
    orig = str(r["asr_text"])
    norm = normalise_asr_text(orig)

    sku_candidates = list(SKU_CANDIDATE_PATTERN.finditer(orig))
    order_candidates = list(ORDER_ID_CANDIDATE_PATTERN.finditer(orig))
    tokens_checked += len(sku_candidates) + len(order_candidates)

    if orig != norm:
        for m in sku_candidates:
            raw_cand = m.group(0)
            norm_sub = normalise_asr_text(raw_cand)
            if raw_cand != norm_sub:
                tokens_recovered += 1
                recovered_list.append((qid, raw_cand, norm_sub))

print(f"Tokens checked: {tokens_checked}")
print(f"Tokens recovered: {tokens_recovered}")
print(f"False matches: {len(false_matches)}")
for qid, raw_cand, norm_sub in recovered_list:
    print(f"  [{qid}] \"{raw_cand}\" -> \"{norm_sub}\"")
