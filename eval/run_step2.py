"""Evaluation runner for Step 2 RAG pipeline on practice/questions.csv."""

from __future__ import annotations
import csv
from datetime import datetime
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import QUESTIONS_DF
import src.retrieval as retrieval_module
import src.rag as rag_module
from src.retrieval import HybridRetriever
from src.rag import answer_question, API_CALL_COUNT
from src.llm import QuotaExceededError

QUOTA_ERRORS = []


def get_splits(df):
    """Sort questions by ID and split into DEV (odd positions) and HOLDOUT (even positions)."""
    df_sorted = df.sort_values(by="id").reset_index(drop=True)
    # 1-indexed: 1st, 3rd, 5th -> DEV (0, 2, 4 in 0-indexed)
    dev_mask = [(i + 1) % 2 == 1 for i in range(len(df_sorted))]
    dev_df = df_sorted[dev_mask].reset_index(drop=True)
    holdout_df = df_sorted[[not m for m in dev_mask]].reset_index(drop=True)
    return df_sorted, dev_df, holdout_df


def run_pipeline(
    input_column: str = "question",
    output_path: str = "outputs/step2_results.csv",
    use_query_rewrite: bool = False,
    top_k: int = 10,
    target_split: Optional[str] = None,
):
    """Run Step 2 RAG answering pipeline on all questions and save results."""
    out_file = PROJECT_ROOT / output_path
    os.makedirs(out_file.parent, exist_ok=True)

    df_sorted, _, _ = get_splits(QUESTIONS_DF)
    is_asr_run = input_column == "asr_text"
    config_dump = {
        "input_column": input_column,
        "split": target_split or "ALL",
        "fix1_category_expansion": retrieval_module.FIX1_CATEGORY_EXPANSION,
        "fix1_sku_guard": retrieval_module.FIX1_SKU_GUARD,
        "fix4_warranty_prompt": rag_module.FIX4_WARRANTY_PROMPT,
        "fix5_asr_normalise": retrieval_module.FIX5_ASR_NORMALISE,
        "fix6_timeline_range": rag_module.FIX6_TIMELINE_RANGE,
        "modality_gate": {
            "is_asr": is_asr_run,
            "fix5_application_gate": "FIX5_ASR_NORMALISE and is_asr",
            "fix5_would_apply": retrieval_module.FIX5_ASR_NORMALISE and is_asr_run,
        },
    }
    print("CONFIG_DUMP " + json.dumps(config_dump, sort_keys=True), flush=True)
    print(f"Initializing HybridRetriever (query_rewrite={use_query_rewrite}, top_k={top_k})...", flush=True)
    retriever = HybridRetriever(use_query_rewrite=use_query_rewrite)

    results = []
    rows_to_process = []
    for idx, row in df_sorted.iterrows():
        split_name = "DEV" if (idx + 1) % 2 == 1 else "HOLDOUT"
        if target_split and split_name != target_split.upper():
            continue
        rows_to_process.append((idx, split_name, row))

    total = len(rows_to_process)
    print(f"Running Step 2 on {total} questions (split={target_split or 'ALL'}) from column '{input_column}'...", flush=True)

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

    # Check for existing completed rows to enable resuming after interruption
    existing_results: Dict[str, Dict[str, Any]] = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("id") and r.get("decision") and r.get("decision") != "unparsed":
                        existing_results[r["id"]] = r
        except Exception:
            existing_results = {}

    # Initialize CSV with header only if file does not exist or is empty
    if not out_file.exists() or out_file.stat().st_size == 0:
        with open(out_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for count, (idx, split_name, row) in enumerate(rows_to_process, 1):
        qid = row["id"]
        if qid in existing_results:
            results.append(existing_results[qid])
            print(f"[{count}/{total}] [{split_name}] {qid} already completed, resuming...", flush=True)
            continue

        query_text = row[input_column]
        is_asr = (input_column == "asr_text")
        print(f"[{count}/{total}] [{split_name}] Answering {qid}...", flush=True)

        max_retries = 3
        rag_res = None
        quota_halted = False
        for attempt in range(max_retries):
            try:
                rag_res = answer_question(
                    question=query_text,
                    retriever=retriever,
                    top_k=top_k,
                    enable_rewrite=use_query_rewrite,
                    is_asr=is_asr,
                )
                break
            except QuotaExceededError as qe:
                err_record = {
                    "qid": qid,
                    "attempt": attempt,
                    "time": datetime.now().isoformat(),
                    "error": str(qe)
                }
                QUOTA_ERRORS.append(err_record)
                print(f"  [429 Quota Exceeded at {err_record['time']}] Halting run immediately per Phase 2G instructions.", flush=True)
                quota_halted = True
                break
            except Exception as e:
                print(f"  [Error] Failed answering {qid}: {e}", flush=True)
                rag_res = {
                    "question": query_text,
                    "decision": "unparsed",
                    "reply": "",
                    "used_sections": [],
                    "retrieved_items": [],
                    "raw_output": f"ERROR: {str(e)}",
                    "finish_reason": "ERROR",
                    "truncated": False,
                }
                break

        if quota_halted:
            print(f"Run stopped immediately on 429. Completed rows before halt: {len(results)}", flush=True)
            break

        if rag_res is None:
            rag_res = {
                "question": query_text,
                "decision": "unparsed",
                "reply": "",
                "used_sections": [],
                "retrieved_items": [],
                "raw_output": "ERROR: Quota exceeded after retries",
                "finish_reason": "ERROR",
                "truncated": False,
            }

        row_res = {
            "id": qid,
            "split": split_name,
            "question": query_text,
            "decision": rag_res["decision"],
            "reply": rag_res["reply"],
            "used_sections": json.dumps(rag_res["used_sections"], ensure_ascii=False),
            "retrieved_top10": json.dumps(rag_res["retrieved_items"], ensure_ascii=False),
            "raw_output": rag_res["raw_output"],
            "finish_reason": rag_res.get("finish_reason", "STOP"),
            "truncated": rag_res.get("truncated", False),
        }
        results.append(row_res)

        # Append row incrementally to disk
        with open(out_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_res)

        print(f"   -> decision={row_res['decision']}, finish={row_res['finish_reason']}, trunc={row_res['truncated']}", flush=True)

        # 4.3s delay keeps requests under the 15 requests per minute Free Tier limit
        time.sleep(4.3)

    from src.rag import API_CALL_COUNT
    print(f"Saved {len(results)} Step 2 results to {out_file}", flush=True)
    print(f"Total API calls made in this run: {API_CALL_COUNT}", flush=True)
    if QUOTA_ERRORS:
        quota_log_file = PROJECT_ROOT / "outputs" / f"step2_quota_errors_{input_column}.json"
        with open(quota_log_file, "w", encoding="utf-8") as f:
            json.dump(QUOTA_ERRORS, f, indent=2)
        print(f"Recorded {len(QUOTA_ERRORS)} quota errors to {quota_log_file}", flush=True)
    return results


if __name__ == "__main__":
    input_col = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("question", "asr_text") else "question"
    split_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].upper() in ("DEV", "HOLDOUT") else None

    if len(sys.argv) > 3:
        out_p = sys.argv[3]
    elif split_arg:
        out_p = f"outputs/step2_{input_col}_{split_arg.lower()}_results.csv"
    else:
        out_p = "outputs/step2_asr_results.csv" if input_col == "asr_text" else "outputs/step2_results.csv"

    # Safety check: do not overwrite protected files
    base_name = Path(out_p).name
    if base_name in ("step2_results.csv", "step2_asr_results_pre_fix.csv"):
        raise ValueError(f"Protected baseline file {out_p} cannot be overwritten in Phase 2G.")

    run_pipeline(input_column=input_col, output_path=out_p, target_split=split_arg)
