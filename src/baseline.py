"""Step 1: Prompt-only baseline without documents or tools."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Union

from src.data import QUESTIONS_DF
from src.llm import complete_with_meta, complete, QuotaExceededError, LLMResponse

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "outputs" / "step1_results.csv"

logger = logging.getLogger("voiceops.baseline")

# Kept exactly as initial prompt — do not modify to artificially raise scores
SYSTEM_PROMPT = """You are ZentroMart's customer support assistant.
In your response, you must state to the customer that you are an AI assistant.
You must NOT invent or assume policies, dates, amounts, fees, timelines, or order details.
If you need an order ID or specific information to look up an order, return decision "clarify" and ask for the order ID.
If the customer asks something that cannot be resolved without human intervention or is an unresolved complaint, use decision "escalate".
If the request is abusive, harmful, or out of scope, use decision "refuse".
If you can provide a factual answer or safe acknowledgement without inventing any details, use decision "answer".

You must respond ONLY with a JSON object in this format:
{"decision": "answer|clarify|escalate|refuse", "reply": "..."}
Do not include any text, thoughts, or markdown formatting outside the JSON object.
"""

VALID_DECISIONS = {"answer", "clarify", "escalate", "refuse"}


def extract_json_objects(text: str) -> List[dict]:
    """Extract all top-level valid JSON objects from text in order of appearance."""
    objs = []
    if not text:
        return objs

    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
        clean = clean.strip()

    # Fast path: whole string parse
    try:
        val = json.loads(clean)
        if isinstance(val, dict):
            return [val]
    except Exception:
        pass

    # Balanced brace scan
    n = len(clean)
    i = 0
    while i < n:
        if clean[i] == '{':
            depth = 0
            in_str = False
            escape = False
            start = i
            end = -1
            for j in range(i, n):
                c = clean[j]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    if in_str:
                        escape = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = j + 1
                            break
            if end != -1:
                chunk = clean[start:end]
                try:
                    parsed = json.loads(chunk)
                    if isinstance(parsed, dict):
                        objs.append(parsed)
                except Exception:
                    pass
                i = end
            else:
                i += 1
        else:
            i += 1
    return objs


def parse_llm_json(raw: str) -> Dict[str, str]:
    """Parse JSON response from LLM according to Phase 1B specifications:
    
    - Take the LAST top-level JSON object that parses and has both 'decision' and 'reply' keys.
    - If there is none, set decision=unparsed. Never match a decision string inside free text.
    - Valid decisions are answer, clarify, escalate and refuse. Anything else is unparsed.
    """
    objs = extract_json_objects(raw)

    # Take the LAST valid top-level JSON object with decision and reply
    for obj in reversed(objs):
        if "decision" in obj and "reply" in obj:
            dec = str(obj["decision"]).strip().lower()
            rep = str(obj["reply"]).strip()
            if dec in VALID_DECISIONS:
                return {"decision": dec, "reply": rep}

    return {"decision": "unparsed", "reply": ""}


def run_baseline(output_path: Union[str, Path] = _DEFAULT_OUTPUT) -> List[Dict[str, Any]]:
    """Run Step 1 prompt-only baseline on all 52 rows of practice/questions.csv."""
    import time
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Pre-flight quota check before full run
    print("Pre-flight quota check for model...", flush=True)
    try:
        check_resp = complete_with_meta(SYSTEM_PROMPT, "Hello, can you help me check an order?")
        print(f"Pre-flight check PASSED (finish_reason={check_resp.finish_reason}, truncated={check_resp.truncated})", flush=True)
    except QuotaExceededError as qe:
        err_time = datetime.now().isoformat()
        print(f"\n[FATAL] 429 Quota Exceeded during pre-flight check at {err_time}:\n{qe}\nStopping run immediately as required.\n", flush=True)
        raise qe

    total = len(QUESTIONS_DF)
    print(f"Running Step 1 baseline on {total} questions (paced at 4.3s to stay under 15 RPM limit)...", flush=True)

    results = []
    fieldnames = ["id", "question", "raw_output", "decision", "reply", "finish_reason", "truncated"]

    for idx, row in QUESTIONS_DF.iterrows():
        qid = row["id"]
        question = row["question"]
        step_num = len(results) + 1

        try:
            raw_output = complete_with_meta(system=SYSTEM_PROMPT, user=question)
            finish_reason = raw_output.finish_reason
            truncated = raw_output.truncated
        except QuotaExceededError as qe:
            err_time = datetime.now().isoformat()
            print(f"\n[FATAL] 429 Quota Exceeded on row {qid} at {err_time}:\n{qe}\nStopping run immediately as required.\n", flush=True)
            # Save partial results before stopping
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            raise qe
        except Exception as e:
            raw_output = LLMResponse(f"ERROR: {str(e)}", finish_reason="ERROR", truncated=False)
            finish_reason = "ERROR"
            truncated = False

        parsed = parse_llm_json(raw_output)
        item = {
            "id": qid,
            "question": question,
            "raw_output": str(raw_output),
            "decision": parsed["decision"],
            "reply": parsed["reply"],
            "finish_reason": finish_reason,
            "truncated": truncated
        }
        results.append(item)
        print(f"[{step_num}/{total}] {qid}: decision={item['decision']} finish={item['finish_reason']} trunc={item['truncated']}", flush=True)

        # Pace requests to remain safely below the 15 requests/minute free-tier limit
        if step_num < total:
            time.sleep(4.3)

    # Save to CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} results to {output_path}", flush=True)
    return results


if __name__ == "__main__":
    run_baseline()
