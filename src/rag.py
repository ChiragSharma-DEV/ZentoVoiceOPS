"""RAG Answering module for VoiceOps 2026.

Uses retrieved knowledge chunks, complete_with_meta from src.llm,
and shared JSON parsing from src.baseline.
"""

from __future__ import annotations
import json
import os
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from src.kb import Chunk, add_business_days
from src.retrieval import HybridRetriever
from src.llm import complete_with_meta, LLMResponse, QuotaExceededError
from src.baseline import parse_llm_json, extract_json_objects, VALID_DECISIONS

REFERENCE_DATE = "2026-10-20"

# Global API call counter
API_CALL_COUNT = 0

# Phase 2D/2E Feature Switches
FIX4_WARRANTY_PROMPT: bool = False
FIX6_TIMELINE_RANGE: bool = False

BASE_SYSTEM_PROMPT = """You are ZentroMart's support assistant.
In your first reply to a customer, say you are an AI assistant.
Answer ONLY from the provided document sections and product rows. Cite the section IDs you used in the "used_sections" list.
If the sections do not cover the question, say you cannot find that in our documents, and offer to connect the customer with a human colleague. Do not guess. Never state a date, amount, fee, or policy that is not in the provided sections.
Choose one decision:
- "answer": the documents cover the question, or documents do not cover it so you state you cannot find that in our documents and offer to connect with a human colleague.
- "clarify": one detail is missing to apply the policy (e.g. which product/category, or order details); ask exactly one question.
- "escalate": the documents say a person must decide, high-value damage review is needed, or an unauthorized account order is reported.
- "refuse": the request breaks a policy or asks for private customer personal information.

Reply in the language the customer used: English, Hindi, or Hinglish.

You must respond ONLY with a JSON object in this format:
{"decision": "answer|clarify|escalate|refuse", "reply": "...", "used_sections": ["..."]}
Do not include any text, thoughts, or markdown formatting outside the JSON object.
"""

WARRANTY_PROMPT_LINE = "When the reply confirms warranty or Assured coverage, state the coverage duration in months, taken from the product row or WAR-2.\n"


def get_system_prompt() -> str:
    """Return system prompt, conditionally including warranty prompt line when FIX4 is active."""
    if FIX4_WARRANTY_PROMPT:
        parts = BASE_SYSTEM_PROMPT.split("Choose one decision:")
        return f"{parts[0]}{WARRANTY_PROMPT_LINE}Choose one decision:{parts[1]}"
    return BASE_SYSTEM_PROMPT


def extract_date_from_query(query: str) -> Optional[str]:
    """Extract an explicit date from query in ISO or English/Hindi format."""
    # ISO format YYYY-MM-DD
    m_iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
    if m_iso:
        return m_iso.group(1)

    # DD Month YYYY
    months_map = {
        "january": "01", "जनवरी": "01",
        "february": "02", "फ़रवरी": "02", "फरवरी": "02",
        "march": "03", "मार्च": "03",
        "april": "04", "अप्रैल": "04",
        "may": "05", "मई": "05",
        "june": "06", "जून": "06",
        "july": "07", "जुलाई": "07",
        "august": "08", "अगस्त": "08",
        "september": "09", "सितंबर": "09", "सितम्बर": "09",
        "october": "10", "अक्टूबर": "10", "अक्तूबर": "10",
        "november": "11", "नवंबर": "11", "नवम्बर": "11",
        "december": "12", "दिसंबर": "12", "दिसम्बर": "12",
    }
    pattern = r"\b(\d{1,2})\s+([a-zA-Z\u0900-\u097F]+)(?:\s+(\d{4}))?\b"
    for match in re.finditer(pattern, query):
        day_str = match.group(1).zfill(2)
        month_word = match.group(2).lower()
        year_str = match.group(3) if match.group(3) else "2026"
        if month_word in months_map:
            return f"{year_str}-{months_map[month_word]}-{day_str}"

    return None


def precompute_refund_calculations(query: str) -> Optional[str]:
    """Compute exact business-day refund completion date/range in Python without LLM arithmetic."""
    q_norm = query.lower()
    if "refund" not in q_norm and "रिफ़ंड" not in q_norm and "रिफंड" not in q_norm:
        return None

    # Check for start date
    start_date = extract_date_from_query(query)
    if not start_date:
        return None

    # Map payment methods to business day ranges per REF-2
    min_days = None
    max_days = None
    method_name = ""
    if "credit card" in q_norm or "debit card" in q_norm or "card" in q_norm:
        min_days, max_days = 5, 7
        method_name = "Credit/debit card (5 to 7 business days)"
    elif "upi" in q_norm:
        min_days, max_days = 1, 3
        method_name = "UPI (1 to 3 business days)"
    elif "net banking" in q_norm or "netbanking" in q_norm:
        min_days, max_days = 3, 5
        method_name = "Net banking (3 to 5 business days)"
    elif "neft" in q_norm or "bank transfer" in q_norm:
        min_days, max_days = 5, 7
        method_name = "Bank transfer (5 to 7 business days)"

    if max_days is not None:
        try:
            latest_deadline = add_business_days(start_date, max_days)
            dt_latest = datetime.strptime(latest_deadline, "%Y-%m-%d")
            en_latest = dt_latest.strftime("%d %B %Y").lstrip("0")

            if FIX6_TIMELINE_RANGE and min_days is not None:
                earliest_deadline = add_business_days(start_date, min_days)
                dt_earliest = datetime.strptime(earliest_deadline, "%Y-%m-%d")
                en_earliest = dt_earliest.strftime("%d %B %Y").lstrip("0")
                return (
                    f"PYTHON PRE-CALCULATED REFUND TIMELINE (per REF-2 & KB-00.3, reference date {REFERENCE_DATE}):\n"
                    f"Refund method: {method_name} ({min_days} to {max_days} business days)\n"
                    f"Refund initiated on: {start_date}\n"
                    f"Timeline range: {min_days} to {max_days} business days (skipping Sundays and holidays in data/holidays.csv)\n"
                    f"Earliest completion date: {earliest_deadline} ({en_earliest})\n"
                    f"Latest completion date: {latest_deadline} ({en_latest})\n"
                    f"State both the earliest and latest dates in your reply (e.g. 'between {en_earliest} and {en_latest}' or '{en_earliest} to {en_latest}')."
                )
            else:
                return (
                    f"PYTHON PRE-CALCULATED REFUND TIMELINE (per REF-2 & KB-00.3, reference date {REFERENCE_DATE}):\n"
                    f"Refund method: {method_name} ({max_days} business days)\n"
                    f"Refund initiated on: {start_date}\n"
                    f"Timeline: {max_days} business days (skipping Sundays and holidays in data/holidays.csv)\n"
                    f"Expected completion date: {latest_deadline} ({en_latest})\n"
                    f"State the expected completion date in your reply (e.g. 'by {en_latest}')."
                )
        except Exception:
            pass

    return None


def answer_question(
    question: str,
    retriever: HybridRetriever,
    top_k: int = 10,
    enable_rewrite: bool = False,
    is_asr: bool = False,
) -> Dict[str, Any]:
    """Run full RAG pipeline: retrieve sections, build grounded context, and generate reply."""
    global API_CALL_COUNT
    # 1. Retrieve top chunks
    retrieved_items = retriever.retrieve(query=question, top_k=top_k, enable_rewrite=enable_rewrite, is_asr=is_asr)
    retrieved_with_scores = [(c.section_id, round(score, 4)) for c, score in retrieved_items]

    # 2. Build context text
    context_parts = []
    for c, score in retrieved_items:
        context_parts.append(f"--- SECTION [{c.section_id}] (Score: {score:.4f}) ---\n{c.text}\n")

    context_str = "\n".join(context_parts)

    # 3. Check for precalculated date math
    precalc_note = precompute_refund_calculations(question)
    if precalc_note:
        context_str = f"{precalc_note}\n\n{context_str}"

    user_prompt = (
        f"CUSTOMER QUESTION:\n{question}\n\n"
        f"KNOWLEDGE BASE CONTEXT (Today is {REFERENCE_DATE}):\n"
        f"{context_str}\n\n"
        f"You must respond ONLY with a JSON object in this format:\n"
        f'{{"decision": "answer|clarify|escalate|refuse", "reply": "...", "used_sections": ["..."]}}\n'
        f"Do not include any text, thoughts, or markdown formatting outside the JSON object."
    )

    try:
        sys_prompt = get_system_prompt()
        API_CALL_COUNT += 1
        raw_output = complete_with_meta(system=sys_prompt, user=user_prompt)
        finish_reason = getattr(raw_output, "finish_reason", "STOP")
        truncated = getattr(raw_output, "truncated", False)
    except QuotaExceededError:
        raise
    except Exception as e:
        raw_output = LLMResponse(f"ERROR: {str(e)}", finish_reason="ERROR", truncated=False)
        finish_reason = "ERROR"
        truncated = False

    parsed = parse_llm_json(raw_output)
    decision = parsed["decision"]
    reply = parsed["reply"]

    used_sections = []
    if decision in VALID_DECISIONS:
        objs = extract_json_objects(raw_output)
        for obj in reversed(objs):
            if "decision" in obj and "reply" in obj:
                if str(obj["decision"]).strip().lower() in VALID_DECISIONS:
                    u = obj.get("used_sections", [])
                    if isinstance(u, list):
                        used_sections = [str(x).strip() for x in u if str(x).strip()]
                    elif isinstance(u, str):
                        used_sections = [u.strip()]
                    break

    return {
        "question": question,
        "decision": decision,
        "reply": reply,
        "used_sections": used_sections,
        "retrieved_items": retrieved_with_scores,
        "raw_output": str(raw_output),
        "finish_reason": finish_reason,
        "truncated": truncated,
    }
