"""Scoring logic for VoiceOps 2026 evaluation."""

from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

MONTHS_HI = {
    1: ["जनवरी"],
    2: ["फ़रवरी", "फरवरी"],
    3: ["मार्च"],
    4: ["अप्रैल"],
    5: ["मई"],
    6: ["जून"],
    7: ["जुलाई"],
    8: ["अगस्त"],
    9: ["सितंबर", "सितम्बर"],
    10: ["अक्टूबर", "अक्तूबर"],
    11: ["नवंबर", "नवम्बर"],
    12: ["दिसंबर", "दिसम्बर"]
}

MONTHS_EN = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]


def normalize_text(text: str) -> str:
    """Normalize text: convert Devanagari digits to ASCII and lowercase."""
    if not text:
        return ""
    return text.translate(DEVANAGARI_DIGITS).lower()


def get_date_variants(date_str: str) -> List[str]:
    """Generate equivalent date representations for an ISO date YYYY-MM-DD."""
    variants = [date_str.lower()]
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        y = dt.year
        m = dt.month
        d = dt.day
        en_month = MONTHS_EN[m - 1]
        
        variants.extend([
            f"{d:02d}-{m:02d}-{y}",
            f"{d}-{m}-{y}",
            f"{d:02d}/{m:02d}/{y}",
            f"{d}/{m}/{y}",
            f"{d} {en_month} {y}",
            f"{d:02d} {en_month} {y}",
            f"{d}th {en_month} {y}",
            f"{d}st {en_month} {y}",
            f"{d}nd {en_month} {y}",
            f"{d}rd {en_month} {y}",
            f"{en_month} {d}, {y}",
            f"{en_month} {d:02d}, {y}",
            f"{d} {en_month[:3]} {y}",
            f"{d} {en_month}",
            f"{d}th {en_month}"
        ])
        
        for hi_month in MONTHS_HI.get(m, []):
            variants.extend([
                f"{d} {hi_month} {y}",
                f"{d} {hi_month}"
            ])
    except ValueError:
        pass

    return [v.lower() for v in set(variants)]


def match_alternative(alt: str, reply: str) -> bool:
    """Check if a single alternative matches within the reply."""
    alt_clean = alt.strip()
    if not alt_clean:
        return True
    
    reply_norm = normalize_text(reply)
    
    # Check if alt is an ISO date
    if re.match(r'^\d{4}-\d{2}-\d{2}$', alt_clean):
        variants = get_date_variants(alt_clean)
        for var in variants:
            if var in reply_norm:
                return True
        return False
    
    alt_norm = normalize_text(alt_clean)
    
    # If it's a standalone number, use regex word boundaries
    if alt_norm.isdigit():
        pattern = r'(?<!\d)' + re.escape(alt_norm) + r'(?!\d)'
        return bool(re.search(pattern, reply_norm))
    
    # Otherwise standard substring match
    return alt_norm in reply_norm


def check_must_include(must_include_str: str, reply: str) -> Tuple[bool, List[str]]:
    """Verify that every fact in must_include (separated by ' ; ') is matched in reply.
    
    Inside a fact, alternatives separated by ' / ' require at least one match.
    """
    if not must_include_str or not must_include_str.strip():
        return True, []
    
    facts = [f.strip() for f in must_include_str.split(" ; ") if f.strip()]
    missing = []
    
    for fact in facts:
        alternatives = [alt.strip() for alt in fact.split(" / ") if alt.strip()]
        matched = any(match_alternative(alt, reply) for alt in alternatives)
        if not matched:
            missing.append(fact)
            
    return len(missing) == 0, missing


def check_must_not_include(must_not_include_str: str, reply: str) -> List[str]:
    """Find any forbidden phrases present in reply."""
    if not must_not_include_str or not must_not_include_str.strip():
        return []
    
    reply_norm = normalize_text(reply)
    phrases = [p.strip() for p in must_not_include_str.split(" ; ") if p.strip()]
    violations = []
    
    for phrase in phrases:
        p_norm = normalize_text(phrase)
        if p_norm.isdigit():
            pattern = r'(?<!\d)' + re.escape(p_norm) + r'(?!\d)'
            if re.search(pattern, reply_norm):
                violations.append(phrase)
        else:
            if p_norm in reply_norm:
                violations.append(phrase)
                
    return violations


def check_unanswerable_flag(reply: str) -> Tuple[bool, List[str]]:
    """Flag replies on unanswerable rows that mention a rupee amount, a date, or a number."""
    if not reply or not reply.strip():
        return False, []
    
    reply_norm = normalize_text(reply)
    reasons = []
    
    # Rupee amount: ₹, rs, inr, rupee(s) followed by number
    if re.search(r'(?:₹|rs\.?|inr|rupees?)\s*\d+', reply_norm):
        reasons.append("contains_rupee_amount")
        
    # Dates: YYYY-MM-DD or 2026 or month name with day
    if re.search(r'\b202[0-9]\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', reply_norm):
        reasons.append("contains_date")
    elif any(m in reply_norm for m in MONTHS_EN):
        # Month name present alongside numbers
        if re.search(r'\d+', reply_norm):
            reasons.append("contains_date_with_month")
            
    # Any number
    if re.search(r'\b\d+\b', reply_norm):
        reasons.append("contains_number")
        
    return len(reasons) > 0, reasons


def score_row(gold: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single question result against ground truth.
    
    must_include and must_not_include are checked against the reply field ONLY.
    An unparsed row earns NO must_include credit.
    """
    decision = str(result.get("decision", "")).strip().lower()
    # Scorer strictly inspects the reply field
    reply = str(result.get("reply", "")).strip()
    
    acceptable = [d.strip().lower() for d in gold.get("acceptable_decisions", "").split(" ; ") if d.strip()]
    decision_correct = decision in acceptable
    
    is_unparsed = decision == "unparsed"

    # must_include and must_not_include checked against reply only
    # An unparsed row earns NO must_include credit
    if is_unparsed:
        must_inc_pass = False
        missing_facts = [f.strip() for f in gold.get("must_include", "").split(" ; ") if f.strip()]
        violations = check_must_not_include(gold.get("must_not_include", ""), reply)
    else:
        must_inc_pass, missing_facts = check_must_include(gold.get("must_include", ""), reply)
        violations = check_must_not_include(gold.get("must_not_include", ""), reply)
    
    answerable_str = str(gold.get("answerable", "True")).strip().lower()
    is_answerable = answerable_str not in ("false", "0")
    
    flagged = False
    flag_reasons = []
    if not is_answerable:
        flagged, flag_reasons = check_unanswerable_flag(reply)
        
    finish_reason = result.get("finish_reason", "UNKNOWN")
    truncated = result.get("truncated", False)
    if not isinstance(truncated, bool):
        truncated = str(truncated).lower() == "true"

    return {
        "id": gold["id"],
        "question": gold["question"],
        "decision": decision,
        "acceptable_decisions": gold.get("acceptable_decisions", ""),
        "decision_correct": decision_correct,
        "must_include_pass": must_inc_pass,
        "missing_facts": missing_facts,
        "must_not_include_violations": violations,
        "answerable": is_answerable,
        "answerable_false_flagged": flagged,
        "flag_reasons": flag_reasons,
        "unparsed": is_unparsed,
        "finish_reason": finish_reason,
        "truncated": truncated
    }


def compute_metrics(eval_details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate evaluation details into overall scores."""
    total_rows = len(eval_details)
    decision_correct_count = sum(1 for d in eval_details if d["decision_correct"])
    must_include_pass_count = sum(1 for d in eval_details if d["must_include_pass"])
    violations_count = sum(len(d["must_not_include_violations"]) for d in eval_details)
    flagged_count = sum(1 for d in eval_details if d["answerable_false_flagged"])
    unparsed_count = sum(1 for d in eval_details if d["unparsed"])
    truncated_count = sum(1 for d in eval_details if d["truncated"])
    
    return {
        "rows_run": total_rows,
        "decision_correct": decision_correct_count,
        "decision_correct_pct": round((decision_correct_count / total_rows * 100), 2) if total_rows else 0.0,
        "must_include_pass": must_include_pass_count,
        "must_include_pass_pct": round((must_include_pass_count / total_rows * 100), 2) if total_rows else 0.0,
        "must_not_include_violations": violations_count,
        "answerable_false_flags": flagged_count,
        "unparsed": unparsed_count,
        "truncated": truncated_count
    }


def test_scorer_checks_reply_only():
    """Quick unit test: reply lacking required fact fails even if raw_output has it, and unparsed gets 0 credit."""
    gold = {
        "id": "TEST_001",
        "question": "Where is my order 1234?",
        "acceptable_decisions": "answer",
        "must_include": "12 ; My Orders",
        "must_not_include": "guarantee",
        "answerable": "True"
    }
    # raw_output has required facts, but reply does NOT
    result = {
        "id": "TEST_001",
        "raw_output": "The customer mentioned 12 in the question. Go to My Orders.",
        "decision": "answer",
        "reply": "I am an AI assistant. How can I help you?",
        "finish_reason": "STOP",
        "truncated": False
    }
    scored = score_row(gold, result)
    assert scored["must_include_pass"] is False, "Test failed: must_include passed on raw_output instead of reply!"
    assert "12" in scored["missing_facts"] and "My Orders" in scored["missing_facts"], "Test failed: missing facts not detected!"

    # Test unparsed earns NO must_include credit
    gold_empty = {
        "id": "TEST_002",
        "question": "Random question",
        "acceptable_decisions": "answer",
        "must_include": "",
        "must_not_include": "",
        "answerable": "True"
    }
    result_unparsed = {
        "id": "TEST_002",
        "raw_output": "Some text",
        "decision": "unparsed",
        "reply": "",
        "finish_reason": "STOP",
        "truncated": False
    }
    scored_unparsed = score_row(gold_empty, result_unparsed)
    assert scored_unparsed["must_include_pass"] is False, "Test failed: unparsed row earned must_include credit!"
    print("[TEST PASSED] Reply lacking required fact correctly failed even though raw_output contained it.")
    print("[TEST PASSED] Missing facts correctly identified:", scored["missing_facts"])
    print("[TEST PASSED] Unparsed row earned zero must_include credit.")
    print("[TEST PASSED] Scorer reply-only assertions verified.")


# Run quick test on import
test_scorer_checks_reply_only()
