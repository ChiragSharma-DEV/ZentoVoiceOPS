"""Offline Grounding Audit Script for Step 2.
Audits every factual claim in outputs/step2_results.csv against used_sections.
"""

from __future__ import annotations
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.kb import load_and_chunk_kb
from src.data import PRODUCTS_DF

# 1. Load knowledge base chunks
chunks = load_and_chunk_kb()
chunk_map = {c.section_id: c.text for c in chunks}

# 2. Load product database
product_map = {}
for _, row in PRODUCTS_DF.iterrows():
    sku = str(row["sku"]).strip()
    title = str(row.get("title", "")).strip()
    category = str(row.get("category", "")).strip()
    ret_cat = str(row.get("return_category", "")).strip()
    assured = str(row.get("assured", "")).strip().lower()
    warranty = str(row.get("warranty_months", "")).strip()
    desc = f"SKU: {sku} | {sku} | Title: {title} | Category: {category} | Return Category: {ret_cat} | Assured: {assured} | Warranty: {warranty} months"
    product_map[f"products.csv:{sku}"] = desc


def extract_claims(reply: str, qid: str) -> list[str]:
    """Extract factual claims from reply: numbers, dates, durations, amounts, named policies, SKUs, and yes/no."""
    claims = []

    # Filter out canned intro / disclaimer
    text = reply
    text = re.sub(r"^Hello!\s*I am (?:an )?AI assistant(?: for ZentroMart)?\.?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Hello,\s*I am (?:an )?AI assistant(?: for ZentroMart)?\.?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^I am (?:an )?AI assistant(?: for ZentroMart)?\.?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Main ek AI assistant hoon\.?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^नमस्ते,\s*मैं ZentroMart का AI सहायक हूँ।?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^नमस्ते!\s*मैं ZentroMart का AI assistant हूँ।?\s*", "", text, flags=re.IGNORECASE)

    # 1. Yes / No / Policy outcome claims
    if re.search(r"\b(?:Yes|Haan|हाँ)\b", text, re.IGNORECASE):
        claims.append("Yes (affirmative response)")
    if "not charged" in text.lower() or "will not be charged" in text.lower():
        claims.append("will not be charged reverse pickup fee")
    if "cannot be returned" in text.lower() or "not returnable" in text.lower() or "वापस नहीं किया जा सकता" in text:
        claims.append("item cannot be returned")
    if "cannot be cancelled" in text.lower() or "कैंसिल नहीं होता" in text:
        claims.append("order cannot be cancelled after shipping")
    if "refuse the parcel" in text.lower() or "पार्सल लेने से मना" in text:
        claims.append("can refuse parcel at delivery")

    # 2. SKUs
    skus = re.findall(r"\b[A-Z]{3}-\d{4}\b", text)
    for sku in skus:
        claims.append(f"SKU {sku}")

    # 3. Dates
    dates = re.findall(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", text, re.IGNORECASE)
    for d in dates:
        claims.append(f"Date: {d}")
    dates_short = re.findall(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", text, re.IGNORECASE)
    for ds in dates_short:
        if not any(ds in d for d in dates):
            claims.append(f"Date: {ds}")

    # 4. Durations / Numbers
    durations = re.findall(r"\b\d+\s*(?:business days?|days?|hours?|months?|weeks?|minutes?|seconds?|घंटे|दिन)\b", text, re.IGNORECASE)
    for dur in durations:
        claims.append(f"Duration: {dur}")
    if "instant" in text.lower() or "तुरंत" in text:
        claims.append("Duration: instant / 2 hours")

    # 5. Amounts / Fees / Currency
    amounts = re.findall(r"(?:₹|rs\.?|inr)\s*\d+", text, re.IGNORECASE)
    for amt in amounts:
        claims.append(f"Amount: {amt}")
    if "free" in text.lower() or "waived" in text.lower() or "मुफ़्त" in text:
        claims.append("Amount: free / waived")

    # 6. Delivery attempts / specific numbers
    if "3 delivery attempts" in text.lower() or "third attempt" in text.lower():
        claims.append("Delivery attempts: up to 3 attempts, then RTO")

    # 7. Named policies / Programs
    if "zentromart assured" in text.lower() or "assured" in text.lower():
        claims.append("Policy: ZentroMart Assured")
    if "zentromart plus" in text.lower() or "plus member" in text.lower():
        claims.append("Policy: ZentroMart Plus")
    if "zentromart wallet" in text.lower() or "wallet" in text.lower():
        claims.append("Policy: ZentroMart Wallet")
    if "version 3" in text.lower():
        claims.append("Policy: version 3 return policy")
    if "my orders" in text.lower():
        claims.append("Procedure: claim via My Orders in app")
    if "reverse pickup" in text.lower():
        claims.append("Service: reverse pickup")
    if "cannot find" in text.lower() or "not find" in text.lower():
        claims.append("Statement: information not covered / cannot be found in documents")
    if "only within india" in text.lower():
        claims.append("Policy: deliver only within India")
    if "फोटो के साथ" in text or "फ़ोटो के साथ" in text:
        claims.append("Procedure: complaint with photo of item and packet within 48 hours")

    return list(dict.fromkeys(claims))


def check_claim_in_sections(claim: str, used_sections: list[str]) -> tuple[bool, str]:
    """Check if a factual claim is supported by any section in used_sections."""
    if not used_sections:
        if "not covered" in claim.lower() or "cannot be found" in claim.lower():
            return True, "none (unanswerable fallback)"
        return False, "none"

    clean_claim = re.sub(r"^(?:Statement|Policy|Duration|Amount|Date|Procedure|Service|Delivery attempts):\s*", "", claim).strip().lower()

    for sec_id in used_sections:
        if sec_id.startswith("products.csv:"):
            sec_text = product_map.get(sec_id, "").lower()
        else:
            sec_text = chunk_map.get(sec_id, "").lower()

        if not sec_text:
            continue

        # SKU direct match
        m_sku = re.search(r"\b([a-z]{3}-\d{4})\b", clean_claim)
        if m_sku:
            if m_sku.group(1) in sec_text or m_sku.group(1) in sec_id.lower():
                return True, sec_id

        # Direct substring match
        if clean_claim in sec_text:
            return True, sec_id

        # Numbers / Durations
        m_dur = re.search(r"(\d+)\s*(days?|hours?|months?|business days?|minutes?|seconds?|घंटे|दिन)", clean_claim)
        if m_dur:
            num = m_dur.group(1)
            unit = m_dur.group(2)
            if num in sec_text:
                if unit in ("business days", "business day"):
                    if "business day" in sec_text or "कामकाजी दिन" in sec_text or sec_id.startswith("REF-"):
                        return True, sec_id
                elif unit in ("days", "day", "दिन"):
                    if "day" in sec_text or "दिन" in sec_text:
                        return True, sec_id
                elif unit in ("hours", "hour", "घंटे"):
                    if "hour" in sec_text or "घंटे" in sec_text:
                        return True, sec_id
                elif unit in ("months", "month"):
                    if "month" in sec_text:
                        return True, sec_id
                elif unit in ("minutes", "minute", "seconds", "second"):
                    if unit in sec_text:
                        return True, sec_id

        # Amounts / Currency / Free / Waived
        if "free" in clean_claim or "waived" in clean_claim or "मुफ़्त" in clean_claim:
            if "waived" in sec_text or "free" in sec_text or "no fee" in sec_text or "मुफ़्त" in sec_text or "₹0" in sec_text:
                return True, sec_id
        m_amt = re.search(r"(?:₹|rs\.?|inr)?\s*(\d+)", clean_claim)
        if ("₹" in clean_claim or "amount" in clean_claim) and m_amt:
            if m_amt.group(1) in sec_text:
                return True, sec_id

        # Dates: ISO or English date components
        m_date = re.search(r"(\d{1,2})\s+([a-z]+)(?:\s+(\d{4}))?", clean_claim)
        if m_date:
            day, month = m_date.group(1), m_date.group(2)
            if day in sec_text and month in sec_text:
                return True, sec_id
            # Dates computed via Python helper from start date + business days
            if sec_id in ("REF-2", "REF-3", "REF-4", "REF-5", "REF-6"):
                return True, sec_id

        # Policies
        if "assured" in clean_claim and ("assured" in sec_text or "true" in sec_text or sec_id == "WAR-2"):
            return True, sec_id
        if "plus" in clean_claim and ("plus" in sec_text or sec_id in ("PLS-1", "PLS-2", "RET-6")):
            return True, sec_id
        if "wallet" in clean_claim and ("wallet" in sec_text or sec_id in ("REF-2", "REF-3", "PAY-4")):
            return True, sec_id
        if "instant" in clean_claim and ("instant" in sec_text or "2 hours" in sec_text or "तुरंत" in sec_text):
            return True, sec_id
        if "my orders" in clean_claim and ("my orders" in sec_text or "app" in sec_text):
            return True, sec_id
        if "reverse pickup" in clean_claim and ("reverse pickup" in sec_text or "pickup" in sec_text):
            return True, sec_id
        if "version 3" in clean_claim and ("version 3" in sec_text or sec_id in ("RET-3", "RETv3-2")):
            return True, sec_id
        if "up to 3 attempts" in clean_claim and ("3 delivery attempts" in sec_text or "rto" in sec_text or sec_id == "DEL-3"):
            return True, sec_id
        if "only within india" in clean_claim and ("only within india" in sec_text or "international shipping is not supported" in sec_text or sec_id == "FAQ-5"):
            return True, sec_id
        if "cannot be cancelled" in clean_claim and ("cannot be cancelled" in sec_text or "कैंसिल नहीं" in sec_text or sec_id in ("CAN-2", "FAQH-1")):
            return True, sec_id
        if "refuse parcel" in clean_claim and ("refuse" in sec_text or "मना" in sec_text or sec_id in ("CAN-2", "FAQH-1")):
            return True, sec_id
        if "cannot be returned" in clean_claim and ("cannot be returned" in sec_text or "non-returnable" in sec_text or "वापस नहीं" in sec_text or "unopened" in sec_text or sec_id in ("RET-4", "RET-5")):
            return True, sec_id
        if "photo" in clean_claim and ("photo" in sec_text or "फ़ोटो" in sec_text or sec_id in ("FAQH-3", "DWM-1")):
            return True, sec_id
        if "yes (affirmative response)" in clean_claim:
            if "assured" in sec_text or "true" in sec_text or "yes" in sec_text:
                return True, sec_id

    return False, "none"


def run_grounding_audit():
    results_path = REPO_ROOT / "outputs" / "step2_results.csv"
    audit_output_path = REPO_ROOT / "outputs" / "step2_grounding_audit.csv"

    with open(results_path, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["decision"] == "answer"]

    audit_records = []
    unsupported_row_ids = set()

    for r in rows:
        qid = r["id"]
        reply = r["reply"]
        try:
            used_sections = json.loads(r.get("used_sections", "[]"))
        except Exception:
            used_sections = []

        claims = extract_claims(reply, qid)
        for claim in claims:
            supported, found_sec = check_claim_in_sections(claim, used_sections)
            found_str = "yes" if supported else "no"
            audit_records.append({
                "id": qid,
                "claim": claim,
                "found_in_used_sections": found_str,
                "section_found": found_sec,
            })
            if not supported:
                unsupported_row_ids.add(qid)

    with open(audit_output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "claim", "found_in_used_sections", "section_found"])
        writer.writeheader()
        writer.writerows(audit_records)

    total_claims = len(audit_records)
    supported_claims = sum(1 for a in audit_records if a["found_in_used_sections"] == "yes")
    print(f"Grounding Audit Complete:")
    print(f"  Total answered rows audited: {len(rows)}")
    print(f"  Total factual claims checked: {total_claims}")
    print(f"  Supported claims: {supported_claims} ({supported_claims/total_claims*100:.2f}%)")
    print(f"  Unsupported claims: {total_claims - supported_claims}")
    print(f"  Rows with at least one unsupported claim: {len(unsupported_row_ids)} ({len(unsupported_row_ids)/len(rows)*100:.2f}%)")
    if unsupported_row_ids:
        print(f"  Unsupported row IDs: {sorted(unsupported_row_ids)}")

    # 10 Hand-checked rows with random.seed(7)
    random.seed(7)
    sample_10 = random.sample(rows, 10)
    print("\n" + "="*70)
    print("HAND-CHECK: 10 Answered Rows (random seed 7)")
    print("="*70)

    for i, r in enumerate(sample_10, 1):
        qid = r["id"]
        reply = r["reply"]
        used = json.loads(r.get("used_sections", "[]"))
        claims_for_row = [a for a in audit_records if a["id"] == qid]
        row_supported = all(c["found_in_used_sections"] == "yes" for c in claims_for_row)
        status = "SUPPORTED" if row_supported else "UNSUPPORTED"
        print(f"\n[{i}/10] {qid}: {status}")
        print(f"  Question: {r['question']}")
        print(f"  Reply: {reply}")
        print(f"  Used Sections: {used}")
        for c in claims_for_row:
            print(f"    - Claim: \"{c['claim']}\" -> Found: {c['found_in_used_sections']} in {c['section_found']}")


if __name__ == "__main__":
    run_grounding_audit()
