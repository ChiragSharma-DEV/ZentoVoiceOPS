"""Offline test suite for VoiceOps Step 3.

Verifies T1 through T5 with zero LLM calls.
Selects test orders dynamically from data at runtime (no hardcoding).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure voiceops repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from src.data import DATASET_DIR, ORDERS_DF, CUSTOMERS_DF
from src.order_tools import (
    lookup_order_unverified,
    lookup_order_verified,
    handover,
)


def run_tests() -> None:
    print("=" * 60)
    print("RUNNING STEP 3 OFFLINE TESTS (T1 to T5 - ZERO LLM CALLS)")
    print("=" * 60)

    # Dynamically select an order and matching customer from dataset at runtime
    first_order = ORDERS_DF.iloc[0]
    sample_oid = str(first_order["order_id"])
    sample_cid = str(first_order["customer_id"])
    sample_pincode = str(first_order["delivery_pincode"])

    cust_match = CUSTOMERS_DF[CUSTOMERS_DF["customer_id"] == sample_cid].iloc[0]
    sample_phone = str(cust_match["phone"])
    sample_phone_suffix = sample_phone[-4:]

    print(f"\n[SETUP] Dynamically selected test order:")
    print(f"  Order ID        : {sample_oid}")
    print(f"  Customer ID     : {sample_cid}")
    print(f"  Delivery Pincode: {sample_pincode}")
    print(f"  Phone Suffix    : {sample_phone_suffix}")

    # -------------------------------------------------------------
    # T1: lookup_order_unverified on a real order ID
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("T1: lookup_order_unverified check")
    print("-" * 60)
    t1_res = lookup_order_unverified(sample_oid)
    print("Raw output:", t1_res)

    has_verification_req = t1_res.get("verification_required") is True
    no_items = "items" not in t1_res
    no_status = "status" not in t1_res
    no_total = "total_inr" not in t1_res

    assert has_verification_req, "T1 failed: verification_required is not True"
    assert no_items, "T1 failed: items leaked in unverified call"
    assert no_status, "T1 failed: status leaked in unverified call"
    assert no_total, "T1 failed: total_inr leaked in unverified call"

    print("T1 Checks:")
    print(f"  verification_required is True : {has_verification_req}")
    print(f"  items withheld                : {no_items}")
    print(f"  status withheld               : {no_status}")
    print(f"  total_inr withheld            : {no_total}")
    print("RESULT: PASS")

    # -------------------------------------------------------------
    # T2: Wrong verification value
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("T2: Wrong verification value check")
    print("-" * 60)
    wrong_value = "0000" if sample_phone_suffix != "0000" else "9999"
    t2_res = lookup_order_verified(sample_oid, wrong_value)
    print("Raw output:", t2_res)

    verified_false = t2_res.get("verified") is False
    no_items_t2 = "items" not in t2_res
    no_status_t2 = "status" not in t2_res
    no_total_t2 = "total_inr" not in t2_res

    assert verified_false, "T2 failed: verified is not False"
    assert no_items_t2, "T2 failed: items leaked on failed verification"
    assert no_status_t2, "T2 failed: status leaked on failed verification"
    assert no_total_t2, "T2 failed: total_inr leaked on failed verification"

    print("T2 Checks:")
    print(f"  verified is False             : {verified_false}")
    print(f"  items withheld                : {no_items_t2}")
    print(f"  status withheld               : {no_status_t2}")
    print(f"  total_inr withheld            : {no_total_t2}")
    print("RESULT: PASS")

    # -------------------------------------------------------------
    # T3: Correct phone suffix and pincode verification
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("T3: Correct phone suffix and pincode verification check")
    print("-" * 60)
    # Check phone suffix
    t3_phone_res = lookup_order_verified(sample_oid, sample_phone_suffix)
    phone_verified = t3_phone_res.get("verified") is True and t3_phone_res.get("found") is True
    has_items_phone = "items" in t3_phone_res and isinstance(t3_phone_res["items"], list)

    print(f"Sub-test 1 (Phone Suffix '{sample_phone_suffix}'):")
    print(f"  found & verified : {phone_verified}")
    print(f"  status revealed  : {t3_phone_res.get('status')}")
    print(f"  items count      : {len(t3_phone_res.get('items', []))}")
    print(f"  total_inr        : {t3_phone_res.get('total_inr')}")
    assert phone_verified, "T3 phone verification failed"
    assert has_items_phone, "T3 missing items on phone verification"

    # Check delivery pincode
    t3_pin_res = lookup_order_verified(sample_oid, sample_pincode)
    pin_verified = t3_pin_res.get("verified") is True and t3_pin_res.get("found") is True
    has_items_pin = "items" in t3_pin_res and isinstance(t3_pin_res["items"], list)

    print(f"\nSub-test 2 (Delivery Pincode '{sample_pincode}'):")
    print(f"  found & verified : {pin_verified}")
    print(f"  status revealed  : {t3_pin_res.get('status')}")
    print(f"  items count      : {len(t3_pin_res.get('items', []))}")
    print(f"  total_inr        : {t3_pin_res.get('total_inr')}")
    assert pin_verified, "T3 pincode verification failed"
    assert has_items_pin, "T3 missing items on pincode verification"

    print("RESULT: PASS")

    # -------------------------------------------------------------
    # T4: Handover summary under 20 characters
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("T4: Handover summary under 20 characters check")
    print("-" * 60)
    short_summary = "Damage reported"  # 15 characters (< 20)
    t4_res = handover(summary=short_summary, order_id=sample_oid)
    print(f"Summary input : '{short_summary}' (length: {len(short_summary)})")
    print("Raw output    :", t4_res)

    created_false = t4_res.get("created") is False
    has_err = "error" in t4_res
    assert created_false, "T4 failed: ticket created with short summary"
    assert has_err, "T4 failed: missing error explanation"

    print("T4 Checks:")
    print(f"  created is False : {created_false}")
    print(f"  error message    : {t4_res.get('error')}")
    print("RESULT: PASS")

    # -------------------------------------------------------------
    # T5: Handover with valid summary (>= 20 characters)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("T5: Handover with valid summary check")
    print("-" * 60)
    valid_summary = (
        f"Verified customer reports damaged item on order {sample_oid}; "
        f"value exceeds threshold; escalated to Priority Resolution."
    )
    t5_res = handover(
        summary=valid_summary,
        order_id=sample_oid,
        reason="high_value_damage",
        intent="damage",
    )
    print(f"Summary input : '{valid_summary}' (length: {len(valid_summary)})")
    print("Raw output    :", t5_res)

    created_true = t5_res.get("created") is True
    has_ticket_id = bool(t5_res.get("ticket_id"))
    assert created_true, "T5 failed: ticket was not created"
    assert has_ticket_id, "T5 failed: missing ticket_id"

    print("T5 Checks:")
    print(f"  created is True  : {created_true}")
    print(f"  ticket_id        : {t5_res.get('ticket_id')}")
    print("Note: Appended ticket row to dataset tools/tickets_created.csv.")
    print("RESULT: PASS")

    # Cleanup test ticket created in T5 so dataset remains pristine
    tickets_path = Path(DATASET_DIR) / "tools" / "tickets_created.csv"
    if tickets_path.exists():
        tickets_path.unlink()
        print("[CLEANUP] Deleted test tickets file from dataset to keep DATASET_DIR pristine.")

    print("\n" + "=" * 60)
    print("ALL 5 OFFLINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
