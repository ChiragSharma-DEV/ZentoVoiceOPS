"""Offline tests for the Step 3 C1-C4 and D1-D4 gate and policy checks.

No test constructs an LLM client or calls Gemini, Gnani, or any network API.
The RAG answer function is replaced with a local fake; tickets are redirected
to a pytest temp directory so the dataset stays read-only.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.conversation import (  # noqa: E402
    ConversationAgent,
    ConversationState,
    _has_claim_or_action_cue,
    _has_order_specific_cue,
    _is_human_request,
    _is_policy_question,
    _refund_max_business_days,
    is_escalation_requested,
)
from src.data import REFERENCE_DATE_OBJ, business_days_between  # noqa: E402
import tools.tools as dataset_tools  # noqa: E402 - DATASET_DIR is on sys.path now


class FakeRetriever:
    """Minimal retriever: records nothing and returns no chunks."""

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        enable_rewrite: Optional[bool] = None,
        is_asr: bool = False,
    ) -> List[Tuple[Any, float]]:
        return []


@pytest.fixture()
def ticket_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send create_ticket output to a temp file, never the dataset folder."""
    path = tmp_path / "tickets.csv"
    monkeypatch.setattr(dataset_tools, "_TICKETS", str(path))
    return path


def _stub_adapter(**kwargs: Any) -> Dict[str, Any]:
    return {"decision": "answer", "reply": "STUB_POLICY_REPLY", "used_sections": []}


def _install_fake_policy(
    monkeypatch: pytest.MonkeyPatch, calls: List[Dict[str, Any]]
) -> None:
    import src.rag as rag

    def fake_answer_question(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {
            "decision": "answer",
            "reply": "FAKE_GROUNDED_POLICY",
            "used_sections": [],
        }

    monkeypatch.setattr(rag, "answer_question", fake_answer_question)


def _read_tickets(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# C1: policy route before the verification gate (IDV-1 line 15)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Can I return the Pagecraft notebook?",
        "Pagecraft notebook ka return kitne din mein hota hai?",
    ],
)
def test_c1_general_policy_question_routes_without_order_facts(
    message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: List[Dict[str, Any]] = []
    _install_fake_policy(monkeypatch, calls)
    agent = ConversationAgent(retriever=FakeRetriever(), input_modality="text")

    result = agent.respond(message)

    assert len(calls) == 1
    assert calls[0]["question"] == message
    assert "VERIFIED ORDER FACTS" not in calls[0]["question"]
    assert calls[0]["is_asr"] is False
    assert result["decision"] == "answer"
    assert result["reply"] == "FAKE_GROUNDED_POLICY"
    assert result["state"]["order_id"] is None


def test_c1_policy_route_forwards_voice_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _install_fake_policy(monkeypatch, calls)
    agent = ConversationAgent(retriever=FakeRetriever(), input_modality="voice")

    agent.respond("Can I return the Pagecraft notebook?")

    assert len(calls) == 1
    assert calls[0]["is_asr"] is True


@pytest.mark.parametrize(
    "message",
    [
        "Mera refund kahan hai?",
        "Where is my order?",
    ],
)
def test_c1_order_specific_turns_stay_at_the_gate(
    message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: List[Dict[str, Any]] = []
    _install_fake_policy(monkeypatch, calls)
    agent = ConversationAgent(retriever=FakeRetriever())

    result = agent.respond(message)

    assert calls == []
    assert result["decision"] == "clarify"
    assert result["state"]["order_id"] is None
    assert result["used_sections"] == ["STY-1"]


def test_c1_order_id_turn_parses_id_and_stays_at_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _install_fake_policy(monkeypatch, calls)
    agent = ConversationAgent(retriever=FakeRetriever())

    result = agent.respond("My order number is 69520867")

    assert calls == []
    assert result["decision"] == "clarify"
    assert result["state"]["order_id"] == "69520867"
    assert result["state"]["verified"] is False
    assert result["used_sections"] == ["IDV-1", "IDV-2"]
    assert any(call["name"] == "lookup_order" for call in result["tool_calls"])


def test_c1_four_digit_sku_is_not_an_order_cue() -> None:
    assert _has_order_specific_cue("Pagecraft notebook ELE-0012 ka rate kya hai?") is False


# ---------------------------------------------------------------------------
# D1: claim/action turns reach the identity gate, not the policy route
# ---------------------------------------------------------------------------

D1_CASES = [
    ("Product damaged aaya", True, False),
    ("Order late aaya, mujhe voucher chahiye", True, False),
    ("Refund abhi tak nahi aaya", True, False),
    ("Cancel karna hai", True, False),
    ("Can I return the Pagecraft notebook?", False, True),
    ("Pagecraft notebook ka return kitne din mein hota hai?", False, True),
]


@pytest.mark.parametrize("message,claim,policy", D1_CASES)
def test_d1_classification(
    message: str, claim: bool, policy: bool
) -> None:
    assert _has_claim_or_action_cue(message) is claim
    assert _is_policy_question(message) is policy


@pytest.mark.parametrize("message", [case[0] for case in D1_CASES[:4]])
def test_d1_claim_turns_reach_identity_gate(
    message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: List[Dict[str, Any]] = []
    _install_fake_policy(monkeypatch, calls)
    agent = ConversationAgent(retriever=FakeRetriever())

    result = agent.respond(message)

    assert calls == []
    assert result["decision"] == "clarify"
    assert result["state"]["order_id"] is None
    assert result["used_sections"] == ["STY-1"]


# ---------------------------------------------------------------------------
# C2: human-request false positives
# ---------------------------------------------------------------------------

C2_CASES = [
    ("The delivery person was rude", False),
    ("Is this an agent-approved product?", False),
    ("aadmi kal aaya tha, sab theek tha", False),
    ("vyakti ne package achha diya", False),
    ("Pagecraft notebook ka rate kya hai?", False),
    ("क्या मैं किसी इंसान से बात कर सकती हूँ?", True),
    ("I want to talk to a human agent", True),
    ("kisi insaan se baat karni hai", True),
    ("connect me to a person", True),
]


@pytest.mark.parametrize("message,expected", C2_CASES)
def test_c2_human_request_detection(
    message: str, expected: bool, ticket_log: Path
) -> None:
    assert _is_human_request(message) is expected

    agent = ConversationAgent(answer_adapter=_stub_adapter)
    result = agent.respond(message)

    assert (result["decision"] == "escalate") is expected
    assert any(
        call["name"] == "create_ticket" for call in result["tool_calls"]
    ) is expected
    assert result["state"]["handover_created"] is expected

    tickets = _read_tickets(ticket_log) if ticket_log.exists() else []
    assert len(tickets) == (1 if expected else 0)


# ---------------------------------------------------------------------------
# C3: cancellation and RTO refunds use the same Rule 5 thresholds
# ---------------------------------------------------------------------------


def _refund_order(**overrides: Any) -> Dict[str, Any]:
    order: Dict[str, Any] = {"order_id": "99999999", "payment_method": "card"}
    order.update(overrides)
    return order


@pytest.mark.parametrize(
    "date_field,method_field,initiated,expected",
    [
        ("cancellation_refund_initiated_on", "cancellation_refund_method", "2026-10-07", False),
        ("cancellation_refund_initiated_on", "cancellation_refund_method", "2026-10-06", True),
        ("rto_refund_initiated_on", "rto_refund_method", "2026-10-07", False),
        ("rto_refund_initiated_on", "rto_refund_method", "2026-10-06", True),
    ],
)
def test_c3_cancellation_and_rto_refund_boundaries(
    date_field: str, method_field: str, initiated: str, expected: bool
) -> None:
    elapsed = business_days_between(initiated, REFERENCE_DATE_OBJ)
    assert elapsed == (11 if expected else 10)

    order = _refund_order(**{date_field: initiated, method_field: "card"})
    state = ConversationState(order_id="99999999", verified=True, order_data=order)

    result = is_escalation_requested("Where is my refund?", state)

    assert result == (
        (True, "refund_delayed", "refund") if expected else (False, None, None)
    )


# ---------------------------------------------------------------------------
# D4: Rule 5 checks every present refund event, not only the first one
# ---------------------------------------------------------------------------


def test_d4_overdue_cancellation_refund_escalates_with_current_return_refund() -> None:
    # Return refund is current (3 business days, UPI max 3 + 3 = 6).
    assert business_days_between("2026-10-15", REFERENCE_DATE_OBJ) == 3
    # Cancellation refund is overdue (11 business days, card max 7 + 3 = 10).
    assert business_days_between("2026-10-06", REFERENCE_DATE_OBJ) == 11

    order = _refund_order(
        return_refund_initiated_on="2026-10-15",
        return_refund_method="upi",
        cancellation_refund_initiated_on="2026-10-06",
        cancellation_refund_method="card",
    )
    state = ConversationState(order_id="99999999", verified=True, order_data=order)

    assert is_escalation_requested("Where is my refund?", state) == (
        True,
        "refund_delayed",
        "refund",
    )


# ---------------------------------------------------------------------------
# C4: wallet and NEFT timelines come from REF-2
# ---------------------------------------------------------------------------


def test_c4_refund_max_days_match_ref2_table() -> None:
    assert _refund_max_business_days("wallet") == 0
    assert _refund_max_business_days("cod_wallet") == 0
    assert _refund_max_business_days("upi") == 3
    assert _refund_max_business_days("netbanking") == 5
    assert _refund_max_business_days("card") == 7
    assert _refund_max_business_days("cod_bank") == 7
    assert _refund_max_business_days("neft") == 7
    assert _refund_max_business_days("") == 5


@pytest.mark.parametrize(
    "method,initiated,expected",
    [
        ("wallet", "2026-10-15", False),  # 3 elapsed = 0 + 3 tolerance
        ("wallet", "2026-10-14", True),   # 4 elapsed > 3
        ("cod_bank", "2026-10-07", False),  # 10 elapsed = 7 + 3 tolerance
        ("cod_bank", "2026-10-06", True),   # 11 elapsed > 10
    ],
)
def test_c4_wallet_and_neft_refund_boundaries(
    method: str, initiated: str, expected: bool
) -> None:
    order = _refund_order(
        return_refund_initiated_on=initiated,
        return_refund_method=method,
        payment_method="card",  # must not override the refund-method timeline
    )
    state = ConversationState(order_id="99999999", verified=True, order_data=order)

    result = is_escalation_requested("Where is my refund?", state)

    assert result == (
        (True, "refund_delayed", "refund") if expected else (False, None, None)
    )


# ---------------------------------------------------------------------------
# E1: Devanagari order and claim cues reach the gate, policy questions route
# ---------------------------------------------------------------------------

E1_GATE_CASES = [
    "नमस्ते, मुझे अपने ऑर्डर के बारे में मदद चाहिए।",
    "प्रोडक्ट टूटा हुआ आया",
    "मेरा ऑर्डर देर से आया",
]


@pytest.mark.parametrize("message", E1_GATE_CASES)
def test_e1_devanagari_turns_reach_the_gate(
    message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: List[Dict[str, Any]] = []
    _install_fake_policy(monkeypatch, calls)
    agent = ConversationAgent(retriever=FakeRetriever())

    result = agent.respond(message)

    assert calls == []
    assert _has_claim_or_action_cue(message) or _has_order_specific_cue(message)
    assert result["decision"] == "clarify"
    assert result["state"]["order_id"] is None
    assert result["used_sections"] == ["STY-1"]


def test_e1_devanagari_policy_question_takes_the_policy_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "पेजक्राफ्ट नोटबुक रिटर्न कर सकते हैं?"
    calls: List[Dict[str, Any]] = []
    _install_fake_policy(monkeypatch, calls)
    agent = ConversationAgent(retriever=FakeRetriever())

    result = agent.respond(message)

    assert len(calls) == 1
    assert calls[0]["question"] == message
    assert result["decision"] == "answer"
    assert result["reply"] == "FAKE_GROUNDED_POLICY"
    assert result["state"]["order_id"] is None
