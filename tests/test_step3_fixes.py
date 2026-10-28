"""Offline regression tests for Step 3 fixes B1-B4.

No test constructs an LLM client or calls Gemini, Gnani, or any network API.
Tickets are redirected to a pytest temp directory; the dataset is read-only.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.conversation import (  # noqa: E402
    ConversationAgent,
    ConversationState,
    _build_order_context,
    is_escalation_requested,
)
import tools.tools as dataset_tools  # noqa: E402 - DATASET_DIR is on sys.path now
from src.data import (  # noqa: E402
    REFERENCE_DATE_OBJ,
    business_days_between,
    get_order,
)
from src.number_words import normalise_digit_words  # noqa: E402


@pytest.fixture()
def ticket_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send create_ticket output to a temp file, never the dataset folder."""
    path = tmp_path / "tickets.csv"
    monkeypatch.setattr(dataset_tools, "_TICKETS", str(path))
    return path


def _read_tickets(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# B1: Rule 5 uses return_status and return_refund_initiated_on
# ---------------------------------------------------------------------------


def test_b1_pc010_refund_delayed_escalates() -> None:
    order = get_order("16640894")
    assert order is not None
    assert order["return_status"] == "refund_initiated"
    assert order["return_refund_initiated_on"] == "2026-07-20"
    state = ConversationState(
        order_id="16640894", verified=True, order_data=order
    )

    assert is_escalation_requested("Mera refund kahan hai?", state) == (
        True,
        "refund_delayed",
        "refund",
    )


def test_b1_upi_three_business_days_does_not_escalate() -> None:
    # 2026-10-19 is a warehouse holiday, so 2026-10-15 is exactly 3 business
    # days before the 2026-10-20 reference date.
    assert business_days_between("2026-10-15", REFERENCE_DATE_OBJ) == 3
    order = {
        "order_id": "99999999",
        "payment_method": "upi",
        "return_status": "refund_initiated",
        "return_refund_initiated_on": "2026-10-15",
    }
    state = ConversationState(
        order_id="99999999", verified=True, order_data=order
    )

    assert is_escalation_requested("Where is my refund?", state) == (
        False,
        None,
        None,
    )


def test_b1_missing_start_date_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    order = {
        "order_id": "88888888",
        "payment_method": "upi",
        "return_status": "pending",
    }
    state = ConversationState(
        order_id="88888888", verified=True, order_data=order
    )
    with caplog.at_level(logging.WARNING, logger="voiceops.conversation"):
        result = is_escalation_requested("Where is my refund?", state)

    assert result == (False, None, None)
    assert "return_refund_initiated_on" in caplog.text


# ---------------------------------------------------------------------------
# B2: human requests hand over before identity verification
# ---------------------------------------------------------------------------


def test_b2_devanagari_human_request_turn_one(ticket_log: Path) -> None:
    agent = ConversationAgent()
    result = agent.respond(
        "क्या मैं किसी इंसान से बात कर सकती हूँ?",
        input_modality="text",
    )

    assert result["decision"] == "escalate"
    assert result["state"]["order_id"] is None
    assert result["state"]["handover_created"] is True
    assert any(call["name"] == "create_ticket" for call in result["tool_calls"])

    tickets = _read_tickets(ticket_log)
    assert len(tickets) == 1
    assert tickets[0]["reason"] == "human_requested"
    assert tickets[0]["order_id"] == ""
    assert "इंसान" in tickets[0]["summary"]


def test_b2_human_request_after_verification(ticket_log: Path) -> None:
    agent = ConversationAgent()
    agent.respond(
        "नमस्ते, मुझे अपने ऑर्डर के बारे में मदद चाहिए।", input_modality="text"
    )
    agent.respond("ऑर्डर नंबर है 3711 3676।", input_modality="text")
    verified = agent.respond("आख़िरी चार अंक 1581 हैं।", input_modality="text")
    assert verified["state"]["verified"] is True

    result = agent.respond(
        "क्या मैं किसी इंसान से बात कर सकती हूँ?", input_modality="text"
    )
    assert result["decision"] == "escalate"

    tickets = _read_tickets(ticket_log)
    assert len(tickets) == 1
    assert tickets[0]["reason"] == "human_requested"
    assert tickets[0]["order_id"] == "37113676"


# ---------------------------------------------------------------------------
# B3: shipped_on is part of the verified-facts block
# ---------------------------------------------------------------------------


def test_b3_pc013_shipped_on_in_context() -> None:
    order = get_order("86152121")
    assert order is not None
    assert order["shipped_on"] == "2026-10-09"

    context = _build_order_context(order)
    assert "- shipped_on: 2026-10-09" in context


# ---------------------------------------------------------------------------
# B4: spoken digit words, voice only
# ---------------------------------------------------------------------------


def test_b4_spoken_digits_become_order_id() -> None:
    agent = ConversationAgent()
    result = agent.respond(
        "my order number is six nine five two oh eight six seven",
        input_modality="voice",
    )

    assert result["state"]["order_id"] == "69520867"
    assert result["state"]["verified"] is False


def test_b4_spoken_pincode_verifies() -> None:
    agent = ConversationAgent()
    agent.respond("Hello, I need help with an order.", input_modality="text")
    agent.respond("My order number is 69520867.", input_modality="text")
    result = agent.respond(
        "my pincode is one six zero zero one seven",
        input_modality="voice",
    )

    assert result["state"]["verified"] is True


def test_b4_digits_unchanged() -> None:
    assert normalise_digit_words("6952") == "6952"


def test_b4_spoken_digits_in_text_mode_not_converted() -> None:
    agent = ConversationAgent()
    result = agent.respond(
        "my order number is six nine five two oh eight six seven",
        input_modality="text",
    )

    assert result["state"]["order_id"] is None
