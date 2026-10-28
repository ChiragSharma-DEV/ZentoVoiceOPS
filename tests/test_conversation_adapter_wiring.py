"""Offline wiring tests for the Step 3 RAG answer adapter.

These tests use a fake retriever and a stubbed LLM response. They make no API
calls and verify that policy turns delegate to src.rag.answer_question exactly
once with the correct modality and order context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.conversation import ConversationAgent, ConversationState
from src.llm import LLMResponse
from src.rag import answer_question as real_answer_question


class FakeRetriever:
    """Minimal retriever that records calls and returns no chunks."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        enable_rewrite: Optional[bool] = None,
        is_asr: bool = False,
    ) -> List[Tuple[Any, float]]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "enable_rewrite": enable_rewrite,
                "is_asr": is_asr,
            }
        )
        return []


def _verified_agent(retriever: FakeRetriever) -> ConversationAgent:
    state = ConversationState(
        order_id="12345678",
        verified=True,
        order_data={
            "payment_method": "upi",
            "delivered_on": "2026-10-12",
            "status": "delivered",
            "is_plus": True,
            "items": [{"return_category": "fashion"}],
        },
    )
    return ConversationAgent(
        state=state,
        retriever=retriever,
        top_k=7,
        enable_rewrite=True,
    )


def test_policy_turn_calls_answer_question_once_with_order_context(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = FakeRetriever()
    agent = _verified_agent(retriever)

    import src.rag as rag

    calls: List[Dict[str, Any]] = []

    def fake_answer_question(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return real_answer_question(**kwargs)

    monkeypatch.setattr(rag, "answer_question", fake_answer_question)
    monkeypatch.setattr(
        rag,
        "complete_with_meta",
        lambda system, user: LLMResponse(
            '{"decision": "answer", "reply": "CANNED_GROUNDED_REPLY", '
            '"used_sections": ["RET-2"]}',
            finish_reason="STOP",
            truncated=False,
        ),
    )

    result = agent.respond("What is the return window?")

    assert len(calls) == 1
    assert len(retriever.calls) == 1
    assert calls[0]["retriever"] is retriever
    assert calls[0]["top_k"] == 7
    assert calls[0]["enable_rewrite"] is True
    assert calls[0]["is_asr"] is False

    question = calls[0]["question"]
    assert "payment_method: upi" in question
    assert "delivery_date: 2026-10-12" in question
    assert "status: delivered" in question
    assert "is_plus: true" in question
    assert "item_categories: fashion" in question

    assert result["decision"] == "answer"
    assert result["reply"] == "CANNED_GROUNDED_REPLY"
    assert result["used_sections"] == ["RET-2"]


def test_text_and_voice_turns_forward_is_asr(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.rag as rag

    calls: List[Dict[str, Any]] = []

    def fake_answer_question(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return real_answer_question(**kwargs)

    monkeypatch.setattr(rag, "answer_question", fake_answer_question)
    monkeypatch.setattr(
        rag,
        "complete_with_meta",
        lambda system, user: LLMResponse(
            '{"decision": "answer", "reply": "CANNED_GROUNDED_REPLY", '
            '"used_sections": ["RET-2"]}',
            finish_reason="STOP",
            truncated=False,
        ),
    )

    text_agent = _verified_agent(FakeRetriever())
    text_agent.respond("What is the return window?", input_modality="text")

    voice_agent = _verified_agent(FakeRetriever())
    voice_agent.respond("What is the return window?", input_modality="voice")

    assert len(calls) == 2
    assert calls[0]["is_asr"] is False
    assert calls[1]["is_asr"] is True


def test_adapter_raises_when_llm_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.rag as rag

    def failing_completion(system: str, user: str) -> LLMResponse:
        raise RuntimeError("LLM unavailable: missing API key")

    monkeypatch.setattr(rag, "complete_with_meta", failing_completion)

    agent = _verified_agent(FakeRetriever())
    with pytest.raises(RuntimeError, match="answer_question failed"):
        agent.respond("What is the return window?")


def test_adapter_propagates_answer_question_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.rag as rag

    def failing_answer_question(**kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("answer_question exploded")

    monkeypatch.setattr(rag, "answer_question", failing_answer_question)

    agent = _verified_agent(FakeRetriever())
    with pytest.raises(RuntimeError, match="answer_question exploded"):
        agent.respond("What is the return window?")
