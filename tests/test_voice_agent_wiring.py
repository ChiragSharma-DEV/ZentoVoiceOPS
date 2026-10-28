"""Offline wiring tests for the Step 4 voice runner.

No Gemini or Gnani call is made. src.rag.answer_question is patched exactly
where rag_answer_adapter looks it up (the src.rag module attribute), so both
turns exercise the real ConversationAgent default adapter path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

import eval.run_voice as run_voice
from src.conversation import ConversationState


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


def _verified_state() -> ConversationState:
    return ConversationState(
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


def test_build_agent_uses_voice_defaults() -> None:
    retriever = FakeRetriever()
    agent = run_voice.build_agent(retriever, state=_verified_state())

    assert agent.top_k == 10
    assert agent.enable_rewrite is False
    assert agent.input_modality == "voice"
    assert agent.retriever is retriever
    assert agent._adapter_accepts_is_asr is True


def test_voice_and_text_policy_turns_forward_is_asr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.rag as rag

    calls: List[Dict[str, Any]] = []

    def fake_answer_question(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {
            "decision": "answer",
            "reply": "CANNED_VOICE_GROUNDED_REPLY",
            "used_sections": ["RET-2"],
            "retrieved_items": [("RET-2", 0.9)],
            "finish_reason": "STOP",
        }

    monkeypatch.setattr(rag, "answer_question", fake_answer_question)

    retriever = FakeRetriever()
    agent = run_voice.build_agent(retriever, state=_verified_state())

    voice_result = run_voice.run_agent_turn(
        agent, "What is the return window?", input_modality="voice"
    )
    text_result = run_voice.run_agent_turn(
        agent, "What is the return window?", input_modality="text"
    )

    assert len(calls) == 2
    assert calls[0]["is_asr"] is True
    assert calls[1]["is_asr"] is False
    assert calls[0]["retriever"] is retriever
    assert calls[0]["top_k"] == 10
    assert calls[0]["enable_rewrite"] is False

    assert voice_result["decision"] == "answer"
    assert voice_result["reply"] == "CANNED_VOICE_GROUNDED_REPLY"
    assert text_result["reply"] == "CANNED_VOICE_GROUNDED_REPLY"

    print(
        f"\nvoice turn: is_asr={calls[0]['is_asr']} "
        f"top_k={calls[0]['top_k']} "
        f"enable_rewrite={calls[0]['enable_rewrite']} "
        f"decision={voice_result['decision']}"
    )
    print(
        f"text turn:  is_asr={calls[1]['is_asr']} "
        f"top_k={calls[1]['top_k']} "
        f"enable_rewrite={calls[1]['enable_rewrite']} "
        f"decision={text_result['decision']}"
    )
    print(
        f"assertions: voice is_asr is True -> {calls[0]['is_asr'] is True}; "
        f"text is_asr is False -> {calls[1]['is_asr'] is False}"
    )


def test_main_skips_without_run_live(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("RUN_LIVE", raising=False)

    assert run_voice.main() == 0

    output = capsys.readouterr().out
    assert "SKIP" in output
    assert "RUN_LIVE=1" in output
