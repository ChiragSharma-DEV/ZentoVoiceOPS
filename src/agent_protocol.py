"""Agent protocol interface for VoiceOps 2026.

Defines the contract that any text or RAG agent must implement to be plugged into
the voice layer or evaluation harness.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Agent(Protocol):
    """Protocol defining the interface for VoiceOps text agents."""

    def respond(self, customer_text: str, language_code: str) -> dict[str, Any]:
        """Process customer text and return a response dict.

        Args:
            customer_text: Customer's transcribed speech or input query.
            language_code: BCP-47 language code (e.g. 'en-IN', 'hi-IN').

        Returns:
            dict containing:
                - reply (str): Customer-facing text reply.
                - decision (str): One of 'answer', 'clarify', 'escalate', 'refuse'.
                - used_sections (list[str]): List of section IDs used to answer.
        """
        ...
