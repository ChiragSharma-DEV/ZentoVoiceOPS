"""Phase 2G modality-gate test for Fix 5 ASR normalisation."""

from __future__ import annotations

import pytest

import src.retrieval as retrieval
from src.retrieval import HybridRetriever


def _empty_retriever() -> HybridRetriever:
    """Build a retriever without loading embeddings or an external model."""
    retriever = object.__new__(HybridRetriever)
    retriever.chunks = []
    retriever.bm25 = None
    retriever.encoder = None
    retriever.faiss_index = None
    retriever.use_query_rewrite = False
    return retriever


def test_fix5_modality_gate(monkeypatch):
    raw = "ELE zero zero one two ka return window kya hai?"
    expected = "ELE-0012 ka return window kya hai?"
    calls = []
    original = retrieval.normalise_asr_text

    def spy(text):
        result = original(text)
        calls.append((text, result))
        return result

    monkeypatch.setattr(retrieval, "normalise_asr_text", spy)
    monkeypatch.setattr(retrieval, "FIX5_ASR_NORMALISE", True)
    retriever = _empty_retriever()

    # Text path: the normaliser must not be called.
    calls.clear()
    retriever.retrieve(raw, top_k=1, enable_rewrite=False, is_asr=False)
    assert calls == []

    # ASR path: the normaliser must be called exactly once and change the token.
    calls.clear()
    retriever.retrieve(raw, top_k=1, enable_rewrite=False, is_asr=True)
    assert calls == [(raw, expected)]

    # Fix 5 off: even the ASR path must leave the query unchanged.
    monkeypatch.setattr(retrieval, "FIX5_ASR_NORMALISE", False)
    calls.clear()
    retriever.retrieve(raw, top_k=1, enable_rewrite=False, is_asr=True)
    assert calls == []
