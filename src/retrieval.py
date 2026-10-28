"""Hybrid Retrieval (BM25 + Dense FAISS) with Query Rewriting and Precedence for VoiceOps 2026."""

from __future__ import annotations
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from src.kb import Chunk, load_and_chunk_kb, PRECEDENCE_TIERS
from src.data import get_product
from src.asr_normalise import normalise_asr_text

# Embedding model identifier
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Phase 2D/2E Feature Switches
FIX1_CATEGORY_EXPANSION: bool = False
FIX1_SKU_GUARD: bool = False
FIX5_ASR_NORMALISE: bool = True  # Active for ASR input only per Phase 2G frozen config

QUERY_REWRITE_SYSTEM_PROMPT = """You are a search query rewriter for ZentroMart customer support knowledge base.
Convert the customer's question (written in English, Hindi, or Hinglish) into a concise English search query naming official policy terms.
Use likely policy keywords such as:
- reverse pickup, pickup fee, return window, return eligibility, condition of return
- cancellation before shipping, refused delivery, RTO
- refund timeline, UPI, credit card, net banking, wallet, COD refund, business days
- damaged wrong missing, high value damage, 48 hours, photo proof
- warranty, brand warranty, Assured badge, claiming warranty
- festive sale return window, 30 days
- exchange size color, grocery non-returnable, innerwear non-returnable
- late express delivery credit, 3 delivery attempts
- human handover, escalation, ticket creation

Do NOT answer the question. Output ONLY the rewritten English search query.
"""


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words/tokens, supporting English, Hindi, Hinglish."""
    return re.findall(r"\w+", text.lower())


def extract_skus(text: str) -> List[str]:
    """Extract SKUs matching the catalog regex pattern [A-Z]{3}-\d{4} or product catalog title."""
    found = re.findall(r"\b[A-Z]{3}-\d{4}\b", text)
    try:
        from src.data import PRODUCTS_DF
        text_lower = text.lower()
        for _, prod in PRODUCTS_DF.iterrows():
            sku = prod["sku"]
            if sku in found:
                continue
            base_title = prod["title"].split(",")[0].strip().lower()
            if len(base_title) > 4 and base_title in text_lower:
                found.append(sku)
    except Exception:
        pass
    return found


class HybridRetriever:
    """Hybrid Retriever combining BM25 and Dense Vector embeddings (FAISS) with Precedence."""

    def __init__(
        self,
        chunks: Optional[List[Chunk]] = None,
        use_query_rewrite: bool = True,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.chunks: List[Chunk] = chunks if chunks is not None else load_and_chunk_kb()
        self.use_query_rewrite = use_query_rewrite
        self.embedding_model_name = embedding_model_name
        self.embedding_provider = "local-sentence-transformers"

        # Initialize BM25
        tokenized_corpus = [tokenize(c.text) for c in self.chunks]
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

        # Initialize Sentence Transformer & FAISS
        self.encoder = None
        self.faiss_index = None
        self.chunk_embeddings = None

        if SentenceTransformer is not None and faiss is not None:
            try:
                self.encoder = SentenceTransformer(self.embedding_model_name)
                raw_embeddings = self.encoder.encode(
                    [c.text for c in self.chunks],
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                # Normalize for cosine similarity via IndexFlatIP
                norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                self.chunk_embeddings = raw_embeddings / norms

                dim = self.chunk_embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.chunk_embeddings.astype(np.float32))
            except Exception as e:
                print(f"[Warning] Failed to initialize local FAISS embeddings: {e}")
                self.faiss_index = None

    def rewrite_query(self, query: str) -> str:
        """Use LLM to rewrite user query into English policy search terms."""
        try:
            from src.llm import complete
            rewrite = complete(system=QUERY_REWRITE_SYSTEM_PROMPT, user=query).strip()
            # Clean up potential quotes or markdown
            rewrite = rewrite.strip('"\n`')
            return rewrite if rewrite else query
        except Exception:
            return query

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        enable_rewrite: Optional[bool] = None,
        is_asr: bool = False,
    ) -> List[Tuple[Chunk, float]]:
        """Retrieve top_k chunks for the given query using hybrid search."""
        # 1. Normalise spelled-out digits inside SKU / Order ID tokens (Fix 5)
        # Frozen config: Fix 5 applies ONLY to ASR input; never mutate text queries
        if FIX5_ASR_NORMALISE and is_asr:
            query = normalise_asr_text(query)

        should_rewrite = self.use_query_rewrite if enable_rewrite is None else enable_rewrite
        rewritten_query = self.rewrite_query(query) if should_rewrite else ""

        # 2. Check for SKU mentions in query and collect category expansion terms (Fix 1)
        skus_in_query = extract_skus(query)
        sku_chunks: List[Chunk] = []
        category_terms: List[str] = []

        for sku in skus_in_query:
            prod = get_product(sku)
            if prod:
                sku_chunk_text = (
                    f"products.csv:{sku} | Title: {prod['title']} | Brand: {prod.get('brand', '')} | "
                    f"Category: {prod.get('category', '')} | Return Category: {prod.get('return_category', '')} | Price: ₹{prod.get('price_inr', '')} | "
                    f"Fulfilled by: {prod.get('fulfilled_by', '')} | Assured: {prod.get('assured', '')} | "
                    f"Warranty: {prod.get('warranty_months', '')} months | Description: {prod.get('description', '')}"
                )
                sku_chunk = Chunk(
                    section_id=f"products.csv:{sku}",
                    doc_id="products.csv",
                    doc_type="catalog",
                    title=prod["title"],
                    section_title="Product Catalog Details",
                    version="1.0",
                    effective_from="2026-10-20",
                    status="current",
                    supersedes="",
                    superseded_by="",
                    piece_no=0,
                    total_pieces=1,
                    text=sku_chunk_text,
                    file_name="data/products.csv",
                )
                sku_chunks.append(sku_chunk)

                # Fix 1: when a product matches products.csv, collect category and return_category terms
                if FIX1_CATEGORY_EXPANSION:
                    skip_expansion = False
                    if FIX1_SKU_GUARD:
                        has_explicit_sku = bool(re.search(r"\b[A-Z]{3}-\d{4}\b", query))
                        if has_explicit_sku:
                            skip_expansion = True
                    if not skip_expansion:
                        cat = str(prod.get("category", "")).strip()
                        ret_cat = str(prod.get("return_category", "")).strip()
                        if cat and cat.lower() not in [t.lower() for t in category_terms]:
                            category_terms.append(cat)
                        if ret_cat and ret_cat.lower() not in [t.lower() for t in category_terms]:
                            category_terms.append(ret_cat)

        # Compute scores for chunks
        n_chunks = len(self.chunks)
        rrf_scores = np.zeros(n_chunks, dtype=float)
        k_rrf = 60.0

        # Fix 1: add category and return_category terms to the search query
        search_query = query
        if FIX1_CATEGORY_EXPANSION and category_terms:
            search_query = f"{query} {' '.join(category_terms)}"

        queries_to_run = [search_query]
        if rewritten_query and rewritten_query.strip() and rewritten_query.lower() != query.lower():
            if FIX1_CATEGORY_EXPANSION and category_terms:
                queries_to_run.append(f"{rewritten_query} {' '.join(category_terms)}")
            else:
                queries_to_run.append(rewritten_query)

        for q in queries_to_run:
            # 1. BM25 search
            if self.bm25 is not None:
                bm25_scores = np.array(self.bm25.get_scores(tokenize(q)))
                bm25_ranked = np.argsort(-bm25_scores)
                for rank, idx in enumerate(bm25_ranked):
                    rrf_scores[idx] += 1.0 / (k_rrf + rank + 1)

            # 2. Dense Vector search
            if self.faiss_index is not None and self.encoder is not None:
                q_emb = self.encoder.encode([q], convert_to_numpy=True)
                q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
                q_norm[q_norm == 0] = 1e-10
                q_emb = (q_emb / q_norm).astype(np.float32)

                sims, indices = self.faiss_index.search(q_emb, n_chunks)
                for rank, idx in enumerate(indices[0]):
                    if idx >= 0:
                        rrf_scores[idx] += 1.0 / (k_rrf + rank + 1)

        # Precedence and Superseded handling
        # Query check for superseded exception: orders/electronics delivered before 1 August 2026
        q_lower = query.lower()
        is_older_order = bool(
            re.search(r"before (?:1 |01 )?august|prior to august|\b(?:july|june|may)\b|version 3|v3", q_lower)
        )

        final_scores: List[float] = []
        for i, chunk in enumerate(self.chunks):
            score = rrf_scores[i]

            # Precedence tier adjustment
            # Policy (1) > Agent Guide (2) > Help (3) > FAQ (4) > Marketing (5)
            tier = PRECEDENCE_TIERS.get(chunk.doc_type, 6)
            if chunk.status == "superseded":
                tier = 10

            # Tier bonus to ensure policies and agent guides win over marketing/help when relevant
            tier_bonus = {
                1: 0.005,  # Policy
                2: 0.003,  # Agent guide
                3: 0.002,  # Help
                4: 0.001,  # FAQ
                5: 0.000,  # Marketing (never overrides policy)
                6: 0.000,
            }.get(tier, 0.0)
            score += tier_bonus

            # Superseded documents handling
            if chunk.status == "superseded":
                if not is_older_order:
                    # Penalize superseded chunks heavily so current chunks win
                    score *= 0.15
                else:
                    # Older electronics order case: superseded policy applies as specified in RET-3
                    score += 0.004

            final_scores.append(score)

        ranked_indices = list(np.argsort(-np.array(final_scores)))

        # Rule: Superseded document never wins over its successor (POL-RET) unless is_older_order
        if not is_older_order:
            # Reorder if any superseded chunk is above successor
            ret_current_indices = [
                idx for idx in ranked_indices if self.chunks[idx].doc_id == "POL-RET"
            ]
            ret_v3_indices = [
                idx for idx in ranked_indices if self.chunks[idx].doc_id == "POL-RET-v3"
            ]
            if ret_current_indices and ret_v3_indices:
                first_current = min(ranked_indices.index(idx) for idx in ret_current_indices)
                for v3_idx in ret_v3_indices:
                    v3_pos = ranked_indices.index(v3_idx)
                    if v3_pos < first_current:
                        ranked_indices.remove(v3_idx)
                        ranked_indices.insert(first_current + 1, v3_idx)

        # Assemble retrieved chunks
        retrieved: List[Tuple[Chunk, float]] = []

        # SKU chunk is given highest priority if present in the question
        for sku_c in sku_chunks:
            retrieved.append((sku_c, 1.0))

        remaining_slots = max(0, top_k - len(retrieved))
        for idx in ranked_indices[:remaining_slots]:
            retrieved.append((self.chunks[idx], float(final_scores[idx])))

        # Ensure exactly top_k items if available
        return retrieved[:top_k]
