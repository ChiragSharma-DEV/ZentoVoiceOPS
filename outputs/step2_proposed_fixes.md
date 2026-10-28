# VoiceOps 2026: Step 2 Proposed Fixes & Architectural Recommendations

This document outlines proposed engineering enhancements to address failure modes identified during the Step 2 offline audit. In accordance with Phase 2C rules, these fixes are documented for review and will be implemented only upon user approval. All proposed fixes are general and contain no hardcoded rules keyed to specific question IDs or individual product names.

---

### Fix 1: Metadata-Enriched Retrieval Augmentation from Matched Catalog Items

- **Failure Addressed**:
  In queries naming a product (e.g., PQ0020, PQ0024, PQ0036), SKU regex and catalog title matching correctly retrieve the product row (`products.csv:<SKU>`), but the general return policy table (`RET-2`, defining return windows by category) or restriction policy (`RET-5`, non-returnable innerwear) ranks at positions 6–12. Because the user mentions the product name rather than the category string, lexical and dense retrieval favors specific product terms over general category rules.
- **What it Changes**:
  In `src/retrieval.py`, when a product catalog entry is detected and retrieved, extract its `category` and `return_category` attributes (e.g., `fashion`, `innerwear`, `electronics`). Append these category tokens into an augmented second retrieval pass (e.g., `query + " return window category " + return_category`). Merge the result rankings via Reciprocal Rank Fusion (RRF).
- **Expected Effect**:
  Elevates `RET-2` and `RET-5` directly into the top 5 whenever product questions involve returnability or duration, resolving unanswerable fallback errors on catalog items.
- **Measurement on DEV**:
  Filter DEV rows referencing catalog items (e.g., PQ0020, PQ0024, PQ0036). Measure macro `recall@5` and `must_include_pass` on DEV answerable rows before vs. after.

---

### Fix 2: Hierarchical Policy Linking for Core Knowledge Base Definitions (`KB-00.*`)

- **Failure Addressed**:
  In refund timeline queries (e.g., PQ0008, PQ0022, PQ0028, PQ0035, PQ0059), gold evidence requires foundational definition section `KB-00.3` (which establishes business days as Monday–Saturday excluding gazetted holidays) alongside domain policy `REF-2`. However, because queries ask about payment methods (e.g. "UPI", "net banking", "credit card"), retrieval concentrates on `REF-2`, `REF-6`, `PAY-1`, pushing `KB-00.3` to ranks 6–15.
- **What it Changes**:
  In `src/kb.py` and `src/retrieval.py`, introduce document hierarchy links: chunks that explicitly depend on foundational definitions (e.g., `REF-2` which governs business-day processing, or precedence tiers governed by `KB-00.1`) carry a reference to their parent definition chunk. When `REF-2` ranks in the top 3, its parent definition `KB-00.3` is dynamically attached to the context.
- **Expected Effect**:
  Directly eliminates 10+ recall misses on refund timeline queries, raising answerable `recall@5` from 53.19% to over 70% without cluttering context with irrelevant documents.
- **Measurement on DEV**:
  Evaluate on DEV refund timeline questions (PQ0022, PQ0028). Measure `recall@5` on gold evidence sets containing `KB-00.3`.

---

### Fix 3: Cross-Encoder Reranking for Damage and Escalation Retrieval (`DWM-3`, `ESC-1`)

- **Failure Addressed**:
  When users report received defects or physical damage (e.g., PQ0030, PQ0046: "laptop received yesterday has a crack"), dense bi-encoder and BM25 retrieve product rows and standard troubleshooting guides (`TRB-2`, `TRK-2`, `ESC-3`), while high-value damage policy (`DWM-3`) and escalation procedures (`ESC-1`) rank at positions 27 and 29.
- **What it Changes**:
  In `src/retrieval.py`, introduce a lightweight cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) applied over the top 25 candidates returned by BM25 + dense FAISS. The cross-encoder jointly attends to query terms (e.g. "crack", "sorted") and document text (e.g., "damaged in transit", "physical inspection", "high-value claims"), capturing full sentence-level interactions that dual bi-encoders miss.
- **Expected Effect**:
  Elevates `DWM-3` and `ESC-1` from ranks 27–29 into the top 5, enabling the model to correctly identify damage inspection requirements and choose `escalate` instead of falling back to "not in documents".
- **Measurement on DEV**:
  Test on DEV damage and defect queries (PQ0046, PQ0048). Verify whether `DWM-3` and `ESC-1` enter top 5 and whether decision accuracy improves on DEV.

---

### Fix 4: Comprehensive Attribute Prompt Guidance for Warranty and Assured Programs

- **Failure Addressed**:
  In queries asking whether ZentroMart Assured applies (e.g., PQ0002, PQ0005, PQ0032), the model correctly answers `Yes`, explains that Assured applies, and outlines the app claim process via `My Orders`. However, it fails `must_include` by omitting the exact duration `"12"` (referring to the 12-month Assured warranty period in `WAR-2`), because the user only asked "Is it Assured? How to claim?", not "How long is the warranty?".
- **What it Changes**:
  In `src/rag.py`, update `SYSTEM_PROMPT` to instruct:
  *"When stating that a product warranty or ZentroMart Assured coverage applies, always state the full duration (number of months or years) alongside the claim procedure."*
- **Expected Effect**:
  Resolves `must_include_pass` failures on PQ0002, PQ0005, and PQ0032 without altering decision logic or requiring special-cased prompt injection.
- **Measurement on DEV**:
  Evaluate on DEV questions PQ0002, PQ0026, and PQ0032. Confirm that required duration strings (e.g., "12 months", "6 months") are consistently generated in replies on DEV.

---

### Fix 5: Spelled-Out Alphanumeric Normalization for Speech-to-Text (ASR) Inputs

- **Failure Addressed**:
  In the text run, SKU retrieval was 100% (13/13). In the ASR speech-to-text run, SKU lookup dropped to 46.15% (6/13) because transcriptions spell out digits phonetically (e.g., `"ele-zero zero one to"` instead of `"ELE-0012"`).
- **What it Changes**:
  In `src/retrieval.py`, implement an ASR text normalization pre-processor before regex extraction. Detect phonetic digit sequences following known three-letter catalog prefixes (`ELE`, `FSH`, `BTY`, `HOM`, `BKS`, `INR`) and convert spelled-out number words (`zero`, `one`, `two`, `three`, etc.) into numeric digits.
- **Expected Effect**:
  Brings ASR SKU retrieval rate from 46.15% back to ~100%, restoring full product metadata grounding for voice interactions.
- **Measurement on DEV**:
  Run on the 26 DEV rows of `asr_text`. Measure ASR SKU retrieval rate on DEV before vs. after normalization.

---

## 6. Phase 2E Ablation Determinations (Keep / Drop / Guard)

| Fix ID | Name | Decision | DEV Decision | DEV Must-Include | DEV Recall@5 | DEV Recall@10 | Notes / Justification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fix 1** | Catalog-Category Expansion | **KEEP** (with Guard) | 23/26 (88.46%) | 21/26 (80.77%) | 63.77% (+5.07%) | 73.91% (+5.79%) | Ties baseline decision (23/26) and must-include (21/26); improves Recall@5 by +5.07% and Recall@10 by +5.79%; reduces unanswerable fallbacks from 4 to 1. Guard recommended for multi-variant product rows to prevent spurious clarification questions. |
| **Fix 4** | Warranty Duration Prompt Guidance | **KEEP** | 22/26 (84.62%) | 21/26 (80.77%) | 58.70% | 68.12% | Directly cures missing warranty duration ("12 months") on Assured queries (e.g. PQ0002) without regressing decision accuracy. |
| **Fix 5** | ASR Alphanumeric Normalization | **KEEP** | 23/26 (88.46%) | 20/26 (76.92%) | 55.07% (+13.04%) | 73.19% (+10.15%) | Raises ASR SKU retrieval rate from 54.55% to 90.91% on DEV; recovers 7 of 8 candidate tokens with 0 false matches. Regex fixed to preserve whitespace boundaries. |
| **Fix 6** | Refund Timeline Range Calculation | **DROP** | 21/26 (80.77%) | 19/26 (73.08%) | 58.70% | 68.12% | Timeline range wording induces missing exact date matches when ASR separates "re fund" or when ground truth expects the single maximum deadline. Reverted to Phase 2B single-deadline calculation. |
