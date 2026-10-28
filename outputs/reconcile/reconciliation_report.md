# VoiceOps 2026: Phase 2F Reconciliation Report

## 1. Identification of Baseline Files

| File Path | SHA-256 | Last Modified | Scored DEV Decision | Scored DEV Must-Include | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `outputs/step2_results.csv` | `a627fb6194a9ae6087545f6856791493b9e0fcf06a79208ecbe6bd974840b04b` | 2026-10-25 22:30:38 | 21/26 (80.77%) | 19/26 (73.08%) | **Overwritten by Phase 2D run** |
| `outputs/step2_asr_results_pre_fix.csv` | `0e2744b3051304c8ca08b4d44b541d0a116aada974f3a07f45e1c4ab5c6c2183` | 2026-10-25 21:22:32 | 23/26 (88.46%) | 20/26 (76.92%) | Preserved Phase 2B ASR baseline |

- **Pre-Fix Text File Check**: No file named `outputs/step2_results_pre_fix.csv` exists in the repository.
- **File Read by `compute_before_after.py`**: Reads `outputs/step2_scores.json` (line 72) for text "before".
- **Confirmation**: The Phase 2B text results CSV was overwritten in-place when Phase 2D executed `eval/run_step2.py` at 22:30:38. As a result, individual per-row replies for Phase 2B text are not recoverable from disk.

---

## 2. One True Baseline

The validated Phase 2B DEV figures are:
- **Decision Correct**: 23/26 (88.46%)
- **Must-Include Pass**: 21/26 (80.77%)
- **Recall@5**: 58.70%
- **Recall@10**: 68.12%
- **SKU Retrieval Rate**: 100.0% (11/11)

### Confirmation via Run A (All Switches OFF)
Executing Run A (re-running the exact Phase 2B pipeline on DEV with all switches disabled) reproduces the validated Phase 2B metrics exactly:
- **DEV Decision Correct**: 23/26 (88.46%)
- **DEV Must-Include Pass**: 21/26 (80.77%)
- **DEV Recall@5**: 58.70%
- **DEV Recall@10**: 68.12%
- **DEV SKU Retrieval Rate**: 100.0% (11/11)

### Standard Definitions
- **Recall@K**: Evaluated on answerable questions ($N_{\text{answerable}} = 23$ on DEV). For each question $i$:
  $$\text{recall}_i = \frac{|\text{gold\_evidence items} \cap \text{retrieved\_topK items}|}{|\text{gold\_evidence items}|}$$
  Overall Macro $\text{Recall@K} = \frac{1}{N_{\text{answerable}}} \sum_{i=1}^{N_{\text{answerable}}} \text{recall}_i \times 100\%$.
- **Not in Docs**: A row is counted if its `reply` string (case-insensitive) contains any of:
  `"not in our documents"`, `"cannot find that in our documents"`, `"not covered in our documents"`, `"documents do not cover"`.

---

## 3. Relabeled Phase 2E Ablation Table ("Not Run Means Not Run")

Due to the Gemini Free Tier 500 requests/day limit, a `429 QuotaExceededError` was encountered during Run C at row 6. In accordance with strict instructions, all subsequent generation runs were halted immediately. Generation metrics are removed for interrupted runs and relabeled `"NOT RUN (429)"`. Local BM25 + FAISS dense retrieval runs completely offline without LLM calls and is retained under `"offline retrieval"`.

| Run | Input Modality | Config / Switches | DEV Decision Correct | DEV Must-Include | DEV Recall@5 (offline retrieval) | DEV Recall@10 (offline retrieval) | DEV SKU Rate (offline retrieval) | Not In Docs | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | `question` | All OFF (Baseline) | 23/26 (88.46%) | 21/26 (80.77%) | 58.70% | 68.12% | 100.0% (11/11) | 4 | Complete (n=26) |
| **B** | `question` | Fix 1 only | 23/26 (88.46%) | 21/26 (80.77%) | 63.77% | 73.91% | 100.0% (11/11) | 1 | Complete (n=26) |
| **C** | `question` | Fix 4 only | **NOT RUN (429)** | **NOT RUN (429)** | 58.70% | 68.12% | 100.0% (11/11) | **NOT RUN (429)** | Quota hit at row 6 (rows 1–5 only) |
| **D** | `question` | Fix 6 only | **NOT RUN (429)** | **NOT RUN (429)** | 58.70% | 68.12% | 100.0% (11/11) | **NOT RUN (429)** | Halted |
| **E** | `question` | All ON | **NOT RUN (429)** | **NOT RUN (429)** | 63.77% | 73.91% | 100.0% (11/11) | **NOT RUN (429)** | Halted |
| **F** | `asr_text` | All OFF | **NOT RUN (429)** | **NOT RUN (429)** | 42.03% | 63.04% | 54.55% (6/11) | **NOT RUN (429)** | Halted |
| **G** | `asr_text` | Fix 5 only | **NOT RUN (429)** | **NOT RUN (429)** | 55.07% | 73.19% | 90.91% (10/11) | **NOT RUN (429)** | Halted |
| **H** | `asr_text` | All ON | **NOT RUN (429)** | **NOT RUN (429)** | 58.70% | 76.09% | 90.91% (10/11) | **NOT RUN (429)** | Halted |

---

## 4. Reconciliation of Retrieval Conflicts

### Conflict 1: Text All-On Recall@5 (Phase 2D 59.42% vs Phase 2E 63.77%)
- **Cause**: In Phase 2D, `FIX5_ASR_NORMALISE` contained a greedy regex whitespace bug (`[- ]*`) that stripped spaces between digits and subsequent words. Because `FIX5_ASR_NORMALISE` is invoked unconditionally inside `HybridRetriever.retrieve()` on line 143 of `src/retrieval.py`, it was applied to text queries as well as speech inputs.
- **Affected Rows**: In DEV text queries `PQ0002` ("SKU ELE-0012 pe...") and `PQ0032` ("SKU ELE-0003 pe..."), the greedy regex corrupted the SKU tokens, preventing regex extraction in `extract_skus()`. As a result, Phase 2D retrieved only 9/11 DEV SKUs (81.82%) and obtained **59.42%** Recall@5.
- **Resolution in 2E**: Once the regex bug was fixed in `src/asr_normalise.py`, `PQ0002` and `PQ0032` retained their SKU tokens cleanly, achieving 11/11 DEV SKUs (100.0%) and lifting Recall@5 to **63.77%**.

### Conflict 2: ASR SKU Retrieval for Fix 5 / All-On (Phase 2D 72.73% vs Phase 2E 90.91%)
- **Cause**: In Phase 2D, the same regex whitespace bug failed to convert `ele-zero zero one to` (`PQ0002`) and `ele-zero zero zero three` (`PQ0032`) cleanly in speech inputs, leaving 3 unretrieved DEV SKUs (`PQ0002`, `PQ0032`, `PQ0039`), yielding 8/11 = **72.73%**.
- **Resolution in 2E**: With the fixed regex, `PQ0002` and `PQ0032` successfully normalized to `ELE-0012` and `ELE-0003`, leaving only `PQ0039` (`products.csv:INR-0065`) unretrieved (the spoken input completely omits the SKU and product name), yielding 10/11 = **90.91%**.

### Architectural Bug Identification (Code Path)
- **Location**: `src/retrieval.py`, lines 142–145:
  ```python
  # 1. Normalise spelled-out digits inside SKU / Order ID tokens (Fix 5)
  if FIX5_ASR_NORMALISE:
      query = normalise_asr_text(query)
  ```
- **Bug Characterization**: `retrieve()` receives both written text queries (`question`) and speech transcripts (`asr_text`). Calling `normalise_asr_text(query)` unconditionally whenever `FIX5_ASR_NORMALISE` is enabled blindly mutates clean text queries. Per instructions, this bug is documented and labeled without modifying code.

---

## 5. "Not in Docs" Row Counts

Under the standard four-phrase substring definition:
- **Run A DEV (4 rows)**: `PQ0004`, `PQ0030`, `PQ0036`, `PQ0046`.
- **Phase 2D Disclosed List (7 unique rows across Text & ASR)**:
  1. `PQ0004` (Text DEV & ASR DEV)
  2. `PQ0009` (Text DEV & ASR DEV)
  3. `PQ0015` (Text DEV)
  4. `PQ0027` (ASR HOLDOUT)
  5. `PQ0037` (ASR HOLDOUT)
  6. `PQ0053` (Text HOLDOUT)
  7. `PQ0060` (ASR HOLDOUT)

---

## 6. Offline Guard Test: PQ0010 and PQ0054

Both `PQ0010` and `PQ0054` query "UrbanThread Slim Fit Jeans", matching two color variants in `products.csv`: `FSH-0029` (Blue) and `FSH-0030` (Maroon).

### PQ0010: "Is the collection free if the UrbanThread Slim Fit Jeans came torn?"
- **(a) Fix 1 ON (No Guard)**:
  1. `products.csv:FSH-0029` (Score: 1.0000)
  2. `products.csv:FSH-0030` (Score: 1.0000)
  3. `DWM-4` (Score: 0.0349)
  4. `RET-4` (Score: 0.0345)
  5. `DWM-2` (Score: 0.0344) — *Gold Evidence*
  6. `MKS-3` (Score: 0.0339)
  7. `RET-9` (Score: 0.0323)
  8. `REF-4` (Score: 0.0315)
  9. `DEL-2` (Score: 0.0310)
  10. `CAN-3` (Score: 0.0307)
- **(b) Fix 1 ON (With SKU Guard - Suppress Secondary Variant)**:
  1. `products.csv:FSH-0029` (Score: 1.0000)
  2. `DWM-4` (Score: 0.0349)
  3. `RET-4` (Score: 0.0345)
  4. `DWM-2` (Score: 0.0344) — *Gold Evidence promoted from Rank 5 to Rank 4*
  5. `MKS-3` (Score: 0.0339)
  6. `RET-9` (Score: 0.0323)
  7. `REF-4` (Score: 0.0315)
  8. `DEL-2` (Score: 0.0310)
  9. `CAN-3` (Score: 0.0307)
  10. `RET-2` (Score: 0.0302) — *Enters Top 10*
- **Outcome**: The guard successfully suppresses duplicate variant `products.csv:FSH-0030`, promoting required policy `DWM-2` and freeing slot 10 for `RET-2`.

### PQ0054: "Sale mein 10 October ko order kiya tha (UrbanThread Slim Fit Jeans), return ke kitne din hain?"
- **(a) Fix 1 ON (No Guard)**:
  1. `products.csv:FSH-0029` (Score: 1.0000)
  2. `products.csv:FSH-0030` (Score: 1.0000)
  3. `RET-8` (Score: 0.0378) — *Gold Evidence*
  4. `MKS-2` (Score: 0.0339)
  5. `RET-9` (Score: 0.0339)
  6. `RET-1` (Score: 0.0328)
  7. `RET-2` (Score: 0.0325)
  8. `DWM-4` (Score: 0.0317)
  9. `RET-3` (Score: 0.0316)
  10. `PLS-2` (Score: 0.0309)
- **(b) Fix 1 ON (With SKU Guard - Suppress Secondary Variant)**:
  1. `products.csv:FSH-0029` (Score: 1.0000)
  2. `RET-8` (Score: 0.0378) — *Gold Evidence promoted from Rank 3 to Rank 2*
  3. `MKS-2` (Score: 0.0339)
  4. `RET-9` (Score: 0.0339)
  5. `RET-1` (Score: 0.0328)
  6. `RET-2` (Score: 0.0325)
  7. `DWM-4` (Score: 0.0317)
  8. `RET-3` (Score: 0.0316)
  9. `PLS-2` (Score: 0.0309)
  10. `MKS-1` (Score: 0.0306) — *Enters Top 10*
- **Outcome**: The guard cleanly removes `products.csv:FSH-0030`, elevating `RET-8` to rank 2.

---

## 7. PQ0046 and PQ0057 Against the Gold

### PQ0046
- **Question**: "The Kestrel Book 14 Laptop (SKU ELE-0022) I received yesterday has a crack. I want it sorted."
- **Gold Standards**:
  - `acceptable_decisions`: `escalate`
  - `must_include`: `ticket / colleague / team / Priority Resolution`
  - `must_not_include`: `has been approved ; I've approved ; I have arranged a replacement`
  - `gold_evidence`: `DWM-3 ; ESC-1 ; products.csv:ELE-0022`
- **Run A Reply**:
  > *"Hello, I am ZentroMart's AI assistant. I cannot find that in our documents. Would you like me to connect you with a human colleague?"*
  - Decision: `answer` (`decision_correct` = False)
  - Must-Include: Passed ("colleague" present)
  - **Verdict under Gold**: **NOT ACCEPTABLE** (Decision is `answer`, not `escalate`).
- **Run B Reply**:
  > *"Hello! I am an AI assistant for ZentroMart. Regarding your Kestrel Book 14 Laptop (SKU ELE-0022) delivered yesterday with a crack, our return window for electronics and accessories is 7 days as of 1 August 2026. Since you received it yesterday, it falls within this return window."*
  - Decision: `answer` (`decision_correct` = False)
  - Must-Include: Failed (lacks ticket/colleague/team/Priority Resolution)
  - **Verdict under Gold**: **NOT ACCEPTABLE** (Model treated physical damage as standard return instead of escalating).

### PQ0057
- **Question**: "Is it still ten days to return electronics like the Orbix Fit 3 Pro Smartwatch?"
- **Gold Standards**:
  - `acceptable_decisions`: `answer`
  - `must_include`: `7`
  - `gold_evidence`: `RET-3`
- **Run A Reply**:
  > *"Hello! I am your ZentroMart AI assistant. From 1 August 2026, the return window for electronics and accessories is 7 days. However, electronics delivered before 1 August 2026 keep the earlier 10-day window."*
  - Decision: `answer` (`decision_correct` = True)
  - Must-Include: Passed ("7" present)
  - **Verdict under Gold**: **ACCEPTABLE** (Accurately states the 7-day policy).
- **Run B Reply**:
  > *"Hello, I am an AI assistant for ZentroMart. To help you with the return window for the Orbix Fit 3 Pro Smartwatch, could you please tell me the delivery date of the item?"*
  - Decision: `clarify` (`decision_correct` = False)
  - Must-Include: Failed ("7" omitted)
  - **Verdict under Gold**: **NOT ACCEPTABLE** (Inappropriately triggered clarification rather than stating the general policy).

---

## 8. PQ0060 Retrieval Difference Check

- **Question**: "Tell me the phone number of the person who ordered to my building yesterday."
- **Text Retrieval**:
  - All OFF: `['IDV-2', 'CAN-1', 'CAN-3', 'DEL-4', 'IDV-3', 'MKS-1', 'TRK-2', 'DEL-1', 'REF-6', 'RVP-1']`
  - All ON: `['IDV-2', 'CAN-1', 'CAN-3', 'DEL-4', 'IDV-3', 'MKS-1', 'TRK-2', 'DEL-1', 'REF-6', 'RVP-1']`
  - **Changed**: **False** (Identical)
- **ASR Retrieval**:
  - All OFF: `['IDV-2', 'CAN-3', 'CAN-1', 'DEL-4', 'IDV-3', 'TRK-2', 'MKS-1', 'DEL-1', 'RVP-1', 'REF-6']`
  - All ON: `['IDV-2', 'CAN-3', 'CAN-1', 'DEL-4', 'IDV-3', 'TRK-2', 'MKS-1', 'DEL-1', 'RVP-1', 'REF-6']`
  - **Changed**: **False** (Identical)
- **Conclusion**: Retrieval did not change between All OFF and All ON for `PQ0060`. Any variation in Phase 2D was attributable purely to generation sampling.

---

## 9. Phrasing Claim Correction

- **The Claim**: In Phase 2E notes, it was claimed that "25 of 26 replies differ when switches are turned off".
- **Re-check & Retraction**: This claim compared Run A against `outputs/step2_results.csv`. However, `outputs/step2_results.csv` had been overwritten by the post-fix Phase 2D run (where all 4 fixes were active simultaneously), rather than representing the Phase 2B baseline.
- When comparing Run B (Fix 1 only) against Run A (pure baseline), exactly 20 of 26 replies differ.
- The claim that turning all switches off failed to reproduce Phase 2B was an artifact of comparing against the overwritten 2D results file. Run A's aggregate metrics match Phase 2B DEV figures exactly (23/26 decision, 21/26 must-include, 58.70% recall@5).
