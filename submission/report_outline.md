# VoiceOps 2026 — Report Outline

**Format:** 4 pages maximum. **Summary budget:** 150 words. **Total target:** ~1,350 words.
**TBD rule:** TBD means the number is not final or no `outputs/` file records it. Every number below cites the file it came from. Quotes are verbatim.

## 1. Summary (150 words)

VoiceOps 2026 is a support agent for fictional Indian store ZentroMart. It answers policy questions from 21 documents [outputs/step2_notes.md], looks up verified orders, hands over, and accepts speech. It includes a prompt-only baseline (`src/baseline.py`), hybrid RAG (`src/rag.py`), a two-tool conversation layer (`src/conversation.py`, `src/order_tools.py`), and a voice pipeline (`src/voice.py`). On 52 practice questions the baseline scored 15/52 (28.85%) decision-correct [outputs/step1_vs_step2.md], and RAG text scored 44/52 (84.62%) after Phase 2D; the fix set is under review [outputs/step2_notes.md; outputs/reconcile/reconciliation_report.md]. Phase 2B text: DEV verified by rerun (23/26 decision, 21/26 must-include); HOLDOUT and ALL carried from the Phase 2B report and not yet re-verified [outputs/reconcile/reconciliation_report.md sections 1-2]. Gain is uneven: recall@5 is 55.67% [outputs/step1_vs_step2.md], PQ0030 still fails [outputs/step2_scores.json], the ASR decision rate fell from 49/52 (94.23%) to 43/52 (82.69%) [outputs/step2_notes.md], and all 6 voice replies are stubs [outputs/voice/summary.csv; outputs/voice/05_sku_warranty/reply.txt], so no voice answer is shown end to end. Step 3 has offline tests. The report states these gaps.

## 2. What was built, by step

**Step 1 — baseline.** Prompt-only answerer in `src/baseline.py`, run with `python run.py step1`. Results: `outputs/step1_results.csv`, `outputs/step1_scores.json`.

**Step 2 — RAG.** `src/kb.py` parses 21 documents into 84 sections and 84 chunks, with 0 duplicate section IDs [outputs/step2_notes.md]. `src/retrieval.py` fuses BM25 and FAISS dense rankings with RRF; `src/rag.py` synthesises grounded answers and pre-computes business-day dates. Evaluation: `eval/run_step2.py`, `eval/score_step2.py`, `eval/run_audit.py`. Results: `outputs/step2_results.csv`, `outputs/step2_scores.json`.

**Step 3 — conversation with two tools.** `src/conversation.py` holds state and verification flow. `src/order_tools.py` wraps the two tools: order lookup and ticket creation. Evaluation: `eval/run_step3.py`, `eval/score_step3.py`. No `outputs/step3_results.csv` exists yet.

**Step 4 — voice.** `src/voice.py` wraps Gnani STT and TTS; `src/agent_protocol.py` defines the agent interface; `src/agent_stub.py` is the placeholder used in the recorded runs. Batch runner: `eval/run_voice.py`, invoked as `python run.py voice <folder>`. Each clip writes `outputs/voice/<clip>/` with `transcript.txt`, `reply.txt`, `reply.wav`, `result.json`.

## 3. Step 1 vs Step 2

| Metric | Step 1 | Step 2 | Source |
| :--- | :--- | :--- | :--- |
| Decision correct | 15/52 (28.85%) | 44/52 (84.62%) | [outputs/step1_vs_step2.md] |
| Must-include pass | 8/52 (15.38%) | 42/52 (80.77%) | [outputs/step1_vs_step2.md] |
| Grounded answers | TBD (no Step 1 audit) | Claim misses PQ0019, PQ0024, PQ0036; probably the Phase 2B text run (the audit was written before the 22:30:38 overwrite of step2_results.csv); not verified. | [outputs/step2_grounding_audit.csv; eval/run_audit.py:224-228] |

Step 2 holdout is 23/26 (88.46%) on both decision and must-include, with 0 must-not-include violations [outputs/step1_vs_step2.md]. Answerable recall@5 is 55.67% and recall@10 is 67.02%; the hard-negative hit rate@5 is 0.0% (n=5) [outputs/step1_vs_step2.md]. The ASR-text path scores 43/52 (82.69%) decision, 40/52 (76.92%) must-include and 51.06% recall@5 [outputs/step1_vs_step2.md]. That ASR result is a regression from Phase 2B, which scored 49/52 (94.23%) decision and 45/52 (86.54%) must-include [outputs/step2_notes.md].

Run naming for the "grounded answers" row: `outputs/step2_grounding_audit.csv` was last written 2026-10-25 22:06:53, before `outputs/step2_results.csv` was overwritten by the Phase 2D text run at 22:30:38 [outputs/reconcile/reconciliation_report.md section 1 (Identification of Baseline Files)]. The exact producing run is therefore **UNVERIFIED**; the audit cannot be attributed to the Phase 2D text run. `eval/run_audit.py:224-228` reads `outputs/step2_results.csv` and audits only rows whose decision is `answer`; the current file is the Phase 2D text run. Caveat: the audit checks whether claims appear in `used_sections`, not whether the answer is correct. It auto-passes a "cannot be found" claim when `used_sections` is empty [eval/run_audit.py:122-125], so it cannot catch a false refusal. PQ0030 is exactly that case (Section 6).

## 4. Step 3 results

Offline validation now, real run later.

- "All 5 offline unit tests executed with zero LLM calls." [outputs/step3_notes.md]
- Offline: decision and tool-call checks passed on 12 turns. Two turns used a stub adapter, so their must-include results are not scored. PC019 turn 4's stub reply lacks the required refusal wording and '7', so it is not a pass. Adapter labeling for all 12 turns is in `outputs/step3_notes.md` section 5; Phase 3C does not validate RAG grounding. [tests/test_phase3c_offline.py:183-241]
- Section-ID validation caches 154 valid IDs (`len(get_valid_section_ids()) = 154`; offline run: 84 KB + 70 product IDs) [src/conversation.py:32-42]; in the 12-turn offline smoke test, 0 order details leaked before verification [tests/test_phase3c_offline.py:354,358]; policy-text literals: TBD until the AST check report exists; false ticket confirmations: TBD. The handover path is untested [outputs/step3_notes.md:40,74-79].
- Real batch run: **TBD**. The "runner is kept ready and not triggered until requested" [outputs/step3_notes.md]. `outputs/step3_results.csv` and `outputs/step3_scores.json` do not exist, so there is no turn-level or conversation-level score to report.
- Handover record from a real run: **TBD**.

## 5. Voice results

**STT.** "6/6 clips matched exactly (ALL MATCH)." [outputs/voice/rerun_t2.txt]. STT latency is 0.765 s to 1.031 s [outputs/voice/rerun_t2.txt].

**Latency.** Per clip, total time is 2.291 s to 2.98 s and reply (TTS) time is 1.256 s to 1.976 s [outputs/voice/summary.csv]. A separate TTS check completed in 1.85 s and wrote a 315 KB WAV [outputs/voice/notes.md].

**What works.** All 6 clips are transcribed, replies are synthesised, per-clip artifacts and timings are written, and the 60-second clip limit is enforced [src/voice.py:302-305].

**What does not.** Every reply is stub output: "[STUB] Echo acknowledged" [outputs/voice/05_sku_warranty/reply.txt]. `result.json` cites `STUB-PLUMBING-01` and no retrieved chunks are stored, so the voice path never ran RAG or the conversation layer. The summary's `answer` decisions are stub decisions, not policy answers [outputs/voice/summary.csv]. Consequently no voice answer can be demonstrated end to end.

**Hinglish.** The note is marked "PROVISIONAL: Hinglish speech-to-text uses" and says the "decision rests on 1 clip" [outputs/voice/notes.md]. That clip is `04_hinglish_return.aac` [outputs/voice/notes.md].

## 6. What went wrong

**Step 2 text regression — reported, not fixed.** Phase 2B text: DEV verified by rerun (23/26 decision, 21/26 must-include); HOLDOUT and ALL carried from the Phase 2B report and not yet re-verified. After Phase 2D, text is 44/52 (DEV 21/26; HOLDOUT 23/26) and 42/52 (DEV 19/26; HOLDOUT 23/26). The 2E "no change" artifact is confirmed: the comparison used an overwritten baseline (reconciliation_report.md section 10, Phrasing Claim Correction). The surviving reconciliation validates the DEV before/after figures and records that `outputs/step2_results.csv` was overwritten by the Phase 2D run; the pre-fix text CSV was not preserved. The ALL/HOLDOUT before values are carried from the Phase 2D before/after table; the Phase 2F reconciliation verifies DEV only (reconciliation_report.md section 2, One True Baseline) [outputs/reconcile/reconciliation_report.md sections 1 (Identification of Baseline Files), 2 (One True Baseline), 10 (Phrasing Claim Correction); outputs/step2_notes.md:70-81].

**Fix 5 also ran on written text — likely contributor, cause not isolated.** In Phase 2D, Fix 5 was applied to written questions as well as ASR; the greedy regex corrupted SKU tokens, leaving 9/11 DEV SKUs retrieved and text recall@5 at 59.42% [outputs/reconcile/reconciliation_report.md section 4]. This likely caused part of the text SKU loss (9/11) and the text recall gap. The text regression's cause is not isolated.

**Step 2 ASR regression — measured, not fixed.** `outputs/step2_notes.md` discloses the Phase 2D drop: ASR ALL decision falls from 49/52 (94.23%) to 43/52 (82.69%) and must-include from 45/52 (86.54%) to 40/52 (76.92%). ASR DEV decision falls from 23/26 (88.46%) to 20/26 (76.92%) and must-include from 20/26 (76.92%) to 17/26 (65.38%). ASR HOLDOUT recall@10 falls from 58.33% to 56.25%. "Not in documents" rows rise from 4 to 5 across ASR, and from 1 to 3 in HOLDOUT [outputs/step2_notes.md]. No fix is in.

**Isolated Fix 1 — net-neutral, with two row-level failures.** Run B (Fix 1 only) holds decisions at 23/26 and must-include at 21/26, so it is net-neutral on decisions [outputs/reconcile/reconciliation_report.md section 3]. Against gold, PQ0057 expects `answer` but gets `clarify`, and PQ0046's damage claim is answered with the return window rather than escalation [outputs/reconcile/reconciliation_report.md section 8 (PQ0046 and PQ0057 Against the Gold)].

**PQ0030 "cannot find" miss.** The question text names a Kestrel Book 14 Laptop (SKU ELE-0023) and says it has a crack [practice/questions.csv, PQ0030]. The acceptable decision is `escalate`; the actual decision is `answer`, `decision_correct` is false, and the missing fact is "ticket / colleague / team / Priority Resolution" [outputs/step2_scores.json]. Gold evidence is "DWM-3 ; ESC-1" [outputs/step2_recall_misses.csv], but neither appears in the retrieved top 10 [outputs/step2_scores.json]. On the ASR-text run the reply says "I cannot find that in our documents." [outputs/step2_asr_text_dev_results.csv]. The grounding audit nevertheless marks PQ0030 "yes" against "none (unanswerable fallback)" [outputs/step2_grounding_audit.csv], because the audit treats a cannot-find phrase as self-justifying when no section is used [eval/run_audit.py]. The audit therefore hides this failure class instead of surfacing it.

**Spoken digits and dates before normalisation.** STT returns "six nine five two zero eight six seven" [outputs/voice/01_pincode_verify/transcript.txt] and "seventeen october twenty six" [outputs/voice/06_credit_refund/transcript.txt]. The normaliser recovers "8 candidate tokens checked, 7 SKU/Order tokens recovered, 0 false matches." [outputs/step2_notes.md]. It was not exercised end to end because the voice agent is a stub.

**Hinglish language choice on one clip.** `en-IN` was chosen over `hi-IN` from a single recording, and the note says the "decision rests on 1 clip" [outputs/voice/notes.md]. That is not enough evidence for a language policy.

**PQ0060 — retrieval unchanged; cause not isolated.** Text and ASR retrieval are identical between All OFF and All ON [outputs/reconcile/reconciliation_report.md section 9 (PQ0060 Retrieval Difference Check)]. No Fix-4-only run completed (Run C stopped at row 6), so the cause of the answer-to-fallback flip is not isolated [outputs/reconcile/reconciliation_report.md section 3].

**2E ablation — partial; no effect claimed.** The runner exists at `eval/run_ablation.py`, and `outputs/ablation/` holds partial run files. There is no finished comparison or summary in `outputs/`, so no 2E generation effect is claimed. The "no change" artifact is confirmed: it came from comparing against an overwritten baseline, not a valid ablation (reconciliation_report.md section 10, Phrasing Claim Correction). The ablation still has to complete and be written up.

## 7. What would be fixed next

1. Freeze the Step 2 configuration (Fix 5 for ASR only; Fixes 1, 4, 6 off), pending the owner's decision.
2. Real Step 3 run, then the full batch.
3. Voice run through the real agent.
4. Packaging fixes (Phase 4-PACK), then re-run the checker.

## 8. Bonus work

Only items that exist in the repo:

- Hybrid BM25 + FAISS retrieval with RRF fusion (`src/retrieval.py`).
- Multilingual dense embeddings with `paraphrase-multilingual-MiniLM-L12-v2` (`src/retrieval.py`).
- ASR normaliser (`src/asr_normalise.py`) with "8 candidate tokens checked, 7 SKU/Order tokens recovered, 0 false matches." [outputs/step2_notes.md].
- Deterministic business-day and refund-date pre-computation (`src/kb.py`, `src/rag.py`).
- Superseded-policy handling for `pol_returns_v3_superseded.md` (sections `RETv3-2`, `RETv3-6`), superseded by `pol_returns_v4.md` (section `RET-3`) (`src/retrieval.py:261-289`, `outputs/step2_notes.md:20-23`).
- Section-ID validation with 154 IDs and pruning of unknown IDs [outputs/step3_notes.md].
- Offline test suites: `tests/test_step3_offline.py`, `tests/test_phase3c_offline.py`.
- Grounding audit tool (`eval/run_audit.py`), with the blind spot noted in Section 6.
- Hinglish TTS routed through `language="auto"` (`README.md`).
- Per-clip voice timings (`outputs/voice/summary.csv`).

Not claimed: a working voice agent, a completed Step 3 batch run, or a 2E ablation result.
