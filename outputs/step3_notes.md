# VoiceOps 2026: Step 3 Engineering Notes & Architectural Choices

## 1. Tool Wrappers (`src/order_tools.py`)
- **Dataset Integration**: `DATASET_DIR` is dynamically resolved from `src.data` and injected into `sys.path` at module load time. Zero modifications are made to the original dataset.
- **Unverified Lookup (`lookup_order_unverified`)**: Strips non-digits from the provided order ID and calls `tools.tools.lookup_order(oid)` with no verification parameter. Discloses only `verification_required=True` and a prompt message; zero customer PII, status, items, or totals are leaked.
- **Verified Lookup (`lookup_order_verified`)**: Strips non-digits from both order ID and customer-typed verification value, calling `tools.tools.lookup_order(oid, verification_value=value)`. Returns full order details only upon a verified match of registered phone last 4 digits or 6-digit delivery pincode.
- **Handover (`handover`)**: Calls `tools.tools.create_ticket`. Enforces minimum summary length of 20 characters (naming the customer issue, checks completed, and escalation reason). Reports ticket creation only when `created=True`.

## 2. Multi-Turn Conversational State & Logic (`src/conversation.py`)
- **State Model (`ConversationState`)**:
  - `order_id`: 8-digit order identifier.
  - `verified`: Boolean flag gating all order-specific disclosures.
  - `verification_attempts`: Integer counter bounded between 0 and 2.
  - `issue_summary`: Detailed log of customer inquiry and checks performed.
  - `handover_created`: Boolean flag indicating if a handover ticket was logged.
  - `turns`: Sliding window preserving strictly the last 6 conversation turns.
  - `order_data`: Full order payload stored in memory only upon successful verification.
  - `language`: Detected turn language ('en', 'hi', or 'hi-en').
- **RAG Answer Adapter (`rag_answer_adapter`)**:
  - Delegates every policy answer to `src.rag.answer_question` through one shared `HybridRetriever` per `ConversationAgent`; it forwards `top_k`, `enable_rewrite`, and the per-turn `is_asr` modality gate and returns only the grounded `decision`, `reply`, and `used_sections`.
- **Verification Flow**:
  1. *Turn 1 (Greeting)*: Agent introduces identity as ZentroMart's AI support assistant (`[STY-1]`), requests the customer's order number, making zero tool calls (`decision="clarify"`).
  2. *Turn 2 (Order ID Introduced)*: When an 8-digit order ID is first detected, calls `lookup_order_unverified` and asks for ONE verification value (last 4 digits of registered phone), never both at once, without revealing expected values (`decision="clarify"`).
  3. *Turn 3 (Verification Attempt)*:
     - **Match**: `lookup_order_verified` returns `verified=True`. Agent transitions state to `verified=True`, confirms verification, and prompts how it can assist (`decision="answer"`).
     - **Attempt 1 Mismatch**: Increments attempts to 1, allows one retry, asks the customer to try again, and never discloses expected values (`decision="clarify"`).
     - **Attempt 2 Mismatch**: Increments attempts to 2, refuses disclosure of all order details, suggests calling from registered number, never reveals secrets, and makes no handover ticket (`decision="refuse"`).
- **Human Handover & Escalation (`[ESC-1]`, `[ESC-2]`, `[ESC-3]`)**:
  - Escalates immediately when any policy condition is met:
    - Damaged, defective, or wrong item worth > ₹10,000.
    - Customer reports unauthorized account activity / fraud.
    - Customer mentions legal action, consumer court, or police complaint.
    - Repeat contact (`contacts_last_7_days >= 2` from order data).
    - Refund delayed more than 3 business days beyond maximum timeline.
    - Customer explicitly requests a human agent.
    - Product safety risk (burning, sparks, swelling battery, injury).
  - Summaries are guaranteed >= 20 characters and specify the issue, checks made, and policy reason.
  - Never promises outcomes or claims actions the code did not take (e.g. does not claim a refund was initiated or replacement approved).
- **Policy Refusals**:
  - Requesting compensation beyond delivery fee refund is refused per policy without ticket creation.
  - Cancelling an order that has already shipped is refused per policy (`[CAN-2]`), advising refusal at delivery or reverse pickup.
- **Deterministic Python Pre-Calculations**:
  - `src.rag.answer_question` calls `precompute_refund_calculations` for refund timelines, using the frozen reference date `2026-10-20` and the holiday calendar. The conversation adapter does not compute policy dates, fees, or amounts itself.

## 3. Evaluation & Scoring Modules (`eval/run_step3.py`, `eval/score_step3.py`)
- **Batch Runner (`eval/run_step3.py`)**:
  - Iterates over `practice/conversations.csv` grouped by `conversation_id` in chronological turn order.
  - Instantiates a fresh `ConversationAgent` per conversation.
  - Records turn decision, customer message, reply, tool calls (name and arguments), used sections, and serialized state after each turn into `outputs/step3_results.csv`.
  - Offline compliance: Built and verified offline; runner is kept ready and not triggered until requested.
- **Turn & Conversation Scorer (`eval/score_step3.py`)**:
  - Evaluates `decision_correct` (`actual == expected_decision`).
  - Evaluates `must_include` and `must_not_include` facts using `src.scoring.score_row`.
  - Evaluates `tool_calls_correct` by matching tool name and argument key-values against `expected_tool_calls` (empty string demands 0 tool calls).
  - Enforces all-or-nothing conversation scoring: a conversation passes only if 100% of its turns pass.
  - Aggregates totals by turn and conversation into `outputs/step3_scores.json`.

## 4. Offline Test Results Summary (T1 to T5)
- All 5 offline unit tests executed with zero LLM calls.
- Dynamic data extraction from `orders.csv` and `customers.csv` successfully verified unverified data withholding, failed verification rejection, dual-credential verification (phone suffix and pincode), summary length validation, and ticket creation logging.

## 5. Phase 3C Answer Path Optimization & Hardcoded Replies Elimination
- **Phase 3C adapter relabel (12 turns)**:

  | Conversation | Turn | Adapter configured | Adapter invoked | Turn path |
  |---|---|---|---|---|
  | PC021 | 1 | default | no | greeting fixed flow |
  | PC021 | 2 | default | no | order-ID fixed flow |
  | PC021 | 3 | default | no | failed-verification retry |
  | PC021 | 4 | default | no | failed-verification refusal |
  | PC015 | 1 | explicit stub | no | greeting fixed flow |
  | PC015 | 2 | explicit stub | no | order-ID fixed flow |
  | PC015 | 3 | explicit stub | no | successful verification |
  | PC015 | 4 | explicit stub | yes | compensation refusal |
  | PC019 | 1 | explicit stub | no | greeting fixed flow |
  | PC019 | 2 | explicit stub | no | order-ID fixed flow |
  | PC019 | 3 | explicit stub | no | successful verification |
  | PC019 | 4 | explicit stub | yes | cancel-shipped-order refusal |

  Phase 3C validates fixed-flow state/tool behavior and injected adapter plumbing. It does **not** validate RAG grounding: fixed-flow turns do not invoke an adapter, and the two policy turns use explicit stubs rather than `answer_question`.
- **Hardcoded Replies Audit (T6)**:
  - Audited all literal strings in `src/conversation.py`.
  - Found and classified all literals:
    - Fixed flow literals (greeting, verification value request, retry prompt, failed verification refusal, handover confirmation) are retained in Python flow.
    - Exactly 0 `POLICY TEXT` literals exist in `ConversationAgent` turn flow. All policy wording (compensation, late delivery, refunds, returns, cancellations) is routed dynamically through `rag_answer_adapter`.
- **Section ID Validation & Pruning (T7)**:
  - Nonexistent ID `POL-DEL` removed entirely from the codebase.
  - Implemented `get_valid_section_ids()` caching 154 valid IDs (84 KB document sections from `kb/*.md` + 70 SKU entries from `products.csv:<SKU>`).
  - Added `validate_used_sections()` which validates section IDs, drops unknown IDs (such as `POL-DEL` and `UNKNOWN-SEC-99`), logs warnings, and records dropped counts into `outputs/step3_results.csv`.
- **Scorer Type Neutrality (T8)**:
  - Updated `match_tool_calls` in `eval/score_step3.py` to convert tool-call arguments to strings on both sides (`str(call_args.get(k, '')).strip().strip('\"\'')` vs `str(v).strip().strip('\"\'')`), ensuring type-neutral comparisons (e.g. integer `77461439` vs string `"77461439"` and preserving leading zeros in verification values).
- **Adapter Injection Verification (T9)**:
  - Verified by dependency injection of stub adapters for PC015 Turn 4 (compensation refusal) and PC019 Turn 4 (refused delivery refund). The agent returns the stub adapter's exact reply without fallback or template additions.
- **Smoke Test Evaluation with `score_row`**:
  - Evaluated all 12 turns of PC021 (failed verification), PC015 (compensation demand), and PC019 (cancel shipped order).
  - All 12 turns achieved 100% pass rate on the fixed-flow/tool checks: `decision_correct` (12/12), `must_include_pass` (12/12), `must_not_include_violations` (0/12), `tool_calls_correct` (12/12). This result does not measure RAG grounding.
  - Confirmed 0 order details leaked prior to identity verification, and 0 false ticket confirmations.
