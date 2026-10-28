# VoiceOps 2026

VoiceOps 2026 is an AI customer-support agent for ZentroMart, a fictional Indian online store. It answers policy questions from the supplied knowledge base, looks up verified orders, escalates when a human is needed, and accepts recorded speech. All LLM output is grounded in retrieved document sections; the agent never states a date, amount, or policy that is not in the retrieved context.

## Repository Layout

```text
voiceops/
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-step2.txt
├── requirements-step3.txt
├── requirements-voice.txt
├── run.py
├── src/                (data, retrieval, RAG, conversation, tools, voice)
├── eval/               (step runners, scorers, voice runner)
├── tests/              (offline test suite)
├── submission/         (offline checker and demo notes)
├── sample_outputs/     (frozen examples copied from the recorded runs)
└── outputs/            (generated at runtime; not shipped)
```

## Setup

### 1. Environment Setup

Create and activate a Python 3.11+ virtual environment:

```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

Install the requirements in this order:

```bash
pip install -r requirements.txt
pip install -r requirements-step2.txt
pip install -r requirements-voice.txt
pip install -r requirements-step3.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in the values locally. `.env` is gitignored and must not be shipped.

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `GNANI_API_KEY` | Gnani STT and TTS token. |
| `LLM_PROVIDER` | `gemini`. |
| `LLM_MODEL` | `gemini-3.5-flash-lite`. |
| `GEMINI_API_KEY` | Key for the Gemini REST endpoint used by `src/llm.py`. |
| `DATASET_DIR` | Absolute or relative path to the extracted `Capstone_Project_Dataset_FTTT_2026` directory. The dataset stays outside the repository and its files are never modified. |

`.env.example` lists every variable with an empty value, including `DATASET_DIR=`. The code no longer falls back to a hard-coded sibling directory: if `DATASET_DIR` is unset or missing, `src/data.py` and `src/kb.py` stop with a clear error.

## Execution Commands

Run every command from the repository root, with the virtual environment active. Live steps call Gemini and Gnani and therefore need keys in `.env`; the test suite is fully offline.

### Step 1 - prompt-only baseline

```bash
python run.py step1
python eval/score_step1.py
```

### Step 2 - grounded RAG

```bash
python eval/run_step2.py question HOLDOUT outputs/step2_holdout_text.csv
python eval/run_step2.py asr_text HOLDOUT outputs/step2_holdout_asr.csv
python eval/score_step2.py
```

`run_step2.py` takes an input column (`question` or `asr_text`), a split (`DEV`, `HOLDOUT`, or `ALL`), and an output CSV. The frozen outputs used in this submission are in `outputs/` and copied to `sample_outputs/`.

### Step 3 - multi-turn conversation and tools

```bash
python eval/run_step3.py
python eval/score_step3.py
```

The runner replays `practice/conversations.csv` turn by turn and writes `outputs/step3_results.csv` plus per-turn records. The v2 run reported here used the same code path and wrote `outputs/step3_batch_v2/`.

### Voice pipeline

```bash
python run.py voice <folder>
python eval/run_voice.py
```

`run.py voice <folder>` is the wired CLI branch (it currently uses the stub agent and is plumbing only). The recorded six-clip evaluation was run with `python eval/run_voice.py`, which reads the clip folder, calls Gnani STT/TTS, runs the conversation agent, and writes transcripts plus reply audio.

### Offline tests and submission check

```bash
python -m pytest tests
python submission/check_submission.py
```

The checker makes no network calls and runs no git commands. It reports `MISSING` for `report.pdf` and `demo.mp4` until those submission assets are added.

## Results

All numbers below come from the frozen files in `outputs/` and are copied into `sample_outputs/`.

| Run | Scored | Pass | Fail | Notes |
|---|---|---|---|---|
| Step 2 frozen, text HOLDOUT | 26 questions | 26/26 decision, 24/26 must-include | 0 decision | Grounded RAG, answer/refuse mix. |
| Step 2 frozen, ASR HOLDOUT | 26 questions | 26/26 decision, 25/26 must-include | 0 decision | Same prompts, ASR transcript input. |
| Step 3 v1 | 96 turns | 79 | 17 | decision 6, must_include 16, must_not 0, tool 2. |
| Step 3 rep1 | 96 turns | 79 | 17 | decision 6, must_include 16, must_not 0, tool 2. |
| Step 3 rep2 | 96 turns | 79 | 17 | decision 6, must_include 17, must_not 0, tool 2. |
| Step 3 v2 | 96 turns | 81 | 15 | decision 4, must_include 15, must_not 0, tool 0. |
| Voice v2 | 6 clips | 2 | 3 | Clip 02 not scored; 0 Gemini calls, 6 STT and 6 TTS calls. |

Voice v2 detail: clips 01 and 03 pass; clip 02 has no matching practice turn and is not scored; clips 04, 05, and 06 fail the `answer` criterion because the claim gate sends the spoken claim to identity verification.

## Decisions

### 1. Decision labels

- `answer`: the retrieved documents cover the question and the reply is grounded in them.
- `clarify`: one detail is missing (for example, an order number or product category); ask exactly one question.
- `escalate`: a person must decide, high-value damage needs review, or the customer asks for a human; create a ticket and hand over.
- `refuse`: the request breaks policy or asks for another customer's private data.

### 2. Gate order

Each turn is evaluated in this order: human request -> claim/action gate -> escalation rules -> policy route (only when a policy topic and question cue are present) -> identity gate. A claim or action (damaged, refund, return, cancellation, RTO, and similar) must pass identity verification before any policy answer.

### 3. LLM and embeddings

`gemini-3.5-flash-lite` is used for every live step, with `temperature = 0.0` and `MAX_OUTPUT_TOKENS = 3500`. Dense retrieval uses `paraphrase-multilingual-MiniLM-L12-v2`, combined with BM25 and exact SKU matching in `src/retrieval.py`. The code calls Gemini through its REST endpoint with `requests`; the unused `google-genai` SDK is not a dependency.

### 4. Dates, business days, and refunds

The dataset is frozen on **20 October 2026** (`2026-10-20`), which is "today" for every deadline and refund calculation. Business days are Monday to Saturday, skipping the warehouse holidays in `data/holidays.csv`. Refund escalation uses the REF-2 maximum for the payment method plus a 3-business-day tolerance.

### 5. Hinglish speech

Gnani TTS rejects `hi-en`; Hinglish replies use `language="auto"`, which lets the TIMBRE-V2.5 engine handle code-mixed Hindi-English text.

### 6. Document precedence

Current policy documents outrank agent guides, help articles, FAQs, and marketing pages, in that order. `status: superseded` documents apply only where their successor says they still cover a case.

## Known Limits

- **Return deadline is not computed.** The agent quotes the applicable return window and category rule but does not calculate the customer's exact last eligible date.
- **Escalation replies carry no RRN.** A ticket is created and recorded in `tickets_created.csv`, but the customer-facing escalation reply does not include the ticket reference number. This is why PC010 T4 still fails its must-include check in Step 3 v2 even though the decision is now correct.
- **STT drops key terms.** In voice v2, clip 04's "lautane" arrived as "lutane" (no return cue) and clip 05's "ZentroMart Assured" arrived as "zentomart issued" (no warranty cue). Both turns therefore fell to `clarify` instead of the policy answer.
- **Voice failures are expectation mismatches, not audio failures.** Clip 06's spoken "abhi tak mujhe koi refund nahi mila hai" is a refund claim; the D1 claim gate correctly asks for identity, so the frozen Step 2 `answer` expectation does not apply to that utterance.
- **Gnani usage across all voice runs.** 36 Gnani calls in total: 12 in the discarded first pass, 12 in `step4_voice`, and 12 in `step4_voice_v2` (6 STT + 6 TTS). E3 v2 itself used 12 Gnani calls and 0 Gemini calls. Step 3 v2 used 19 Gemini calls.
- **`run.py voice` is plumbing only.** It hardcodes the stub agent; use `eval/run_voice.py` for the recorded voice evaluation.
