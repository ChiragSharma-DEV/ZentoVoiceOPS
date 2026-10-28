# VoiceOps 2026 — Demo Script

**Target runtime:** 2:00 maximum. **Current recordable length:** 1:02 (0:00–0:35, 1:20–1:32 and 1:45–2:00); the rest is blocked by missing artifacts.
**Rule:** no stub output is shown. A shot marked **GAP** must not be recorded or replaced with a stand-in. All timestamps are cumulative.

## Qualification check (read before recording)

Intended clip: `voice_clips/05_sku_warranty.aac`. Its transcript asks about SKU ELE-0012, ZentroMart Assured and how to claim. The answer is in KB section **WAR-2** (warranty) and catalog row `products.csv:ELE-0012` [outputs/voice/transcripts_review.md].

The briefing requires a clip whose Step 3 result passes in `outputs/`. That condition cannot be met: `outputs/step3_results.csv` does not exist, and all 6 voice replies are stub output [outputs/voice/summary.csv; outputs/voice/05_sku_warranty/reply.txt]. The Q&A, follow-up and handover shots below are therefore gaps, not recordable scenes. Do not play `reply.wav`: it speaks the stub echo, not a policy answer.

## Shot list

| Time | Shot | Audio / visual | Source file | Status |
| :--- | :--- | :--- | :--- | :--- |
| 0:00–0:10 | One sentence: "VoiceOps 2026 is a speech-enabled support agent for ZentroMart that answers from company documents, verifies an order before disclosing it, and hands over to a human when policy requires." | Narrator over the repo tree or title card. | Narration (no file) | **RECORDABLE** |
| 0:10–0:20 | Play the customer clip. Caption the first line of the transcript. | Audio: `05_sku_warranty.aac`. Visual: transcript text. | `voice_clips/05_sku_warranty.aac`; `outputs/voice/05_sku_warranty/transcript.txt` | **RECORDABLE** (STT transcript is real) |
| 0:20–0:35 | Show the full STT transcript: "sku ele zero zero one two is zentomart issued he kya and how can i take the claim for that". Point at `ele zero zero one two` as the pre-normalisation form. | Screen: transcript file, highlighted token. | `outputs/voice/05_sku_warranty/transcript.txt` | **RECORDABLE** |
| 0:35–0:50 | **GAP.** Show the reply text for the same turn. There is no non-stub reply to show. | Nothing is displayed. | Would come from `outputs/voice/05_sku_warranty/reply.txt` (currently `[STUB] Echo acknowledged`) | **GAP** |
| 0:50–1:05 | **GAP.** Play the reply audio for the same turn. `reply.wav` is TTS of the stub echo, so it is excluded. | Nothing is played. | Would come from `outputs/voice/05_sku_warranty/reply.wav` | **GAP** |
| 1:05–1:20 | **GAP.** Follow-up in the same conversation: "Can I claim it from the app, and how long is the warranty?" No conversation state or turn record exists to show. | Nothing is displayed. | Would come from `outputs/step3_results.csv` (absent) | **GAP** |
| 1:20–1:32 | Handover: play `03_human_agent.aac` and show the transcript "hello i want to talk to a human agent". | Audio: clip. Visual: transcript text. | `voice_clips/03_human_agent.aac`; `outputs/voice/03_human_agent/transcript.txt` | **RECORDABLE** (STT transcript is real) |
| 1:32–1:45 | **GAP.** Show the decision (`escalate`) and the handover record (ticket summary). The only stored voice decision is a stub `answer`, and no Step 3 run produced a ticket. | Nothing is displayed. | Would come from `outputs/step3_results.csv` (absent); the stub decision is in `outputs/voice/03_human_agent/result.json` | **GAP** |
| 1:45–2:00 | One sentence: "What was not done: the voice path still runs a stub agent, Step 3 has no batch run, the Step 2 regression is unfixed, and the 2E ablation has no result yet." | Narrator over the gap list. | Narration (no file) | **RECORDABLE** |

## Section ID used by the answer

**None.** No non-stub answer was produced for this clip, so no `used_sections` value exists to name. The section the real answer must cite is **WAR-2**, with the catalog row `products.csv:ELE-0012`; both are identified from the KB in `outputs/voice/transcripts_review.md`, not from a run.

## Gap log (what must exist before the blocked shots can be recorded)

1. A real `ConversationAgent` run over `05_sku_warranty.aac` that writes a non-stub `reply.txt` and `reply.wav`. Today the only reply is `[STUB] Echo acknowledged` [outputs/voice/05_sku_warranty/reply.txt].
2. A retrieval record for that turn naming `WAR-2` and `products.csv:ELE-0012` (a retrieved-chunks file or `used_sections` entry).
3. A two-turn conversation record for the follow-up, stored in `outputs/step3_results.csv`.
4. A handover record for `03_human_agent.aac`: decision `escalate` plus the ticket summary. No such row exists; `outputs/step3_results.csv` is absent and the voice decision is a stub [outputs/voice/03_human_agent/result.json].

Do not record the gap shots (0:35–1:20 and 1:32–1:45) until items 1–4 exist. If time is short, publish the 1:02 cut with this gap log attached; do not fill it with stub audio.
