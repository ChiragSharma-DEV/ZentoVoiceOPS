# E4 Final Report

## E1 - offline test fix (no API calls)
- `python -m pytest tests`: **59 passed, 0 failed** (wiring 4, modality 1, negative 1, fixes 10, gate 40, voice wiring 3). Gate grew 36 -> 40 by the 4 new tests.
- New tests: the original help sentence, "प्रोडक्ट टूटा हुआ आया" and "मेरा ऑर्डर देर से आया" reach the gate with zero policy calls; "पेजक्राफ्ट नोटबुक रिटर्न कर सकते हैं?" takes the policy route.
- Cue probe `अपने आप चलता है?`: order cue False, claim cue False, policy cue False, so it takes the identity/claim gate, not the policy route. It is not a policy question, so the cue lists were not changed.
- Cause found while adding them: Devanagari words end in matras, which are not regex word chars, so `\b` never closed after `अपने`, `टूटा`, `मेरा`. Possessive and claim cues now match as substrings; ASCII cues keep `\b`.
- `tests/test_phase3c_offline.py` SHA256 `9D8D00B3C32E7F9E14E73F09BEDD65602DCEA040BA4D09D2A77AE26996C72D2D` equals the pre-C recorded hash `9D8D00B3...96C72D2D` (compared against that pre-C value).
- Hashes: conversation.py `2E42191B0E2CEC226B29C73BB0FFA705EE7F7951B12254F830F7A6EF8CC6AC6C`; test_step3_fixes.py `21FE5EF50F01EE702CBA6D2F56253C03F55483D7648CE2A4FF0D9904E00162B8`; test_step3_gate.py `D810E92CB0635455481BE849A99AECB6231CE764657CDCE0B988E8B73659BA4F`; number_words.py `69B39C4576C14762A13888A582A21B93938ECB53CAF74F151A15FD76D8345E5E`; run_voice.py `1F49B21C7AFC9A0B43C3563B707FBE4BBDF418EFCAF7C3A90A6833A7F608909B`.

## E2 - 96-turn batch v2 (`outputs/step3_batch_v2/`, same manifest fields + `code_hashes`)

| Run | Pass | Fail | decision | must_include | must_not | tool_calls | exceptions |
|---|---|---|---|---|---|---|---|
| v1 | 79 | 17 | 6 | 16 | 0 | 2 | 0 |
| rep1 | 79 | 17 | 6 | 16 | 0 | 2 | 0 |
| rep2 | 79 | 17 | 6 | 17 | 0 | 2 | 0 |
| **v2** | **81** | **15** | **4** | **15** | **0** | **0** | **0** |

Changed turns (pass/fail vs v1), with both replies:
- PC005 T4 "क्या मैं किसी इंसान से बात कर सकती हूँ?" fail -> pass. v1: "नमस्ते! मैं ZentroMart की AI सहायक हूँ। मुझे हमारे दस्तावेज़ों में किसी इंसान से बात करने के विकल्प के बारे में जानकारी नहीं मिली है। क्या मैं आपको किसी मानव colleague से कनेक्ट करने में मदद कर सकती हूँ?" / v2: "मैंने इस मामले के लिए टिकट (ticket) दर्ज कर दिया है। हमारी प्रायोरिटी रेज़ोल्यूशन टीम (Priority Resolution team) या सहयोगी आपसे जल्द ही संपर्क करेंगे।"
- PC013 T4 coupon question pass -> fail. v1: "Hello! I am ZentroMart's AI assistant. Main aapko bata doon ki ek order par sirf ek hi coupon apply kiya ja sakta hai. Coupons ko combine nahi kiya ja sakta." / v2: "Hello! I am an AI assistant for ZentroMart. I cannot find that in our documents. Would you like me to connect you with a human colleague?"
- PC013 T5 "mera order kab ship hua tha?" fail -> pass. v1: "Hello! I am an AI assistant for ZentroMart. I cannot find that in our documents. Would you like me to connect you with a human colleague?" / v2: "Hello! I am an AI assistant. Aapka order 2026-10-09 ko ship hua tha."
- PC014 T5 "when was it shipped?" fail -> pass. v1: "I am an AI assistant for ZentroMart. I cannot find the specific shipping date for your order in our documents. Would you like me to connect you with a human colleague?" / v2: "Hello! I am an AI assistant for ZentroMart. According to our verified order facts, your order was shipped on 2026-10-08."
- Attribution: PC005 T4 = Rule 6 human request (ESC-1 item 6); PC010 T4 = Rule 5 refund_delayed (ESC-1 item 5, return refund 2026-07-20). PC013/PC014 reply flips kept the same policy route (1 RAG call each, no section change) and are LLM answer variance, not routing.

New escalations: PC005 T4 (Rule 6 `human_requested`, ESC-1/2/3) and PC010 T4 (Rule 5 `refund_delayed`, ESC-1/2/3). v1 answered both. New policy-route turns: **none** (every v2 policy turn already used the route in v1).

PC010 T4 ("Mera refund kahan hai?"): v2 decision is `escalate`, which now matches the expected decision. The turn still fails must-include: the reply is "Maine aapke order ke liye support ticket create kar diya hai. Hamari Priority Resolution team aapse jald hi sampark karegi." It gives no RRN (315633825001), no ticket number, and no colleague wording.

## E3 - voice v2 (`outputs/step4_voice_v2/`, Prompt B5 mapping, runner unchanged)

**2 of 6 clips pass (01 and 03). Clip 02 is not scored. Clips 04-06 fail.**
| Clip | STT transcript (short) | Decision | Expected | used_sections | TTS | Error | Result |
|---|---|---|---|---|---|---|---|
| 01 | hello my order number is six nine five two zero eight six seven ... pin code one six zero zero one seven | answer | answer | IDV-2 | 3.84s | none | PASS (69520867, verified) |
| 02 | i got the damaged product so i want the five hundred rupees compensation right now | clarify | n/a | STY-1 | 5.12s | none | not scored |
| 03 | hello i want to talk to a human agent | escalate | escalate | ESC-1/2/3 | 5.76s | none | PASS after PC006 setup |
| 04 | mujhe pagecraft a five rule notebook lutane ke liye kitne din milenge | clarify | answer | STY-1 | 5.92s | none | FAIL |
| 05 | sku ele zero zero one two is zentomart issued he kya and how can i take the claim for that | clarify | answer | STY-1 | 6.08s | none | FAIL |
| 06 | credit card vaala refund seventeen october twenty six ko shuru hua tha ... abhi tak mujhe koi refund nahi mila hai ... | clarify | answer | STY-1 | 6.08s | none | FAIL |

`is_asr` is null for all six because `answer_question` was never called (rag_called=false); STT and TTS errors are 0/6 each.
STT vs reference, per clip:
- 01: reference "Hello, my order number is 69520867 and my pincode is 160017. Could you please tell me by what time my order will be delivered?" STT verbalises the digits but keeps the order number, pincode, and delivery question; no key term is lost.
- 02: reference "I got the damaged product, so I want the 500 rupees compensation right now." STT keeps "damaged", "five hundred", and "compensation"; no key term is lost. Not scored because no practice turn matches this demand.
- 03: reference "Hello, I want to talk to a human agent." STT is identical; no key term is lost.
- 04: reference "... Pagecraft A5 Ruled Notebook lautane ke liye kitne din milenge?" STT "... pagecraft a five rule notebook lutane ke liye kitne din milenge". STT turned "lautane" (to return) into "lutane", so the return topic cue disappears and the turn falls to `clarify`.
- 05: reference "... SKU ELE-0012 pe ZentroMart Assured hai kya and how can I take the claim for that?" STT "... sku ele zero zero one two is zentomart issued he kya and how can i take the claim for that". STT turned "Assured" into "issued", so no warranty topic cue survives; "claim" alone is not a policy topic, so the turn falls to `clarify`.
- 06: the customer says "abhi tak mujhe koi refund nahi mila hai" ("no refund yet"), which the frozen Step 2 probe does not contain. D1 treats it as a refund claim and asks for identity, so the gate is correct for this utterance and the Step 2 `answer` expectation does not apply.
Parity vs E2 text: clip 03 = PC006 T4 -> voice escalate, text escalate; same decision and same must-include result (both pass, same ESC-2 confirmation), no difference. Clip 01 has no E2 turn (order 69520867 is absent from the batch); clips 04-06 are Step 2 frozen probes with no Step 3 turn, so parity is not applicable. The clip 04-06 failures are caused by the STT term loss and the claim expectation above, not by routing or audio quality.

## API calls and quota
- E1 offline pytest: Gemini 0, Gnani 0.
- E2: v1 21 Gemini, rep1 21, rep2 21, v2 19; Gnani 0 in every run. v2 quota retries observed 0.
- E3: Gemini 0 (no RAG call reached), Gnani STT 6, Gnani TTS 6.
- Quota at run start: no 429 and no quota exhaustion on either Gemini key at the start of E2 v2 and E3; `gemini_quota_errors` 0, quota retries 0. E3 consumed no Gemini quota at all.

**Status: E1 gate passed. E2 completed (81/96). E3 completed: 2 of 6 clips pass (01, 03), clip 02 is not scored, clips 04-06 failed the specified criterion, so no further runs were made.**
