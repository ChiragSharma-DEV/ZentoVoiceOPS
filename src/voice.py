"""Voice layer for VoiceOps 2026.

Integrates Gnani STT and TTS services for voice-based customer support.
All speech requests route strictly through Gnani's API (gnani-vachana SDK).
"""

from __future__ import annotations

import logging
import os
import re
import time
import wave
from pathlib import Path
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv

# Ensure environment variables are loaded BEFORE initializing any Gnani client
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _REPO_ROOT / ".env"
load_dotenv(_ENV_PATH)

from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient
from gnani.tts.client import SUPPORTED_TTS_LANGUAGES
from src.agent_protocol import Agent

logger = logging.getLogger("voiceops.voice")

T = TypeVar("T")

# Month name mapping for date normalisation
_MONTH_MAP: dict[str, str] = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
    "1": "January", "2": "February", "3": "March", "4": "April",
    "5": "May", "6": "June", "7": "July", "8": "August", "9": "September",
    "jan": "January", "january": "January",
    "feb": "February", "february": "February",
    "mar": "March", "march": "March",
    "apr": "April", "april": "April",
    "may": "May",
    "jun": "June", "june": "June",
    "jul": "July", "july": "July",
    "aug": "August", "august": "August",
    "sep": "September", "sept": "September", "september": "September",
    "oct": "October", "october": "October",
    "nov": "November", "november": "November",
    "dec": "December", "december": "December",
}

_MONTH_REGEX = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)


def _retry(func: Callable[[], T], retries: int = 1, delay_sec: float = 1.0) -> T:
    """Execute a callable, retrying once on failure before raising."""
    attempts = retries + 1
    last_exception: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_exception = exc
            logger.warning(
                "Gnani API call failed on attempt %d/%d: %s",
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts:
                time.sleep(delay_sec)
    assert last_exception is not None
    raise last_exception


def _resolve_tts_language(language_code: str | None) -> str:
    """Map language code to a valid BCP-47 code accepted by Gnani timbre-v2.5.

    TIMBRE-V2.5 accepts: auto, bn-IN, en-IN, gu-IN, hi-IN, kn-IN, ml-IN, mr-IN, pa-IN, ta-IN, te-IN.
    Rejects: 'hinglish', 'hi', 'en', 'hindi'.
    """
    if not language_code:
        return "en-IN"
    lang = language_code.strip()
    if lang in SUPPORTED_TTS_LANGUAGES:
        return lang
    lower = lang.lower()
    if lower in {"hinglish", "hi-en", "en-hi"}:
        return "auto"
    if lower in {"hi", "hindi"}:
        return "hi-IN"
    if lower in {"en", "english"}:
        return "en-IN"
    return "en-IN"


def get_audio_duration_seconds(audio_path: str | Path) -> float | None:
    """Return duration of audio file in seconds, or None if unable to determine."""
    p = Path(audio_path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        with wave.open(str(p), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    return None


def transcribe(path: str | Path, language_code: str = "en-IN") -> str:
    """Transcribe an audio file using GnaniSTTClient with 1 retry on failure.

    Args:
        path: Path to the audio file (under 60s).
        language_code: BCP-47 language code (defaults to 'en-IN').

    Returns:
        The transcribed text string.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    def _call() -> str:
        # Load .env again in case keys were added or updated at runtime
        load_dotenv(_ENV_PATH)
        client = GnaniSTTClient()
        res = client.transcribe(str(p), language_code=language_code)
        return str(res.get("transcript", "") or "")

    return _retry(_call, retries=1)


def speak(
    text: str,
    language_code: str = "en-IN",
    out_path: str | Path = "reply.wav",
) -> Path:
    """Synthesize text to speech using GnaniTTSClient with voice Kaveri and model timbre-v2.5.

    Args:
        text: Text to speak.
        language_code: Language code (mapped to valid timbre-v2.5 language).
        out_path: Output file path for reply.wav.

    Returns:
        Path to the synthesized WAV file.
    """
    dest = Path(out_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tts_lang = _resolve_tts_language(language_code)

    def _call() -> Path:
        load_dotenv(_ENV_PATH)
        client = GnaniTTSClient()
        client.synthesize(
            text=text,
            voice="Kaveri",
            model="timbre-v2.5",
            language=tts_lang,
            output_file=str(dest),
        )
        return dest

    return _retry(_call, retries=1)


def normalise_for_speech(text: str, default_year: str = "2026") -> str:
    """Normalise text for natural, clear spoken output.

    Rules applied:
    1. Rupees: '₹' replaced with 'rupees' in spoken text.
    2. Dates: written with year, e.g. '27 October 2026'.
    3. Order IDs: 8-digit order IDs grouped with spaces, e.g. '8978 2837'.
    4. Sentence count: capped at at most 3 sentences.
    """
    if not text:
        return ""

    # 1. Rupee sign handling:
    # Handle '₹ 500' -> '500 rupees', '₹500.50' -> '500.50 rupees'
    text = re.sub(r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)", r"\1 rupees", text)
    # Handle '500 ₹' -> '500 rupees'
    text = re.sub(r"([\d,]+(?:\.\d+)?)\s*₹", r"\1 rupees", text)
    # Handle any remaining standalone '₹'
    text = re.sub(r"₹", "rupees", text)

    # 2. Date normalisation with explicit year:
    # A. ISO dates: 2026-10-27 -> 27 October 2026
    def _repl_iso(m: re.Match[str]) -> str:
        year, month_num, day = m.group(1), m.group(2), m.group(3)
        month_name = _MONTH_MAP.get(month_num, month_num)
        return f"{int(day)} {month_name} {year}"

    text = re.sub(
        r"\b(20\d{2})-(0[1-9]|1[0-2])-([0-2][0-9]|3[01])\b",
        _repl_iso,
        text,
    )

    # B. DD/MM/YYYY or DD-MM-YYYY
    def _repl_dmy(m: re.Match[str]) -> str:
        day, month_num, year = m.group(1), m.group(2), m.group(3)
        month_name = _MONTH_MAP.get(month_num, month_num)
        return f"{int(day)} {month_name} {year}"

    text = re.sub(
        r"\b([0-2]?[0-9]|3[01])[-/](0[1-9]|1[0-2]|[1-9])[-/](20\d{2})\b",
        _repl_dmy,
        text,
    )

    # C. '27 October', '27th Oct 2026', '27 October' (without year)
    def _repl_day_month(m: re.Match[str]) -> str:
        day = m.group(1)
        raw_m = m.group(2).lower()
        month_name = _MONTH_MAP.get(raw_m, m.group(2))
        year = m.group(3) or default_year
        return f"{int(day)} {month_name} {year}"

    text = re.sub(
        rf"\b([0-2]?[0-9]|3[01])(?:st|nd|rd|th)?\s+({_MONTH_REGEX})(?:\s+(20\d{{2}}))?\b",
        _repl_day_month,
        text,
        flags=re.IGNORECASE,
    )

    # D. 'October 27', 'October 27, 2026'
    def _repl_month_day(m: re.Match[str]) -> str:
        raw_m = m.group(1).lower()
        month_name = _MONTH_MAP.get(raw_m, m.group(1))
        day = m.group(2)
        year = m.group(3) or default_year
        return f"{int(day)} {month_name} {year}"

    text = re.sub(
        rf"\b({_MONTH_REGEX})\s+([0-2]?[0-9]|3[01])(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b",
        _repl_month_day,
        text,
        flags=re.IGNORECASE,
    )

    # 3. Order ID grouping: 8 digits become '8978 2837'
    # Match prefixes like 'order OD89782837', 'order 89782837', or standalone 8 digits
    text = re.sub(r"(?i)\b(order|od|id)\s*#?:?\s*(\d{4})(\d{4})\b", r"\1 \2 \3", text)
    text = re.sub(r"\b(?<!\d)(\d{4})(\d{4})(?!\d)\b", r"\1 \2", text)

    # 4. Sentence trimming: keep at most 3 sentences without altering decision
    sentence_boundary = r"(?<!\d)(?<=[.!?])\s+"
    raw_sentences = re.split(sentence_boundary, text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if len(sentences) > 3:
        text = " ".join(sentences[:3])
    else:
        text = " ".join(sentences)

    return text.strip()


def voice_turn(
    audio_path: str | Path,
    agent: Agent,
    language_code: str = "en-IN",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute a single end-to-end voice turn:

    1. Check clip duration (must be <= 60s).
    2. Transcribe via Gnani STT.
    3. If transcript is empty, reply 'Sorry, I didn't catch that. Could you say it again?'.
    4. Otherwise, call agent.respond(customer_text, language_code).
    5. Normalise reply for speech and synthesize reply.wav via Gnani TTS.
    6. Record timings, errors, decisions, used_sections.

    Returns dict containing:
        transcript, reply, decision, used_sections, reply_wav, error, timings.
    """
    total_start = time.perf_counter()
    p = Path(audio_path)
    output_directory = Path(out_dir) if out_dir else p.parent
    reply_wav_path = output_directory / "reply.wav"

    transcript = ""
    reply = ""
    decision = "clarify"
    used_sections: list[str] = []
    error_msg: str | None = None
    spoken_reply = ""

    stt_duration = 0.0
    agent_duration = 0.0
    tts_duration = 0.0

    # Check 60s limit
    duration_sec = get_audio_duration_seconds(p)
    if duration_sec is not None and duration_sec > 60.0:
        error_msg = f"Clip exceeds 60s limit (duration: {duration_sec:.1f}s)"
        total_duration = time.perf_counter() - total_start
        return {
            "transcript": "",
            "reply": "",
            "spoken_reply": "",
            "decision": "refuse",
            "used_sections": [],
            "reply_wav": str(reply_wav_path),
            "error": error_msg,
            "timings": {
                "stt_seconds": 0.0,
                "agent_seconds": 0.0,
                "tts_seconds": 0.0,
                "total_seconds": round(total_duration, 3),
            },
        }

    # Step 1: STT
    stt_start = time.perf_counter()
    try:
        transcript = transcribe(p, language_code=language_code)
        stt_duration = round(time.perf_counter() - stt_start, 3)
    except Exception as exc:
        stt_duration = round(time.perf_counter() - stt_start, 3)
        error_msg = f"STT failed: {exc}"
        decision = "error"
        logger.error("voice_turn STT error for %s: %s", p.name, exc)

    # Step 2: Agent Response
    agent_start = time.perf_counter()
    customer_text = transcript.strip()
    if error_msg:
        decision = "error"
        reply = "An error occurred while processing audio."
    elif not customer_text:
        # Fallback requirement from brief:
        # "if the transcript is empty, reply 'Sorry, I didn't catch that. Could you say it again?'"
        reply = "Sorry, I didn't catch that. Could you say it again?"
        decision = "clarify"
        used_sections = []
    else:
        try:
            agent_result = agent.respond(customer_text, language_code)
            reply = agent_result.get("reply", "")
            decision = agent_result.get("decision", "answer")
            used_sections = agent_result.get("used_sections", [])
        except Exception as exc:
            reply = "I encountered an internal error. Please repeat."
            decision = "error"
            error_msg = f"{error_msg}; Agent error: {exc}" if error_msg else f"Agent error: {exc}"
    agent_duration = round(time.perf_counter() - agent_start, 3)

    # Step 3: Normalise & TTS
    tts_start = time.perf_counter()
    spoken_reply = normalise_for_speech(reply)
    if spoken_reply:
        try:
            speak(spoken_reply, language_code=language_code, out_path=reply_wav_path)
            tts_duration = round(time.perf_counter() - tts_start, 3)
        except Exception as exc:
            tts_duration = round(time.perf_counter() - tts_start, 3)
            tts_err = f"TTS failed: {exc}"
            error_msg = f"{error_msg}; {tts_err}" if error_msg else tts_err
            decision = "error"
            logger.error("voice_turn TTS error for %s: %s", p.name, exc)

    total_duration = round(time.perf_counter() - total_start, 3)

    return {
        "transcript": transcript,
        "reply": reply,
        "spoken_reply": spoken_reply,
        "decision": decision,
        "used_sections": used_sections,
        "reply_wav": str(reply_wav_path),
        "error": error_msg,
        "timings": {
            "stt_seconds": stt_duration,
            "agent_seconds": agent_duration,
            "tts_seconds": tts_duration,
            "total_seconds": total_duration,
        },
    }
