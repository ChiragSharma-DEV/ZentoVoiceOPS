"""LLM client for VoiceOps 2026.

All LLM calls throughout the project route through complete(system, user) -> str.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Tuple
import time
import requests
from dotenv import load_dotenv

# Load environment variables from repo root .env
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 3500  # At least 2048 per Phase 1B requirements

_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")

logger = logging.getLogger("voiceops.llm")


class QuotaExceededError(RuntimeError):
    """Raised immediately on 429 quota exhaustion without retry loops."""
    pass


class LLMResponse(str):
    """String subclass that preserves finish_reason and truncated metadata."""
    finish_reason: str
    truncated: bool

    def __new__(cls, text: str, finish_reason: str = "STOP", truncated: bool = False):
        obj = str.__new__(cls, text)
        obj.finish_reason = finish_reason
        obj.truncated = truncated
        return obj


def complete_with_meta(system: str, user: str) -> LLMResponse:
    """Generate completion and return LLMResponse with text, finish_reason, and truncated."""
    if not _API_KEY:
        raise ValueError("Missing Gemini API key in environment variables (GEMINI_API_KEY).")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent?key={_API_KEY}"
    
    full_user_text = f"{system.strip()}\n\nCustomer question: {user.strip()}" if system and system.strip() else user.strip()
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": full_user_text}]
            }
        ],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS
        }
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"LLM network request failed: {e}")
            time.sleep(2)
            continue

        # Immediate fail on 429 quota exhaustion - do not retry in a loop
        if resp.status_code == 429:
            err_msg = f"HTTP 429 Quota Exceeded: {resp.text}"
            logger.error(err_msg)
            raise QuotaExceededError(err_msg)

        # Retry transient 500/503 errors
        if resp.status_code in (500, 503):
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise RuntimeError(f"LLM API error (status {resp.status_code}): {resp.text}")

        if resp.status_code != 200:
            raise RuntimeError(f"LLM API error (status {resp.status_code}): {resp.text}")

        break

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return LLMResponse("", finish_reason="EMPTY", truncated=False)

    cand = candidates[0]
    finish_reason = cand.get("finishReason", "UNKNOWN")
    truncated = finish_reason in ("MAX_TOKENS", "LENGTH")

    if truncated:
        logger.warning(f"Response was truncated by length! finishReason={finish_reason}")

    parts = cand.get("content", {}).get("parts", [])
    
    # Filter out thought/reasoning parts, keeping only answer text parts
    answer_parts = []
    for p in parts:
        if p.get("thought") is True:
            continue
        if "text" in p:
            answer_parts.append(p["text"])

    # If no separate answer text part was generated (e.g. truncated during thought), do not return thoughts
    result_text = "".join(answer_parts).strip()
    return LLMResponse(result_text, finish_reason=finish_reason, truncated=truncated)


def complete(system: str, user: str) -> str:
    """Generate completion routing through complete(system, user) -> str."""
    return complete_with_meta(system, user)
