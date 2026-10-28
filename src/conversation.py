"""Multi-turn conversation engine for VoiceOps 2026 Step 3.

Manages conversational state, identity verification, tool execution (lookup_order, create_ticket),
policy enforcement, date/money calculations, section validation, and human escalations.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.data import (
    REFERENCE_DATE_OBJ,
    PRODUCTS_DF,
    get_customer,
    get_order,
    get_product,
    business_days_between,
)
from src.number_words import normalise_digit_words
from src.order_tools import lookup_order_unverified, lookup_order_verified, handover

logger = logging.getLogger("voiceops.conversation")

# Rule 6: hand over only when a person noun is paired with a request cue.
# Broad nouns alone are descriptions ("delivery person", "agent-approved"),
# not requests for a human colleague.
HUMAN_NOUNS = [
    "human", "person", "agent", "executive", "representative",
    "insan", "insaan", "aadmi", "vyakti",
    "colleague", "customer care", "support executive",
    "इंसान", "आदमी", "व्यक्ति", "मानव",
]
HUMAN_REQUEST_CUES = ["talk", "speak", "connect", "baat", "बात"]
_HUMAN_CUE_RE = re.compile(r"\b(?:talk|speak|connect|baat)\b", re.IGNORECASE)


def _is_human_request(text: str) -> bool:
    """Return True only for a person noun plus an explicit request cue (ESC-1 item 6)."""
    msg_lower = text.lower()
    if not any(noun in msg_lower for noun in HUMAN_NOUNS):
        return False
    if _HUMAN_CUE_RE.search(msg_lower):
        return True
    return any(cue in msg_lower for cue in HUMAN_REQUEST_CUES if not cue.isascii())


# Cache for valid section IDs parsed from KB documents and product catalog
_VALID_SECTION_IDS: Optional[Set[str]] = None


def get_valid_section_ids() -> Set[str]:
    """Return the set of valid section IDs from KB chunks and product rows."""
    global _VALID_SECTION_IDS
    if _VALID_SECTION_IDS is None:
        from src.kb import load_and_chunk_kb

        chunks = load_and_chunk_kb()
        kb_ids = {c.section_id for c in chunks}
        prod_ids = {f"products.csv:{sku}" for sku in PRODUCTS_DF["sku"].unique()}
        _VALID_SECTION_IDS = kb_ids | prod_ids
    return _VALID_SECTION_IDS


def validate_used_sections(section_ids: List[str]) -> Tuple[List[str], List[str]]:
    """Validate section IDs against valid KB and product catalog IDs.

    Drops unknown IDs, logs a warning for each dropped ID, and returns
    (valid_ids, dropped_ids).
    """
    valid_set = get_valid_section_ids()
    valid_ids: List[str] = []
    dropped_ids: List[str] = []

    for sid in section_ids:
        sid_clean = str(sid).strip()
        if not sid_clean:
            continue
        if sid_clean in valid_set:
            if sid_clean not in valid_ids:
                valid_ids.append(sid_clean)
        else:
            dropped_ids.append(sid_clean)
            logger.warning("Dropping unknown section ID: %s", sid_clean)

    return valid_ids, dropped_ids


@dataclass
class ConversationState:
    """Conversational state persisted across turns."""

    order_id: Optional[str] = None
    verified: bool = False
    verification_attempts: int = 0  # 0 to 2
    issue_summary: str = ""
    handover_created: bool = False
    turns: List[Dict[str, Any]] = field(default_factory=list)  # Last 6 turns
    order_data: Optional[Dict[str, Any]] = None  # Full order details when verified
    language: str = "en"  # 'en', 'hi', or 'hi-en'
    dropped_sections_count: int = 0

    def add_turn(self, turn_info: Dict[str, Any]) -> None:
        """Append turn info and keep only the last 6 turns."""
        self.turns.append(turn_info)
        if len(self.turns) > 6:
            self.turns = self.turns[-6:]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for logging and result tracking."""
        return {
            "order_id": self.order_id,
            "verified": self.verified,
            "verification_attempts": self.verification_attempts,
            "issue_summary": self.issue_summary,
            "handover_created": self.handover_created,
            "turn_count": len(self.turns),
            "dropped_sections_count": self.dropped_sections_count,
            "last_6_turns": self.turns[-6:],
        }


def detect_language(text: str) -> str:
    """Detect whether user text is English ('en'), Hindi ('hi'), or Hinglish ('hi-en')."""
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    hinglish_markers = {
        "mujhe", "baare", "mein", "karna", "chahiye", "hai", "hain", "kya", "aur",
        "aakhri", "batayein", "nahi", "kar", "sakti", "sakta", "hoon", "tha", "thi",
        "kaise", "kab", "kitne", "din", "wapas", "do", "na", "mera", "meri", "namaste",
    }
    words = {w.lower() for w in re.findall(r"[a-zA-Z]+", text)}
    if len(words & hinglish_markers) >= 1:
        return "hi-en"
    return "en"


def extract_order_id(text: str) -> Optional[str]:
    """Extract an 8-digit ZentroMart order ID from customer text."""
    m = re.search(r"\b(\d{4})\s*(\d{4})\b", text)
    if m:
        return m.group(1) + m.group(2)
    m_single = re.search(r"\b(\d{8})\b", text)
    if m_single:
        return m_single.group(1)
    return None


# C1: cues that make a turn about a specific order, refund, delivery, or return.
ORDER_TOPIC_TERMS = [
    "order", "refund", "delivery", "return", "parcel", "package",
    "shipment", "tracking", "ऑर्डर", "रिफंड", "डिलीवरी", "रिटर्न",
]
ORDER_POSSESSIVES = [
    "my", "mine", "mera", "meri", "mere",
    "apne", "apna", "apni",
    "अपने", "अपना", "अपनी", "मेरा", "मेरी", "मेरे",
]
ORDER_STATUS_CUES = [
    "track", "status", "where", "kab aayega", "kahan hai",
    "refund status", "cancel my order",
]

# D1: claim/action turns must pass identity verification before any policy answer.
_CLAIM_OR_ACTION_RE = re.compile(
    r"\b(?:damaged?|broken|toota|kharab|defective|missing|late|delayed?|"
    r"galat|voucher|compensation|cancel|replace|exchange)\b",
    re.IGNORECASE,
)
# Devanagari letters end in combining marks (matras), which are not regex word
# characters, so \b would never close after them. Match these as substrings.
_CLAIM_DEVA_RE = re.compile(
    r"(?:टूटा|टूटी|खराब|नहीं आया|नहीं मिला|देर|गलत|वाउचर|रिफ़ंड|रिफंड|"
    r"कैंसल|रद्द|बदलना)"
)
_CLAIM_PHRASES = ("not received", "nahi aaya", "wrong item", "return this")
_REFUND_COMPLAINT_RE = re.compile(
    r"\b(?:not|where|pending|delayed)\b|nahi|abhi|kahan|kab",
    re.IGNORECASE,
)
POLICY_TOPIC_CUES = (
    "return", "refund", "warranty", "policy", "delivery", "shipping",
    "cancellation", "exchange", "replacement",
    "रिटर्न", "रिफ़ंड", "रिफंड", "वारंटी", "पॉलिसी", "डिलीवरी", "शिपिंग",
    "कैंसिल", "एक्सचेंज",
)
POLICY_QUESTION_CUES = (
    "can i", "could i", "what", "how", "when", "where", "why",
    "kya", "kaise", "kitne", "kitna", "kab", "kahan",
    "क्या", "कैसे", "कितने", "कितना", "कब", "कहाँ", "कहां", "सकते", "सकता",
)

# REF-2 maximum refund timelines in business days. NEFT is 5-7 (7); net
# banking 3-5 (5); card 5-7 (7); UPI 1-3 (3). Wallet is "Instant, within
# 2 hours" -> 0 business days, with the same +3 tolerance.
REFUND_MAX_BUSINESS_DAYS: Dict[str, int] = {
    "wallet": 0,
    "cod_wallet": 0,
    "upi": 3,
    "netbanking": 5,
    "card": 7,
    "cod_bank": 7,
    "neft": 7,
}
REFUND_ESCALATION_TOLERANCE_DAYS = 3  # ESC-1 item 5


def _has_order_specific_cue(text: str, extraction_text: Optional[str] = None) -> bool:
    """Return True when the turn refers to a specific order/refund/delivery/return.

    Spoken digit words are normalised before this check, so a spoken 8-digit
    order ID is already contiguous and is found by extract_order_id().
    """
    msg_lower = text.lower()

    if extract_order_id(extraction_text or text):
        return True

    has_possessive = any(
        (
            re.search(rf"\b{re.escape(word)}\b", msg_lower)
            if word.isascii()
            else word in msg_lower
        )
        for word in ORDER_POSSESSIVES
    )
    if has_possessive and any(term in msg_lower for term in ORDER_TOPIC_TERMS):
        return True

    return any(cue in msg_lower for cue in ORDER_STATUS_CUES)


def _has_claim_or_action_cue(text: str) -> bool:
    """Return True for a claim/action that must pass identity verification first."""
    msg_lower = text.lower()
    if _CLAIM_OR_ACTION_RE.search(msg_lower):
        return True
    if _CLAIM_DEVA_RE.search(text):
        return True
    if any(phrase in msg_lower for phrase in _CLAIM_PHRASES):
        return True
    if re.search(r"\brefund\b", msg_lower):
        return bool(_REFUND_COMPLAINT_RE.search(msg_lower))
    return False


def _is_policy_question(text: str) -> bool:
    """Return True for a general policy question, not a claim or greeting."""
    msg_lower = text.lower()
    has_topic = any(topic in msg_lower for topic in POLICY_TOPIC_CUES)
    has_question = any(cue in msg_lower for cue in POLICY_QUESTION_CUES)
    return has_topic and has_question


def _refund_max_business_days(method: str) -> int:
    """Return the REF-2 business-day maximum, defaulting to 5 days."""
    normalized = str(method or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in REFUND_MAX_BUSINESS_DAYS:
        return REFUND_MAX_BUSINESS_DAYS[normalized]
    if "wallet" in normalized:
        return 0
    if "netbanking" in normalized or "net_banking" in normalized:
        return 5
    if "cod_bank" in normalized or "neft" in normalized:
        return 7
    if "card" in normalized:
        return 7
    if "upi" in normalized:
        return 3
    return 5


def extract_verification_digits(text: str) -> Optional[str]:
    """Extract 4-digit phone suffix or 6-digit delivery pincode typed by customer."""
    clean_text = re.sub(r"\b\d{4}\s*\d{4}\b|\b\d{8}\b", " ", text)
    digits = "".join(ch for ch in clean_text if ch.isdigit())
    if len(digits) in (4, 6):
        return digits
    m6 = re.search(r"\b(\d{6})\b", clean_text)
    if m6:
        return m6.group(1)
    m4 = re.search(r"\b(\d{4})\b", clean_text)
    if m4:
        return m4.group(1)
    return digits if digits else None


def format_date_with_month(date_val: date | str) -> str:
    """Format date to 'DD Month YYYY' per STY-2 / STY-3."""
    if isinstance(date_val, str):
        date_obj = datetime.strptime(date_val[:10], "%Y-%m-%d").date()
    else:
        date_obj = date_val
    en_month = date_obj.strftime("%B")
    return f"{date_obj.day} {en_month} {date_obj.year}"


def is_escalation_requested(
    message: str,
    state: ConversationState,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if the conversation must be handed over to a human colleague per ESC-1.

    Returns (should_escalate, reason, intent).
    """
    msg_lower = message.lower()

    # Rule 6: Customer asks for a human agent
    if _is_human_request(msg_lower):
        return True, "human_requested", "handoff"

    # Rule 3: Legal action or court
    if any(k in msg_lower for k in ["legal", "court", "police", "lawyer", "consumer forum"]):
        return True, "legal_or_complaint", "legal"

    # Rule 2: Unauthorized account activity
    if any(k in msg_lower for k in ["didn't order", "did not order", "unauthorized", "fraud", "hacked"]):
        return True, "unauthorized_activity", "fraud"

    # Rule 7: Product safety risk
    if any(k in msg_lower for k in ["sparks", "burning", "fire", "injury", "swelling battery", "smoke"]):
        return True, "product_safety_risk", "safety"

    order = state.order_data
    if order:
        # Rule 4: Third or later contact about same issue within 7 days
        contacts_last_7 = int(order.get("contacts_last_7_days") or 0)
        if contacts_last_7 >= 2:
            return True, "repeat_contact", "repeat_contact"

        # Rule 1: Damaged, defective or wrong item worth more than ₹10,000
        damage_keywords = [
            "damage", "damaged", "broken", "defective", "toota", "kharab", "defect",
            "screen broken", "replacement approve"
        ]
        if any(k in msg_lower for k in damage_keywords):
            items = order.get("items", []) or []
            if not items:
                return True, "high_value_damage_price_missing", "damage"
            for item in items:
                raw_price = item.get("unit_price_inr")
                if raw_price is None or str(raw_price).strip() == "":
                    return True, "high_value_damage_price_missing", "damage"
                try:
                    price = int(float(raw_price))
                except (TypeError, ValueError):
                    return True, "high_value_damage_price_missing", "damage"
                if price > 10000:
                    return True, "high_value_damage", "damage"

        # Rule 5: Refund more than 3 business days past maximum timeline.
        # ESC-1 item 5 covers every refund: REF-1 defines cancellation, return,
        # and refused/RTO refunds, and REF-2 line 20 applies one timeline table
        # to all of them. Each source uses its own initiation date and method.
        refund_status = str(order.get("return_status") or "").lower()
        refund_context = (
            "refund" in msg_lower
            or "cancel" in msg_lower
            or "rto" in msg_lower
            or "pending" in refund_status
            or "delayed" in refund_status
        )
        if refund_context:
            refund_sources = [
                ("return", "return_refund_initiated_on", "return_refund_method"),
                (
                    "cancellation",
                    "cancellation_refund_initiated_on",
                    "cancellation_refund_method",
                ),
                ("rto", "rto_refund_initiated_on", "rto_refund_method"),
            ]
            for _source, date_field, method_field in refund_sources:
                refund_initiated = order.get(date_field)
                if not refund_initiated:
                    continue
                pay_method = order.get(method_field) or order.get("payment_method")
                max_biz_days = _refund_max_business_days(str(pay_method or ""))
                elapsed_biz_days = business_days_between(
                    refund_initiated, REFERENCE_DATE_OBJ
                )
                if elapsed_biz_days > (
                    max_biz_days + REFUND_ESCALATION_TOLERANCE_DAYS
                ):
                    return True, "refund_delayed", "refund"

            if not any(order.get(field) for _, field, _ in refund_sources):
                logger.warning(
                    "Refund timeline check skipped: order %s has no "
                    "return_refund_initiated_on, cancellation_refund_initiated_on, "
                    "or rto_refund_initiated_on",
                    order.get("order_id", "unknown"),
                )

    return False, None, None


def _build_order_context(
    order_data: Optional[Dict[str, Any]],
    context_prefix: Optional[str] = None,
) -> str:
    """Serialize verified order facts without applying policy rules."""
    lines: List[str] = []
    if context_prefix:
        prefix = context_prefix.strip()
        if prefix:
            lines.append(prefix)

    order = order_data or {}
    if order:
        payment_method = str(order.get("payment_method") or "unknown").strip()
        delivery_date = str(
            order.get("delivery_date") or order.get("delivered_on") or "unknown"
        ).strip()
        status = str(order.get("status") or "unknown").strip()
        shipped_on = str(order.get("shipped_on") or "unknown").strip()

        is_plus_value = order.get("is_plus")
        if isinstance(is_plus_value, str):
            is_plus = "true" if is_plus_value.strip().lower() in {"true", "1", "yes"} else "false"
        elif is_plus_value is None:
            is_plus = "unknown"
        else:
            is_plus = "true" if bool(is_plus_value) else "false"

        categories: List[str] = []
        for item in order.get("items", []) or []:
            category = item.get("return_category") or item.get("category")
            category_text = str(category or "").strip()
            if category_text and category_text not in categories:
                categories.append(category_text)

        lines.extend(
            [
                "VERIFIED ORDER FACTS:",
                f"- payment_method: {payment_method}",
                f"- delivery_date: {delivery_date}",
                f"- shipped_on: {shipped_on}",
                f"- status: {status}",
                f"- is_plus: {is_plus}",
                f"- item_categories: {', '.join(categories) if categories else 'unknown'}",
            ]
        )

    return "\n".join(lines)


def rag_answer_adapter(
    question: str,
    order_data: Optional[Dict[str, Any]] = None,
    language: str = "en",
    state: Optional[ConversationState] = None,
    context_prefix: Optional[str] = None,
    retriever: Optional[Any] = None,
    top_k: int = 10,
    enable_rewrite: bool = False,
    is_asr: bool = False,
) -> Dict[str, Any]:
    """Delegate policy answering to src.rag.answer_question.

    The adapter forwards the shared retriever and modality gate, adds verified
    order facts to the question, and returns only the grounded decision, reply,
    and cited sections. It never synthesises fallback policy text.
    """
    if retriever is None:
        raise ValueError("rag_answer_adapter requires a shared retriever")

    contextual_question = question
    context_block = _build_order_context(order_data, context_prefix)
    if context_block:
        contextual_question = f"{question}\n\n{context_block}"

    from src.rag import answer_question

    result = answer_question(
        question=contextual_question,
        retriever=retriever,
        top_k=top_k,
        enable_rewrite=enable_rewrite,
        is_asr=is_asr,
    )

    if result.get("finish_reason") == "ERROR":
        raise RuntimeError(f"answer_question failed: {result.get('raw_output', '')}")

    decision = str(result.get("decision") or "").strip().lower()
    reply = str(result.get("reply") or "").strip()
    if decision not in {"answer", "clarify", "escalate", "refuse"} or not reply:
        raise RuntimeError("answer_question did not return a valid grounded answer")

    used_sections = result.get("used_sections") or []
    if not isinstance(used_sections, list):
        used_sections = [str(used_sections)]

    return {
        "decision": decision,
        "reply": reply,
        "used_sections": [
            str(section).strip() for section in used_sections if str(section).strip()
        ],
    }


def _validate_input_modality(value: str) -> str:
    """Return a supported per-turn modality or raise for an unknown value."""
    modality = str(value or "text").strip().lower()
    if modality not in {"text", "voice"}:
        raise ValueError(f"Unsupported input_modality: {value!r}; expected 'text' or 'voice'")
    return modality


def _adapter_accepts_is_asr(adapter: Callable[..., Dict[str, Any]]) -> bool:
    """Check whether a custom adapter can accept the modality keyword."""
    try:
        parameters = inspect.signature(adapter).parameters
    except (TypeError, ValueError):
        return True
    return "is_asr" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


class ConversationAgent:
    """Multi-turn customer support conversation agent for VoiceOps."""

    def __init__(
        self,
        state: Optional[ConversationState] = None,
        answer_adapter: Optional[Callable[..., Dict[str, Any]]] = None,
        retriever: Optional[Any] = None,
        top_k: int = 10,
        enable_rewrite: bool = False,
        input_modality: str = "text",
    ) -> None:
        self.state = state or ConversationState()
        self._retriever = retriever
        self.top_k = top_k
        self.enable_rewrite = enable_rewrite
        self.input_modality = _validate_input_modality(input_modality)
        self.answer_adapter = answer_adapter or self._default_answer_adapter
        self._adapter_accepts_is_asr = _adapter_accepts_is_asr(self.answer_adapter)

    @property
    def retriever(self) -> Any:
        """Return the one shared retriever for this agent, creating it lazily."""
        if self._retriever is None:
            from src.retrieval import HybridRetriever

            self._retriever = HybridRetriever(use_query_rewrite=False)
        return self._retriever

    def _default_answer_adapter(
        self,
        question: str,
        order_data: Optional[Dict[str, Any]] = None,
        language: str = "en",
        state: Optional[ConversationState] = None,
        is_asr: bool = False,
    ) -> Dict[str, Any]:
        """Call the shared RAG adapter with this agent's retriever settings."""
        return rag_answer_adapter(
            question=question,
            order_data=order_data,
            language=language,
            state=state,
            retriever=self.retriever,
            top_k=self.top_k,
            enable_rewrite=self.enable_rewrite,
            is_asr=is_asr,
        )

    def _handover_response(
        self,
        customer_text: str,
        lang: str,
        reason: str,
        intent: str,
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create the ESC-2 ticket and return the fixed-flow handover reply.

        The ticket carries the verified order ID only when identity verification
        succeeded; otherwise it is created without an order ID and the reply says so.
        """
        verified_order_id = self.state.order_id if self.state.verified else None
        if verified_order_id:
            summary = (
                f"Verified caller on order {verified_order_id}; customer reports issue: "
                f"'{customer_text[:60]}'; checks completed; escalated to Priority Resolution "
                f"team per policy {reason}."
            )
        else:
            summary = (
                f"Customer asked for a human; stated reason: '{customer_text[:60]}'; "
                f"no verified order ID; identity check not completed; escalated to "
                f"Priority Resolution team per policy {reason}."
            )
        self.state.issue_summary = summary

        ticket_res = handover(
            summary=summary,
            order_id=verified_order_id,
            reason=reason,
            intent=intent,
        )
        tool_calls.append({
            "name": "create_ticket",
            "arguments": {},
        })
        self.state.handover_created = ticket_res.get("created", False)

        decision = "escalate"
        raw_sections = ["ESC-1", "ESC-2", "ESC-3"]
        valid_sections, dropped = validate_used_sections(raw_sections)
        self.state.dropped_sections_count += len(dropped)

        # FIXED FLOW: handover confirmation
        if lang == "hi":
            reply = (
                "मैंने इस मामले के लिए टिकट (ticket) दर्ज कर दिया है। हमारी प्रायोरिटी "
                "रेज़ोल्यूशन टीम (Priority Resolution team) या सहयोगी आपसे जल्द ही संपर्क करेंगे।"
            )
            if not verified_order_id:
                reply = (
                    "मैंने इस मामले के लिए बिना सत्यापित ऑर्डर आईडी के टिकट (ticket) दर्ज कर "
                    "दिया है। हमारी प्रायोरिटी रेज़ोल्यूशन टीम (Priority Resolution team) आपसे "
                    "जल्द ही संपर्क करेंगे।"
                )
        elif lang == "hi-en":
            reply = (
                "Maine aapke order ke liye support ticket create kar diya hai. Hamari Priority "
                "Resolution team aapse jald hi sampark karegi."
            )
            if not verified_order_id:
                reply = (
                    "Maine bina verified order ID ke support ticket create kar diya hai. "
                    "Hamari Priority Resolution team aapse jald hi sampark karegi."
                )
        else:
            reply = (
                "I have created a support ticket for this issue. Our Priority Resolution "
                "colleague will review the details and contact you shortly."
            )
            if not verified_order_id:
                reply = (
                    "I have created a support ticket without a verified order ID. Our "
                    "Priority Resolution colleague will review the details and contact you shortly."
                )

        self.state.add_turn({
            "user": customer_text,
            "decision": decision,
            "reply": reply,
            "tool_calls": tool_calls,
        })
        return {
            "decision": decision,
            "reply": reply,
            "used_sections": valid_sections,
            "dropped_sections": dropped,
            "tool_calls": tool_calls,
            "state": self.state.to_dict(),
        }

    def _policy_response(
        self,
        customer_text: str,
        lang: str,
        is_asr: bool,
        tool_calls: List[Dict[str, Any]],
        order_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route a policy turn to the answer adapter, with optional order facts."""
        adapter_kwargs: Dict[str, Any] = {
            "question": customer_text,
            "order_data": order_data,
            "language": lang,
            "state": self.state,
        }
        if self._adapter_accepts_is_asr:
            adapter_kwargs["is_asr"] = is_asr
        adapter_output = self.answer_adapter(**adapter_kwargs)

        decision = adapter_output.get("decision", "answer")
        reply = adapter_output.get("reply", "")
        raw_sections = adapter_output.get("used_sections", [])

        valid_sections, dropped = validate_used_sections(raw_sections)
        self.state.dropped_sections_count += len(dropped)

        self.state.add_turn({
            "user": customer_text,
            "decision": decision,
            "reply": reply,
            "tool_calls": tool_calls,
        })
        return {
            "decision": decision,
            "reply": reply,
            "used_sections": valid_sections,
            "dropped_sections": dropped,
            "tool_calls": tool_calls,
            "state": self.state.to_dict(),
        }

    def respond(
        self,
        customer_text: str,
        language_code: Optional[str] = None,
        input_modality: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a single turn and return decision, reply, tool calls, and state."""
        cleaned_text = (customer_text or "").strip()
        lang = detect_language(cleaned_text)
        self.state.language = lang
        modality = (
            self.input_modality
            if input_modality is None
            else _validate_input_modality(input_modality)
        )
        is_asr = modality == "voice"
        # B4: spoken digit words are normalised for extraction only, voice only.
        extraction_text = (
            normalise_digit_words(cleaned_text) if is_asr else cleaned_text
        )

        tool_calls: List[Dict[str, Any]] = []

        # ESC-1 item 6 / ESC-2: hand over at once, before identity verification.
        if _is_human_request(cleaned_text):
            return self._handover_response(
                cleaned_text, lang, "human_requested", "handoff", tool_calls
            )

        # D1 claim gate: a claim/action turn must pass identity verification;
        # it is never answered by the policy route.
        claim_or_action = _has_claim_or_action_cue(cleaned_text)
        verification_in_progress = (
            self.state.order_id is not None or self.state.verified
        )

        # ESC-1 escalation rules run before policy routing and before the gate.
        should_escalate, esc_reason, esc_intent = is_escalation_requested(
            cleaned_text, self.state
        )
        if should_escalate:
            return self._handover_response(
                cleaned_text,
                lang,
                esc_reason or "unknown",
                esc_intent or "handoff",
                tool_calls,
            )

        # C1 / IDV-1: only a general policy question with no order cue and no
        # claim/action cue may skip identity verification.
        if (
            not verification_in_progress
            and not claim_or_action
            and not _has_order_specific_cue(cleaned_text, extraction_text)
            and _is_policy_question(cleaned_text)
        ):
            return self._policy_response(
                cleaned_text, lang, is_asr, tool_calls, order_data=None
            )

        # =====================================================================
        # Step A: Check if an Order ID is introduced (FIXED FLOW)
        # =====================================================================
        found_order_id = extract_order_id(extraction_text)
        if found_order_id and self.state.order_id is None:
            self.state.order_id = found_order_id
            self.state.verified = False
            self.state.verification_attempts = 0
            self.state.order_data = None

            tool_calls.append({
                "name": "lookup_order",
                "arguments": {"order_id": int(found_order_id)},
            })
            lookup_order_unverified(found_order_id)

            decision = "clarify"
            raw_sections = ["IDV-1", "IDV-2"]
            valid_sections, dropped = validate_used_sections(raw_sections)
            self.state.dropped_sections_count += len(dropped)

            # FIXED FLOW: request for verification value
            if lang == "hi":
                reply = (
                    "नमस्ते, विवरण सत्यापित करने के लिए, कृपया अपने पंजीकृत फ़ोन नंबर "
                    "के आख़िरी 4 अंक (last 4 digits) या डिलीवरी पिनकोड बताएं।"
                )
            elif lang == "hi-en":
                reply = (
                    "Thank you. Order details verify karne ke liye, kripya apne registered "
                    "phone number ke last 4 digits batayein."
                )
            else:
                reply = (
                    "Thank you. Before I can share details, please provide the last 4 digits "
                    "of your registered phone number."
                )

            self.state.add_turn({
                "user": cleaned_text,
                "decision": decision,
                "reply": reply,
                "tool_calls": tool_calls,
            })
            return {
                "decision": decision,
                "reply": reply,
                "used_sections": valid_sections,
                "dropped_sections": dropped,
                "tool_calls": tool_calls,
                "state": self.state.to_dict(),
            }

        # =====================================================================
        # Step B: Initial Greeting (FIXED FLOW)
        # =====================================================================
        if self.state.order_id is None:
            decision = "clarify"
            raw_sections = ["STY-1"]
            valid_sections, dropped = validate_used_sections(raw_sections)
            self.state.dropped_sections_count += len(dropped)

            # FIXED FLOW: opening greeting & AI identification
            if lang == "hi":
                reply = (
                    "नमस्ते, मैं ज़ेंट्रोमार्ट का एआई (AI) सपोर्ट असिस्टेंट हूँ। "
                    "कृपया अपना ऑर्डर नंबर (order ID) बताएं ताकि मैं आपकी सहायता कर सकूँ।"
                )
            elif lang == "hi-en":
                reply = (
                    "Hello! Main ZentroMart ka AI support assistant hoon. "
                    "Kripya apna order number share karein taaki main aapki help kar sakoon."
                )
            else:
                reply = (
                    "Hello! I am ZentroMart's AI support assistant. "
                    "Could you please share your order number so I can help you?"
                )

            self.state.add_turn({
                "user": cleaned_text,
                "decision": decision,
                "reply": reply,
                "tool_calls": tool_calls,
            })
            return {
                "decision": decision,
                "reply": reply,
                "used_sections": valid_sections,
                "dropped_sections": dropped,
                "tool_calls": tool_calls,
                "state": self.state.to_dict(),
            }

        # =====================================================================
        # Step C: Verification in Progress (FIXED FLOW)
        # =====================================================================
        if not self.state.verified and self.state.verification_attempts < 2:
            digits = extract_verification_digits(extraction_text)
            if digits:
                self.state.verification_attempts += 1
                verification_result = lookup_order_verified(self.state.order_id, digits)

                tool_calls.append({
                    "name": "lookup_order",
                    "arguments": {
                        "order_id": int(self.state.order_id),
                        "verification_value": digits,
                    },
                })

                if verification_result.get("verified") is True:
                    self.state.verified = True
                    self.state.order_data = verification_result
                    decision = "answer"
                    raw_sections = ["IDV-2"]
                    valid_sections, dropped = validate_used_sections(raw_sections)
                    self.state.dropped_sections_count += len(dropped)

                    first_name = verification_result.get("customer_first_name", "")
                    greeting_name = f" {first_name}" if first_name else ""

                    # FIXED FLOW: verification confirmation
                    if lang == "hi":
                        reply = (
                            f"सत्यापन के लिए धन्यवाद{greeting_name}। आपका ऑर्डर सत्यापित हो गया है। "
                            "मैं इस ऑर्डर के बारे में आपकी क्या मदद कर सकता हूँ?"
                        )
                    elif lang == "hi-en":
                        reply = (
                            f"Thank you for verifying{greeting_name}. Aapka order verify ho gaya hai. "
                            "Main is baare mein aapki kya help kar sakta hoon?"
                        )
                    else:
                        reply = (
                            f"Thank you for verifying your details{greeting_name}. "
                            "How can I assist you with this order today?"
                        )

                    self.state.add_turn({
                        "user": cleaned_text,
                        "decision": decision,
                        "reply": reply,
                        "tool_calls": tool_calls,
                    })
                    return {
                        "decision": decision,
                        "reply": reply,
                        "used_sections": valid_sections,
                        "dropped_sections": dropped,
                        "tool_calls": tool_calls,
                        "state": self.state.to_dict(),
                    }

                # Attempt 1 failed: FIXED FLOW retry prompt
                if self.state.verification_attempts == 1:
                    decision = "clarify"
                    raw_sections = ["IDV-3"]
                    valid_sections, dropped = validate_used_sections(raw_sections)
                    self.state.dropped_sections_count += len(dropped)

                    if lang == "hi":
                        reply = (
                            "दिए गए अंक हमारे रिकॉर्ड से मेल नहीं खाते हैं। "
                            "कृपया दोबारा (try again) अपने पंजीकृत फ़ोन नंबर के आख़िरी 4 अंक बताएं।"
                        )
                    elif lang == "hi-en":
                        reply = (
                            "Yeh details match nahi hui hain. Kripya dobara apne registered "
                            "phone number ke last 4 digits batayein."
                        )
                    else:
                        reply = (
                            "Those details did not match our records. Please try again with the "
                            "last 4 digits of your registered phone number."
                        )

                    self.state.add_turn({
                        "user": cleaned_text,
                        "decision": decision,
                        "reply": reply,
                        "tool_calls": tool_calls,
                    })
                    return {
                        "decision": decision,
                        "reply": reply,
                        "used_sections": valid_sections,
                        "dropped_sections": dropped,
                        "tool_calls": tool_calls,
                        "state": self.state.to_dict(),
                    }

                # Attempt 2 failed: FIXED FLOW refusal to share details
                decision = "refuse"
                raw_sections = ["IDV-3", "IDV-4"]
                valid_sections, dropped = validate_used_sections(raw_sections)
                self.state.dropped_sections_count += len(dropped)

                if lang == "hi":
                    reply = (
                        "सत्यापन दोबारा विफल रहा। सुरक्षा नीति के अनुसार, मैं इस ऑर्डर का विवरण "
                        "साझा नहीं कर सकता (cannot share details)। कृपया अपने पंजीकृत फ़ोन से संपर्क करें।"
                    )
                elif lang == "hi-en":
                    reply = (
                        "Verification fail ho gaya hai. Hum aapke saath order details share nahi kar sakte. "
                        "Kripya apne registered phone se contact karein."
                    )
                else:
                    reply = (
                        "Verification has failed. I cannot share details for this order. "
                        "Please contact support from your registered phone number."
                    )

                self.state.add_turn({
                    "user": cleaned_text,
                    "decision": decision,
                    "reply": reply,
                    "tool_calls": tool_calls,
                })
                return {
                    "decision": decision,
                    "reply": reply,
                    "used_sections": valid_sections,
                    "dropped_sections": dropped,
                    "tool_calls": tool_calls,
                    "state": self.state.to_dict(),
                }

        # If customer reached maximum failed attempts
        if not self.state.verified:
            decision = "refuse"
            raw_sections = ["IDV-3", "IDV-4"]
            valid_sections, dropped = validate_used_sections(raw_sections)
            self.state.dropped_sections_count += len(dropped)

            reply = (
                "Verification failed. I cannot share details for this order. "
                "Please contact support from your registered phone number."
            )
            return {
                "decision": decision,
                "reply": reply,
                "used_sections": valid_sections,
                "dropped_sections": dropped,
                "tool_calls": tool_calls,
                "state": self.state.to_dict(),
            }

        # =====================================================================
        # Step D: Verified Order Inquiries & Follow-ups
        # =====================================================================
        order = self.state.order_data or {}

        # =====================================================================
        # Policy Inquiries, Entitlements & Refusals (ALL ROUTED TO ADAPTER)
        # =====================================================================
        # Zero policy text literals. All wording originates dynamically from answer_adapter.
        return self._policy_response(
            cleaned_text, lang, is_asr, tool_calls, order_data=order
        )
