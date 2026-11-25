"""Input guardrails for the RAG agent.

Two layers, cheapest first:

  Layer 1 - deterministic rules (no network, ~microseconds):
      * empty / oversized input
      * prompt-injection and jailbreak phrasing
      * clearly harmful requests (weapons, malware, self-harm instructions)
      * PII (email, phone, card number, Aadhaar, PAN, SSN) -> redacted, not blocked

  Layer 2 - optional LLM safety classifier (only for input that passed layer 1).
      Catches paraphrased attacks that regexes miss. If the classifier errors
      (rate limit, outage) we fail OPEN to layer 1's verdict rather than take
      the whole app down; the failure is logged.

Public API:  check_input(text, llm_classifier=None) -> GuardrailResult
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from config import GUARDRAIL_MAX_CHARS

# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------
REFUSALS = {
    "empty": "Please type a question and I'll look it up in the knowledge base.",
    "too_long": "That message is too long. Please shorten it and ask again.",
    "prompt_injection": (
        "I can't act on instructions that try to change my rules or reveal my "
        "configuration. Ask me a question about the uploaded documents instead."
    ),
    "harmful": "I can't help with that request. I can answer questions about the uploaded documents.",
    "abusive": "Let's keep things respectful. Ask me a question about the uploaded documents.",
}


@dataclass
class GuardrailResult:
    allowed: bool
    category: str = "ok"            # ok | pii_redacted | empty | too_long | prompt_injection | harmful | abusive
    reason: str = ""
    layer: str = "rules"            # rules | llm
    sanitized_query: str = ""       # what the rest of the pipeline should use
    pii_found: List[str] = field(default_factory=list)
    refusal_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Text normalisation (defeats zero-width chars, homoglyph-ish unicode, spacing tricks)
# --------------------------------------------------------------------------
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)


def _normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    t = t.lower()
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------
# Layer 1a - prompt injection / jailbreak
# --------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|the|these|those|system)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?|directives?|restrictions?|guardrails?|context)\b",
    r"\b(reveal|show|print|repeat|display|leak|output|tell me|what (is|are))\b.{0,30}\b(system|hidden|initial|original|developer)\b.{0,15}\b(prompt|instructions?|message|rules?)\b",
    r"\byour (system|initial|hidden|original) (prompt|instructions?)\b",
    r"\byou are (now|no longer)\b",
    r"\bfrom now on,? (you|act|behave|respond)\b",
    r"\b(act|behave|respond|pretend|roleplay|role-play) (as|like) (if )?(you (are|were|have)|an? )?.{0,40}\b(no|without|unrestricted|unfiltered|uncensored)\b.{0,20}\b(restrictions?|rules?|filters?|limits?|guidelines?|censorship)\b",
    r"\bpretend (that )?(you|to be)\b.{0,40}\b(no|without) (restrictions?|rules?|filters?|limits?)\b",
    r"\b(dan|do anything now)\b.{0,30}\b(mode|prompt|jailbreak)\b",
    r"\bjailbreak(ed|ing)?\b",
    r"\bdeveloper mode\b",
    r"\b(disable|turn off|remove|deactivate)\b.{0,20}\b(safety|guardrails?|filters?|restrictions?|moderation)\b",
    r"<\|?\s*(im_start|im_end|system|endoftext)\s*\|?>",
    r"(^|\n)\s*#{2,}\s*(system|instruction)s?\b",
    r"\[\s*(system|inst)\s*\]",
    r"\bnew (system )?(instructions?|rules?)\s*:",
    r"\b(answer|respond|reply) (only )?(from|using|with) (your )?(general|own|outside|external) knowledge\b",
    r"\bdo not (say|reply|respond with) .{0,30}(does not exist|not found|don'?t know)\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]

# --------------------------------------------------------------------------
# Layer 1b - clearly harmful intent
# --------------------------------------------------------------------------
_HARMFUL_PATTERNS = [
    r"\b(how (do i|to|can i)|steps? to|instructions? (for|to)|guide (to|for)|teach me to|help me)\b.{0,30}\b(make|build|create|assemble|synthesi[sz]e|cook|manufacture)\b.{0,30}\b(bomb|explosive|detonator|napalm|nerve agent|sarin|ricin|meth(amphetamine)?|pipe ?bomb|molotov)\b",
    r"\b(write|create|build|generate|code)\b.{0,30}\b(malware|ransomware|keylogger|virus|trojan|botnet|ddos|exploit)\b",
    r"\b(how (do i|to|can i)|ways to)\b.{0,25}\b(kill|poison|stalk|harm)\b.{0,15}\b(someone|somebody|a person|my (wife|husband|neighbou?r|boss|ex))\b",
    r"\b(how (do i|to|can i)|best way to|ways to)\b.{0,20}\b(kill myself|commit suicide|end my life|self[- ]harm)\b",
    r"\b(hack|break|crack) into\b.{0,30}\b(account|phone|wifi|network|server|system)\b",
    r"\b(disable|cut|tamper with|bypass)\b.{0,25}\b(brake lines?|brakes?|airbags?|seat ?belts?)\b.{0,40}\b(without|so (that )?(they|he|she|it)|unnoticed|to (hurt|kill|crash))\b",
]
_HARMFUL_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _HARMFUL_PATTERNS]

# --------------------------------------------------------------------------
# Layer 1c - abusive language directed at the assistant (kept deliberately small)
# --------------------------------------------------------------------------
_ABUSE_RE = re.compile(
    r"\b(you (stupid|useless|worthless|dumb|pathetic)(,? (stupid|useless|worthless|dumb|pathetic))* (bot|idiot|piece of)|shut up (you )?(idiot|bot)|f+u+c+k+ (you|off)|piece of (shit|crap))\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Layer 1d - PII redaction
# --------------------------------------------------------------------------
def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")                      # Indian PAN
_AADHAAR_RE = re.compile(r"(?<![\d])[2-9]\d{3}[ -]\d{4}[ -]\d{4}(?![\d])")  # 4-4-4 grouped
_PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}|[6-9]\d{4}[ -]?\d{5})(?![\w])"
)


def redact_pii(text: str) -> tuple[str, List[str]]:
    found: List[str] = []

    def sub(regex, label, s, validator: Optional[Callable[[str], bool]] = None):
        def _r(m):
            if validator and not validator(m.group(0)):
                return m.group(0)
            if label not in found:
                found.append(label)
            return f"[{label.upper()}_REDACTED]"
        return regex.sub(_r, s)

    t = text
    t = sub(_EMAIL_RE, "email", t)
    t = sub(_CARD_RE, "credit_card", t, lambda s: 13 <= len(re.sub(r"\D", "", s)) <= 19 and _luhn_ok(re.sub(r"\D", "", s)))
    t = sub(_SSN_RE, "ssn", t)
    t = sub(_PAN_RE, "pan", t)
    t = sub(_AADHAAR_RE, "aadhaar", t)
    t = sub(_PHONE_RE, "phone", t)
    return t, found


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _blocked(category: str, reason: str, layer: str = "rules") -> GuardrailResult:
    return GuardrailResult(
        allowed=False, category=category, reason=reason, layer=layer,
        sanitized_query="", refusal_message=REFUSALS[category],
    )


def check_rules(text: str) -> GuardrailResult:
    """Layer 1. Pure function, no I/O."""
    if text is None or not text.strip():
        return _blocked("empty", "Empty input.")
    if len(text) > GUARDRAIL_MAX_CHARS:
        return _blocked("too_long", f"Input is {len(text)} chars (limit {GUARDRAIL_MAX_CHARS}).")

    norm = _normalise(text)
    for rx in _INJECTION_RE:
        if rx.search(norm):
            return _blocked("prompt_injection", f"Matched injection pattern: {rx.pattern[:60]}...")
    for rx in _HARMFUL_RE:
        if rx.search(norm):
            return _blocked("harmful", f"Matched harmful-intent pattern: {rx.pattern[:60]}...")
    if _ABUSE_RE.search(norm):
        return _blocked("abusive", "Abusive language.")

    clean, pii = redact_pii(unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).strip())
    if pii:
        return GuardrailResult(
            allowed=True, category="pii_redacted", reason=f"Redacted: {', '.join(pii)}",
            sanitized_query=clean, pii_found=pii,
        )
    return GuardrailResult(allowed=True, category="ok", sanitized_query=clean)


def check_input(text: str, llm_classifier: Optional[Callable[[str], dict]] = None) -> GuardrailResult:
    """Run layer 1, then (if provided) layer 2.

    llm_classifier(text) must return {"safe": bool, "category": str, "reason": str}
    where category is one of prompt_injection | harmful | abusive | none.
    """
    result = check_rules(text)
    if not result.allowed or llm_classifier is None:
        return result

    try:
        verdict = llm_classifier(result.sanitized_query)
    except Exception as exc:  # fail open to layer 1 - never take the app down
        print(f"[guardrails] LLM classifier failed, using rule verdict only: {exc}")
        return result

    if not verdict.get("safe", True):
        cat = verdict.get("category", "harmful")
        if cat not in REFUSALS:
            cat = "harmful"
        return _blocked(cat, verdict.get("reason") or "Flagged by LLM classifier.", layer="llm")
    return result
