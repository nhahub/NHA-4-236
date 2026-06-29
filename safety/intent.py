"""Lightweight intent routing — run before retrieval.

The UI mode toggle (General vs. Symptom) is only a *hint*. This classifier
looks at what the user actually typed and can override it, so a definition
question asked in Symptom mode gets a plain answer instead of a forced
differential, and a greeting/meta message ("hi", "can I ask something else?")
gets a friendly redirect instead of running the full RAG pipeline over noise.

Design choice: we deliberately do NOT ship a big medical-keyword list to detect
off-topic questions (that approach was a bug magnet in a sibling project —
normal words got misread as medical terms). Instead, only high-precision
greeting/meta patterns short-circuit here; genuinely off-topic but
question-shaped input ("what is the capital of egypt") is allowed through to
retrieval, where the rerank confidence gate declines it. Two cheap, robust
checks beat one brittle vocabulary.
"""
from __future__ import annotations

import random
import re

# Routing outcomes.
CHITCHAT = "chitchat"
QA = "qa"
SYMPTOM = "symptom"

# A message that is (or starts as) a greeting / pleasantry / meta-question.
_GREETING = re.compile(
    r"^\s*(hi+|hey+|hello+|yo|hiya|sup|greetings|good\s+(morning|afternoon|evening|day)"
    r"|thanks?|thank\s+you|thx|ok(ay)?|cool|great|nice|bye|goodbye|see\s+ya)\b",
    re.IGNORECASE,
)
_META = re.compile(
    r"\b(can\s+i\s+ask|ask\s+(you\s+)?something|something\s+else|another\s+question"
    r"|what\s+can\s+you\s+(do|help)|how\s+do\s+you\s+work|just\s+testing)\b",
    re.IGNORECASE,
)

# "Who/what are you" style identity questions — answered with a clear statement
# that this is software, not a human or a doctor.
_IDENTITY = re.compile(
    r"\b(who\s+are\s+you|what\s+are\s+you"
    r"|are\s+you\s+(a\s+|an\s+|the\s+)?(real\s+)?"
    r"(human|person|real|alive|sentient|conscious|machine|robot|bot|chat\s?bot"
    r"|ai|gpt|llm|doctor|physician|nurse))\b",
    re.IGNORECASE,
)

_IDENTITY_REPLY = (
    "I'm a medical information assistant — software, not a human or a doctor, "
    "and I can't diagnose. I can help you understand symptoms, conditions, "
    "medications, and other health topics. What can I help you with?"
)

# Small talk that *looks* like a question ("what's up") but isn't a real ask —
# checked before the definition pattern so it doesn't get mistaken for one.
_SMALLTALK = re.compile(
    r"^\s*(what'?s\s+up|sup|how\s+are\s+you|how'?s\s+it\s+going|how\s+do\s+you\s+do)\b",
    re.IGNORECASE,
)

# Treatment / management follow-up — "what's the treatment?", "and the cure?",
# "how do I treat this?" — should get a plain Q&A answer even in Symptom mode.
_TREATMENT = re.compile(
    r"\b(treat(ment|ing|ed)?|therap(y|ies|eutic)|manag(e|ing|ement)"
    r"|cure|remedies?|medication|medicine|prescription"
    r"|how\s+(to\s+)?(treat|manage|cure|fix)\b"
    r"|what\s+.{0,20}\s+(treat|therap|manag|cure)"
    r"|what\s+causes?\s+(it|this|them|these)\b"
    r"|why\s+(do|does|did|is|are|would)\b"
    r"|how\s+(long|serious|dangerous|contagious|common)\b"
    # Medication safety questions: "is ibuprofen safe?", "can I take X?", "should I use X?"
    r"|\bis\s+\w+\s+(safe|okay|ok|fine|recommended|dangerous|harmful)\b"
    r"|\bcan\s+i\s+(take|use|have|drink|eat)\b"
    r"|\bshould\s+i\s+(take|use|avoid|stop|continue)\b"
    # Dosage / drug information questions
    r"|\b(dosage|dose|doses|dosing)\b"
    r"|\bside\s+effect"
    # Severity / worry follow-ups: "is this serious?", "should I be worried?"
    r"|\bis\s+(this|it)\s+(serious|dangerous|urgent|bad|normal|common|contagious)\b"
    r"|\bshould\s+i\s+(be\s+)?(worried|concerned|scared|alarmed)\b"
    r"|\bhow\s+(serious|dangerous|urgent|bad|worried)\b"
    # Bare follow-up context questions
    r"|\btell\s+me\s+more\b|\bmore\s+detail\b|\belaborate\b"
    r"|\bwhat\s+about\s+(children|kids|elderly|pregnant|adults|infants|babies)\b"
    r"|\band\s+if\s+i'?m?\b|\bif\s+i'?m?\s+(pregnant|diabetic|allergic)\b"
    r"|\breally\??\s*$|\bsay\s+more\b)",
    re.IGNORECASE,
)

# A definition / explainer question — should get a plain Q&A answer regardless
# of the mode toggle. Matches "what is X", "what are X", "what's X", "define X",
# "tell me about X", "explain X", "what does X mean".
_DEFINITION = re.compile(
    r"\b(what\s+(is|are)\b|what'?s\b|what\s+does\b.*\bmean\b|define\b"
    r"|definition\s+of\b|tell\s+me\s+about\b|explain\b)",
    re.IGNORECASE,
)

# First-person framing + a symptom cue => the user is reporting symptoms, which
# warrants the ranked-differential flow even if the toggle says General.
_FIRST_PERSON = re.compile(r"\b(i|i'?m|i'?ve|i\s+have|my|me)\b", re.IGNORECASE)
_SYMPTOM_CUES = re.compile(
    r"\b(pain|ache|aches|aching|hurts?|hurting|sore|fever|cough|nausea|nause"
    r"|dizzy|dizziness|vomit|throw\s+up|rash|swollen|swelling|bleed|bleeding"
    r"|cramps?|diarr?h?ea|constipat|chills?|congest|sneez|short\s+of\s+breath"
    r"|fatigue|exhaust|sick|nauseous|burning|itch)\w*",
    re.IGNORECASE,
)

_PUNCT_ONLY = re.compile(r"^[\s\W_]*$")


def _is_gibberish(text: str) -> bool:
    """Return True if the text looks like random keysmashing / non-language.

    Heuristic: if over half the word-length is in a single run of the same
    character (no spacing), or if fewer than 20 % of characters are standard
    ASCII letters/digits, treat it as noise rather than a real question.
    Only fires on very short inputs (< 40 chars) to stay conservative.
    """
    stripped = text.strip()
    if len(stripped) > 60 or " " in stripped:
        # Multi-word inputs or longer text — let the retrieval gate decide.
        return False
    letters = sum(1 for c in stripped if c.isalpha())
    if len(stripped) == 0:
        return True
    if letters / len(stripped) < 0.5:
        # Mostly punctuation/digits/symbols (e.g. "!@#$%", "12345").
        return True
    # Check for a long single-char run (e.g. "aaaaaaaaa", "kjhgfds").
    if len(stripped) >= 6:
        max_run = max(
            sum(1 for _ in g) for _, g in __import__("itertools").groupby(stripped)
        )
        if max_run / len(stripped) > 0.5:
            return True
    return False

_CHITCHAT_REPLIES = (
    "Hi! I'm a medical information assistant. Ask me about a symptom, "
    "condition, medication, or other health topic and I'll do my best to help.",
    "Happy to help! I focus on health and medical questions — tell me about a "
    "symptom or condition you're curious about and I'll explain what the "
    "literature says.",
    "Hello! I can answer general medical questions and help you explore "
    "symptoms. What health topic can I help you with?",
    "I'm here for health and medical questions. Describe a symptom or ask "
    "about a condition, and I'll walk you through what the medical sources say.",
)


def is_greeting_or_meta(query: str) -> bool:
    return bool(
        _GREETING.match(query) or _META.search(query) or _IDENTITY.search(query)
    )


def is_definition(query: str) -> bool:
    return bool(_DEFINITION.search(query))


def is_symptom_report(query: str) -> bool:
    return bool(_FIRST_PERSON.search(query) and _SYMPTOM_CUES.search(query))


def classify_intent(query: str, mode_hint: str) -> str:
    """Return CHITCHAT, QA, or SYMPTOM for ``query``.

    ``mode_hint`` (QA or SYMPTOM, from the UI toggle) is the fallback when the
    text gives no strong signal of its own.
    """
    q = query.strip()
    if not q or _PUNCT_ONLY.match(q) or _is_gibberish(q):
        return CHITCHAT
    # Question-shaped small talk ("what's up") is chit-chat, not a definition.
    if _SMALLTALK.match(q):
        return CHITCHAT
    # Identity questions ("what are you", "are you a doctor") are chit-chat —
    # check before the definition pattern, which would catch "what are you".
    if _IDENTITY.search(q):
        return CHITCHAT
    # Definition or treatment/management questions get a plain Q&A answer even
    # in Symptom mode — they're follow-up clarifications, not new differentials.
    if is_definition(q) or _TREATMENT.search(q):
        return QA
    # First-person symptom reports get the differential even in General mode.
    if is_symptom_report(q):
        return SYMPTOM
    # Pleasantries / meta with no real medical ask -> friendly redirect.
    if is_greeting_or_meta(q):
        return CHITCHAT
    # No strong signal: honour the UI toggle.
    return mode_hint


def chitchat_reply(query: str = "") -> str:
    """A redirect for off-scope chit-chat (no LLM involved).

    Identity questions ("are you a doctor?") get a clear, fixed answer; other
    pleasantries get a varied warm redirect.
    """
    if query and _IDENTITY.search(query):
        return _IDENTITY_REPLY
    return random.choice(_CHITCHAT_REPLIES)
