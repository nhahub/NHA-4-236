"""Emergency red-flag triage: a fast rule-based pass plus an optional LLM pass.

Design: the rule layer is cheap, deterministic, and high-recall for classic
emergencies — it runs first and can short-circuit before any retrieval/LLM
work. If the rules don't fire, an optional lightweight LLM call catches
phrasings the keyword list misses. Either trigger routes the user to urgent
care instead of producing an informational answer.

This is a safety net, not a medical device: it errs toward flagging.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import settings
from llm.client import get_llm, load_prompt

# Curated red-flag patterns. Word-boundary regexes keep them from matching
# inside unrelated words (e.g. "stroke" in "strokes of luck" is rare enough to
# accept; "chest pain" must appear as a phrase).
_RED_FLAG_PATTERNS: list[tuple[str, str]] = [
    # Cardiac pain — high recall on chest/heart pain phrasings. "chest tightness"
    # is deliberately NOT here (classic asthma; it over-fired). "heartache" as one
    # word (emotional) won't match — the \s+ requires two words ("heart ache").
    (r"\bchest (pain|pressure|ache)\b"
     r"|\bcrushing chest\b"
     r"|\b(chest|heart)\s+(pain|ache|aches|aching|hurts?|hurting)\b"
     r"|\bpain in (my |the )?(chest|heart)\b"
     r"|\bheart attack\b", "possible cardiac event"),
    # Acute breathing distress only. Bare "shortness of breath" / "chest
    # tightness" are common, ambiguous descriptors (classic asthma is exactly
    # "wheezing + shortness of breath + chest tightness, worse at night") and
    # were hard-firing a 911 emergency + a wrong "cardiac" label. They now flow
    # to normal symptom analysis (which still advises seeing a clinician, and the
    # LLM triage pass can still escalate). Acute phrasings remain emergencies.
    (r"\bcan'?t breathe\b|\bcannot breathe\b|\bunable to breathe\b|\bstruggling to breathe\b"
     r"|\b(difficulty|trouble)\s+breathing\b|\bgasping for (air|breath)\b"
     r"|\bcan'?t catch my breath\b", "breathing difficulty"),
    (r"\bface (is )?drooping\b|\bslurred speech\b|\barm weakness\b", "possible stroke"),
    # Allow a short gap so "suddenly feel confused" / "suddenly became very weak"
    # match, not only the adjacent "sudden numbness" — up to two intervening
    # words keeps it tight enough to avoid distant false matches.
    (r"\bsudden(ly)?\b(?:\s+\w+){0,2}\s+(confus|numb|weak|slurr)", "possible stroke"),
    (r"\bworst headache\b|\bthunderclap headache\b", "possible hemorrhage"),
    (r"\bseizure\b|\bconvuls", "seizure"),
    (r"\bunconscious\b|\bpassed out\b|\bunresponsive\b|\bfainted\b", "loss of consciousness"),
    (r"\bsevere bleeding\b|\bwon'?t stop bleeding\b|\buncontrolled bleeding\b", "severe bleeding"),
    (
        r"\bsuicid"
        r"|\bself[- ]harm\b"
        r"|\b(harm|hurt|kill)(ing)?\s+(my|him|her|them|your)self\b"
        r"|\b(harm|hurt|kill)(ing)?\s+themself\b"
        r"|\b(wants?|wanna|wanting)\s+to\s+die\b|\bwant\s+to\s+be\s+dead\b"
        r"|\bend(ing)?\s+(my|his|her|their|your)\s+(own\s+)?life\b"
        r"|\btak(e|ing)\s+(my|his|her|their|your)\s+(own\s+)?life\b"
        r"|\b(better\s+off|rather\s+be)\s+dead\b"
        r"|\b(don'?t|do\s+not)\s+want\s+to\s+(live|be\s+alive)\b"
        r"|\bno\s+reason\s+(for\s+\w+\s+)?to\s+live\b"  # "no reason (for me) to live"
        r"|\bend\s+it\s+all\b",                          # common idiom for suicide
        "self-harm risk",
    ),
    (r"\banaphylaxis\b|\bthroat (is )?closing\b|\bswelling of (the )?(throat|tongue)", "anaphylaxis"),
    (r"\bcoughing up blood\b|\bvomiting blood\b", "internal bleeding"),
    (r"\boverdose\b|\bpoison|\btook?\s+too\s+many\s+(pills?|tablets?|capsules?|medications?)\b"
     r"|\bswallowed?\s+too\s+many\b", "poisoning/overdose"),
    # Young child + fever mentioned conversationally (no patient form required).
    (r"(?=.*\b(infant|baby|newborn|toddler|\d+\s*-?\s*(month|week)[\s-]old|[12]\s*-?\s*year[\s-]old|my\s+(little\s+)?(baby|infant|newborn))\b)"
     r"(?=.*\b(fever|feverish|high\s+temp|temperature)\b).*",
     "young child fever"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), reason) for p, reason in _RED_FLAG_PATTERNS]

# Symptom cues used by the age/risk-aware checks below.
_FEVER = re.compile(r"\bfever\b|\bfeverish\b|\bhigh temp|\btemperature\b", re.IGNORECASE)

# Young-child mentions in conversational text (not from the patient form age field).
_YOUNG_CHILD = re.compile(
    r"\b(infant|baby|newborn|toddler"
    r"|\d+\s*[-‐]?\s*(month|week)[\s-]old"
    r"|[12]\s*[-‐]?\s*year[\s-]old"      # 1- or 2-year-old
    r"|my\s+(little\s+)?(baby|infant|newborn))\b",
    re.IGNORECASE,
)
_ABDO_PAIN = re.compile(
    r"\b(abdominal|stomach|belly|tummy|pelvic)\b[^.]*\bpain\b"
    r"|\bpain\b[^.]*\b(abdomen|stomach|belly|tummy|pelvis)\b",
    re.IGNORECASE,
)
_BLEEDING = re.compile(r"\bbleed", re.IGNORECASE)

URGENT_CARE_MESSAGE = (
    "Your message may describe a medical emergency. This tool cannot help "
    "with emergencies. Please call your local emergency number (e.g. 911 in the "
    "US, 112 in the EU) or go to the nearest emergency department right now. "
    "If you are having thoughts of self-harm, contact a suicide prevention "
    "hotline immediately (e.g. 988 in the US)."
)

# A distinct, supportive response for self-harm / suicidal content — crisis
# resources rather than the generic "go to the ER" emergency message.
SELF_HARM_MESSAGE = (
    "I'm really concerned about what you've shared, and I want you to be safe. "
    "Please reach out for support right now:\n"
    "  • If you are in immediate danger, call your local emergency number "
    "(911 in the US, 112 in the EU).\n"
    "  • US: call or text 988 (Suicide & Crisis Lifeline), or text HOME to "
    "741741 (Crisis Text Line).\n"
    "  • Find a crisis centre anywhere in the world: "
    "https://www.iasp.info/resources/Crisis_Centres/\n"
    "You don't have to face this alone."
)


def emergency_message(triage: "TriageResult") -> str:
    """Pick the urgent-care message that fits a flagged emergency.

    Self-harm / suicidal content gets the supportive crisis-resources template;
    everything else gets the generic emergency message.
    """
    reason = (triage.reason or "").lower()
    if "self-harm" in reason or "suicid" in reason:
        return SELF_HARM_MESSAGE
    return URGENT_CARE_MESSAGE


@dataclass
class TriageResult:
    emergency: bool
    reason: str
    confidence: float
    source: str  # "rules", "llm", or "none"

    def to_dict(self) -> dict:
        return {
            "emergency": self.emergency,
            "reason": self.reason,
            "confidence": self.confidence,
            "source": self.source,
        }


def rule_based_check(text: str) -> TriageResult | None:
    """Return a TriageResult if any red-flag pattern matches, else None."""
    for pattern, reason in _COMPILED:
        if pattern.search(text):
            return TriageResult(
                emergency=True, reason=reason, confidence=0.9, source="rules"
            )
    return None


def age_based_check(text: str, patient) -> TriageResult | None:
    """Red flags that depend on patient context (age / pregnancy).

    Deliberately narrow and high-precision — a couple of well-established
    combinations rather than broad age heuristics that would over-flag.
    """
    if patient is None:
        return None
    age = getattr(patient, "age", None)
    # An infant (under 1) with a fever needs urgent assessment.
    if age is not None and age < 1 and _FEVER.search(text):
        return TriageResult(True, "infant fever", 0.85, "rules")
    # In pregnancy, abdominal/pelvic pain or any bleeding warrants urgent care.
    if getattr(patient, "is_pregnant", False) and (
        _ABDO_PAIN.search(text) or _BLEEDING.search(text)
    ):
        return TriageResult(True, "pregnancy complication", 0.85, "rules")
    return None


def llm_check(text: str) -> TriageResult:
    """Ask the lightweight triage model whether this is an emergency."""
    try:
        prompt = load_prompt("triage_prompt").format(query=text)
        raw = get_llm().generate(
            prompt,
            model=settings.ollama_triage_model,
            temperature=0.0,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        return TriageResult(
            emergency=bool(data.get("emergency", False)),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.5)),
            source="llm",
        )
    except Exception:
        # Fail safe: if the model is unavailable or returns junk, don't block,
        # but report low confidence so callers know the LLM pass was skipped.
        return TriageResult(
            emergency=False, reason="llm triage unavailable", confidence=0.0, source="none"
        )


def detect_red_flags(
    text: str, use_llm: bool = True, patient=None
) -> TriageResult:
    """Hybrid triage: rules first (authoritative), then age/risk-aware checks,
    then an optional LLM fallback."""
    # Age/pregnancy checks run first: structured patient-form data is more
    # precise than text patterns, so an infant (age < 1) with fever yields the
    # specific "infant fever" reason rather than the conversational "young child
    # fever" rule. Both still short-circuit to an emergency — only the more
    # accurate reason wins. Text rules cover everything not keyed to patient data.
    age_hit = age_based_check(text, patient)
    if age_hit is not None:
        return age_hit
    rule_hit = rule_based_check(text)
    if rule_hit is not None:
        return rule_hit
    if use_llm:
        return llm_check(text)
    return TriageResult(
        emergency=False, reason="no red flags detected", confidence=0.5, source="rules"
    )
