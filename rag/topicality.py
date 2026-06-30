"""Cheap, deterministic "is this query medical?" check for the grounding gate.

When retrieval grounding is weak, the assistant must decline — but it should
decline *differently* for an off-topic query (``"capital of Egypt"``) than for a
genuine health question the corpus just doesn't cover (``"IVF success rates"``).
The first deserves a "I only do medical topics" redirect; the second deserves an
honest "that's a health question, but it's not in my sources".

A single cross-encoder grounding score can't tell those apart (both score low),
so this module supplies the orthogonal signal: a lexical medical-topic detector.
It is intentionally a heuristic — broad recall for "looks medical" — not a
classifier. Morphology (medical suffixes like *-itis*, *-emia*) plus a curated
term list cover the long tail without a model.
"""
from __future__ import annotations

import re

# Greek/Latin medical morphology. A token longer than the suffix that ends with
# one of these is almost always clinical (arthritis, leukemia, nephropathy...).
_MEDICAL_SUFFIXES = (
    "itis", "emia", "aemia", "osis", "oma", "opathy", "ectomy", "ostomy",
    "otomy", "plasty", "algia", "plasia", "megaly", "penia", "rrhea", "rrhoea",
    "uria", "pnea", "phagia", "trophy", "sclerosis", "stenosis", "cytosis",
)

# Curated common health vocabulary (lowercase, singular-ish). Not exhaustive —
# the suffix rule catches most disease names; this catches everyday health words
# and procedures/tests that have no tell-tale morphology.
_MEDICAL_TERMS = frozenset({
    # general
    "medical", "medicine", "health", "healthcare", "clinical", "clinic",
    "hospital", "doctor", "physician", "nurse", "patient", "diagnosis",
    "diagnose", "diagnostic", "prognosis", "treatment", "treat", "therapy",
    "therapeutic", "cure", "remedy", "screening", "biopsy", "surgery",
    "surgical", "operation", "procedure", "referral", "specialist",
    # states / descriptors
    "disease", "illness", "infection", "infectious", "condition", "disorder",
    "syndrome", "symptom", "symptoms", "chronic", "acute", "benign", "malignant",
    "inflammation", "inflamed", "swelling", "injury", "wound", "fracture",
    "bruise", "lesion", "ulcer",
    # symptoms
    "pain", "ache", "fever", "cough", "nausea", "vomiting", "vomit", "diarrhea",
    "diarrhoea", "rash", "itch", "fatigue", "dizziness", "dizzy", "bleeding",
    "cramp", "cramps", "numbness", "shortness", "breath", "wheeze", "wheezing",
    "chills", "sweats", "soreness",
    # body / systems
    "blood", "heart", "cardiac", "cardiovascular", "lung", "lungs", "pulmonary",
    "kidney", "kidneys", "renal", "liver", "hepatic", "brain", "neurological",
    "stomach", "gastric", "intestine", "bowel", "bladder", "bone", "joint",
    "muscle", "skin", "nerve", "artery", "vein", "thyroid", "pancreas",
    # common conditions
    "cancer", "tumor", "tumour", "diabetes", "diabetic", "insulin",
    "hypertension", "asthma", "allergy", "allergic", "stroke", "seizure",
    "epilepsy", "anemia", "anaemia", "migraine", "arthritis", "depression",
    "anxiety", "obesity", "pneumonia", "influenza", "flu", "covid", "hiv",
    "cholesterol", "glucose", "hormone",
    # reproductive / fertility (the IVF gap)
    "pregnancy", "pregnant", "fertility", "fertilization", "fertilisation",
    "ivf", "conceive", "contraception", "contraceptive", "menstrual",
    "menopause", "ovulation", "miscarriage",
    # meds / immunization
    "medication", "drug", "drugs", "dose", "dosage", "antibiotic",
    "antibiotics", "vaccine", "vaccination", "vaccinated", "immunization",
    "immunisation", "prescription", "steroid", "painkiller", "nsaid",
    # tests / imaging
    "mri", "ecg", "ekg", "eeg", "xray", "ultrasound", "scan", "bloodwork",
    "cholesterol", "biomarker",
})

_TOKEN_RE = re.compile(r"[a-z]+")


def looks_medical(text: str) -> bool:
    """True if the query plausibly concerns a health/medical topic.

    High-recall heuristic: a single medical term or a word with medical
    morphology is enough. Designed to separate "medical-but-uncovered" from
    "off-topic" at the grounding gate, not to be a precise classifier.
    """
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _MEDICAL_TERMS:
            return True
        if len(tok) > 5 and tok.endswith(_MEDICAL_SUFFIXES):
            return True
    return False
