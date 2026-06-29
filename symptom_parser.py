"""Parse free-text symptom descriptions into DDXPlus feature vectors.

Pipeline:
  1. scispaCy NER (en_core_sci_md) extracts medical entities from user text.
  2. Each entity is matched to a DDXPlus evidence code via:
       a. Exact string match against evidence question text
       b. Lowercase-normalised match
       c. Fuzzy match (difflib, threshold 0.80)
  3. Demographic hints (age / sex) are extracted via regex and encoded.
  4. If fewer than 3 symptoms matched, returns {"features": None} to trigger
     pure LLM+RAG fallback — ML predictions are omitted when signal is too weak.

Public API:
    parse(text: str) -> dict
        {
          "features":         np.ndarray | None,
          "matched_count":    int,
          "matched_symptoms": list[str],    # DDXPlus evidence codes matched
          "unmatched_terms":  list[str],    # NER entities with no code mapping
        }
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from ml_model.features import encode_patient
from ml_model.predict import artifacts_available, feature_columns

_DATA_DIR = Path(__file__).parent / "data" / "raw" / "ddxplus"
_EVIDENCES_PATH = _DATA_DIR / "release_evidences.json"
_HF_REPO = "aai530-group6/ddxplus"


def _ensure_evidences_file() -> None:
    """Download release_evidences.json from HuggingFace if not present locally."""
    if _EVIDENCES_PATH.exists():
        return
    try:
        import shutil
        from huggingface_hub import hf_hub_download
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        src = hf_hub_download(repo_id=_HF_REPO, filename="release_evidences.json", repo_type="dataset")
        shutil.copy(src, _EVIDENCES_PATH)
    except Exception:
        pass  # if download fails, _load_evidences returns {} and ML is skipped

_MIN_SYMPTOM_MATCH = 3
_FUZZY_THRESHOLD = 0.80

# DDXPlus questions mix present-symptom questions ("Do you have a cough?") with
# family-history / past-medical-history / treatment-history questions ("Have any
# of your family members been diagnosed with cluster headaches?"). A lay term
# like "headache" or "nausea" substring-matches the history question, which then
# gets fed to the classifier as if it were a present symptom — actively wrong.
# Any question containing one of these markers is demoted: it is only ever used
# when no present-symptom question matches (and for most lay terms, never).
_META_MARKERS = (
    "family", "members", "relative", "diagnosed", "treated in hospital",
    "ever had", "ever been", "contact with", "recently for", "in the past",
)

# Lay / colloquial term -> a keyword that actually appears in DDXPlus question
# text. Without this, "nausea" only matches the IV-treatment-history question
# (E_147) instead of the symptom (E_148, phrased "nauseous"), and "dizziness"
# matches nothing (questions say "dizzy"). Keys are matched against normalised
# NER entities; values are substrings searched against the question text.
_SYNONYMS = {
    "shortness of breath": "shortness of breath",
    "short of breath": "shortness of breath",
    "sob": "shortness of breath",
    "breathless": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "trouble breathing": "shortness of breath",
    "dizziness": "dizzy",
    "vertigo": "dizzy",
    "light headed": "lightheaded",
    "nausea": "nauseous",
    "nauseated": "nauseous",
    "queasy": "nauseous",
    "throwing up": "vomiting",
    "throw up": "vomiting",
    "puking": "vomiting",
    "vomit": "vomiting",
    "tired": "fatigued",
    "tiredness": "fatigued",
    "exhausted": "fatigued",
    "exhaustion": "fatigued",
    "fatigue": "fatigued",
    "stuffy nose": "nasal congestion",
    "blocked nose": "nasal congestion",
    "congestion": "nasal congestion",
    "racing heart": "palpitations",
    "heart racing": "palpitations",
    "high temperature": "fever",
    "shivering": "chills",
    "shivers": "chills",
    "itchy": "itching",
    "itch": "itching",
    "swollen": "swelling",
    "sweaty": "sweating",
    "diarrhoea": "diarrhea",
    "loose stools": "diarrhea",
}

# Scaffolding words stripped before fuzzy token comparison so the ratio reflects
# the symptom content, not the shared question boilerplate.
# Noun chunks / NER spans that are never symptoms — pronouns, time words, and
# generic nouns. Without this guard they substring-match question text ("days"
# inside a duration question, "i" inside "pain") and inject spurious evidence.
_IGNORE_TERMS = frozenset({
    "i", "you", "it", "we", "they", "he", "she", "me", "him", "her", "them",
    "this", "that", "thing", "something", "anything", "everything", "nothing",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "time", "times", "while", "morning", "night", "evening", "today",
    "reason", "problem", "issue", "lot", "bit", "couple", "few",
})

_QUESTION_WORDS = frozenset({
    "do", "you", "have", "a", "an", "the", "is", "are", "your", "feel", "or",
    "of", "to", "in", "on", "with", "and", "does", "did", "how", "what",
    "where", "when", "been", "ever", "had", "has", "that", "this", "at", "it",
    "for", "i", "my", "me", "somewhere", "related", "more", "than", "usual",
    "any", "been", "recently", "significantly", "either", "felt", "measured",
})

_AGE_RE = re.compile(r"\b(\d{1,3})\s*(?:year[s]?\s*old|yo|y\.?o\.?|years?)\b", re.I)
_SEX_RE = re.compile(r"\b(male|female|man|woman|boy|girl|he|she)\b", re.I)

_SEX_MAP = {
    "male": "M", "man": "M", "boy": "M", "he": "M",
    "female": "F", "woman": "F", "girl": "F", "she": "F",
}


@lru_cache(maxsize=1)
def _load_evidences() -> dict:
    _ensure_evidences_file()
    if not _EVIDENCES_PATH.exists():
        return {}
    with open(_EVIDENCES_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_lookup() -> dict[str, str]:
    """Return {normalised_question_text: evidence_code} for fuzzy/exact matching."""
    evidences = _load_evidences()
    lookup = {}
    for code, meta in evidences.items():
        question = meta.get("question_en", "")
        if question:
            lookup[question.lower().strip()] = code
    return lookup


def _is_meta(question: str) -> bool:
    """True for family-/medical-/treatment-history questions (not present symptoms)."""
    q = question.lower()
    return any(marker in q for marker in _META_MARKERS)


# Attribute/characterization questions presuppose the symptom rather than
# establishing it ("Where is the swelling located?", "How severe is the
# itching?", "Is the rash swollen?"). Prefer a presence question ("Do you have
# swelling...?") when one exists for the same term.
_ATTRIBUTE_RE = re.compile(
    r"^\s*(how|what|where|which|characterize|is the|are the|does the)\b", re.I
)


def _is_attribute(question: str) -> bool:
    return bool(_ATTRIBUTE_RE.match(question))


def _content_tokens(text: str) -> list[str]:
    """Tokenise to lowercase words, dropping question scaffolding/stop words."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in _QUESTION_WORDS]


def _token_overlap(search: str, question: str) -> float:
    """Coverage of the search term's content tokens within the question.

    For each search token, take its best fuzzy ratio against any question token
    and average. ~1.0 when every search word closely matches a question word,
    so single-word misspellings ("diarhea" vs "diarrhea") still clear the bar.
    """
    s_tokens = _content_tokens(search)
    q_tokens = _content_tokens(question)
    if not s_tokens or not q_tokens:
        return 0.0
    total = 0.0
    for st in s_tokens:
        total += max(SequenceMatcher(None, st, qt).ratio() for qt in q_tokens)
    return total / len(s_tokens)


def _match_entity(entity: str, lookup: dict[str, str]) -> str | None:
    """Return the best-matching DDXPlus evidence code for an NER entity, or None.

    Ordered strategy:
      1. Exact question-text match.
      2. Synonym expansion (lay term -> clinical keyword) then exact match.
      3. Substring containment, preferring the *shortest* (most direct)
         present-symptom question and skipping history/family/meta questions.
      4. Fuzzy token-overlap fallback over non-meta questions, best score wins.
    """
    normalised = entity.lower().strip()

    # Drop spans whose every alphabetic token is a pronoun / time word / generic
    # noun ("5 days" -> {"days"}, "it" -> {"it"}). Without this they fuzzy- or
    # substring-match unrelated question text and inject spurious evidence.
    tokens = re.findall(r"[a-z]+", normalised)
    if not tokens or all(t in _IGNORE_TERMS for t in tokens) or len(normalised) < 3:
        return None

    # 1. Exact match against the question text.
    if normalised in lookup:
        return lookup[normalised]

    # 2. Map a colloquial term to a clinical keyword; exact-match that if present.
    search = _SYNONYMS.get(normalised, normalised)
    if search in lookup:
        return lookup[search]

    # 3. Containment: among non-meta questions that contain the search phrase as
    #    a whole word (so "i" does not match inside "pain"), prefer a presence
    #    question over a characterization one, then the shortest (most direct) —
    #    "Do you have swelling...?" over "Where is the swelling located?", "Do
    #    you have a cough?" over "...with coughing...". History questions are
    #    excluded entirely: a wrong symptom flag is worse than none.
    pattern = re.compile(rf"\b{re.escape(search)}\b")
    contained = [
        (q, c) for q, c in lookup.items() if pattern.search(q) and not _is_meta(q)
    ]
    if contained:
        presence = [qc for qc in contained if not _is_attribute(qc[0])]
        pool = presence or contained
        question, code = min(pool, key=lambda qc: len(qc[0]))
        return code

    # 4. Fuzzy fallback (handles misspellings / minor variants), non-meta only,
    #    keeping the single best-scoring candidate above the threshold.
    best_score = 0.0
    best_code: str | None = None
    for question, code in lookup.items():
        if _is_meta(question):
            continue
        score = _token_overlap(search, question)
        if score > best_score:
            best_score = score
            best_code = code
    if best_score >= _FUZZY_THRESHOLD:
        return best_code

    return None


def _extract_demographic(text: str) -> tuple[int | None, str | None]:
    """Extract age (int) and sex ('M'/'F') from free text via regex."""
    age = None
    sex = None

    age_match = _AGE_RE.search(text)
    if age_match:
        try:
            age = int(age_match.group(1))
        except ValueError:
            pass

    sex_match = _SEX_RE.search(text)
    if sex_match:
        sex = _SEX_MAP.get(sex_match.group(1).lower())

    return age, sex


def _ner_entities(text: str) -> list[str]:
    """Extract medical entity spans from text using scispaCy en_core_sci_md.

    Falls back to a whitespace-tokenised word list when scispaCy is not
    installed, so the rest of the pipeline still works (just less accurately).
    """
    try:
        nlp = _get_nlp()
        doc = nlp(text)

        spans: list[str] = []
        seen: set[str] = set()

        def _add(span: str) -> None:
            span = span.strip()
            key = span.lower()
            if span and key not in seen:
                seen.add(key)
                spans.append(span)

        # Named entities — rich when the scispaCy medical model is installed.
        for ent in doc.ents:
            _add(ent.text)
        # Noun chunks + their head noun. With only the general en_core_web_sm
        # model, doc.ents misses symptoms entirely ("cough"/"fever" are not named
        # entities), but noun chunks capture them. The chunk root drops leading
        # determiners/adjectives ("a persistent cough" -> "cough") so the code
        # matcher can resolve them. This is what keeps the ML path alive when the
        # medical model is absent — otherwise no symptoms are ever extracted.
        for chunk in doc.noun_chunks:
            _add(chunk.text)
            _add(chunk.root.text)
        return spans
    except Exception:
        # Minimal fallback: split on punctuation, yield multi-word chunks
        words = re.findall(r"[A-Za-z][a-z']+(?:\s+[A-Za-z][a-z']+){0,3}", text)
        return words


@lru_cache(maxsize=1)
def _get_nlp():
    import spacy  # noqa: PLC0415

    for model in ("en_core_sci_md", "en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(model)
        except OSError:
            continue
    raise OSError(
        "No spaCy model found. Run: python -m spacy download en_core_web_sm"
    )


def parse(text: str) -> dict:
    """Parse a free-text symptom description into a DDXPlus feature vector.

    Returns a dict with keys:
        features (np.ndarray | None): ready to pass to ml_model.predict.predict()
        matched_count (int): number of DDXPlus evidence codes mapped
        matched_symptoms (list[str]): the evidence codes that were matched
        unmatched_terms (list[str]): NER entities with no code mapping
    """
    if not artifacts_available():
        return {"features": None, "matched_count": 0, "matched_symptoms": [], "unmatched_terms": []}

    lookup = _build_lookup()
    entities = _ner_entities(text)

    matched_codes: list[str] = []
    unmatched: list[str] = []
    seen_codes: set[str] = set()

    for entity in entities:
        code = _match_entity(entity, lookup)
        if code and code not in seen_codes:
            matched_codes.append(code)
            seen_codes.add(code)
        elif not code:
            unmatched.append(entity)

    if len(matched_codes) < _MIN_SYMPTOM_MATCH:
        return {
            "features": None,
            "matched_count": len(matched_codes),
            "matched_symptoms": matched_codes,
            "unmatched_terms": unmatched,
        }

    age, sex = _extract_demographic(text)
    age = age if age is not None else 30   # neutral default when not stated
    sex = sex if sex is not None else "M"  # neutral default when not stated

    all_codes = feature_columns()
    code_to_idx = {c: i for i, c in enumerate(all_codes)}

    # Build a synthetic EVIDENCES string from matched codes
    evidences_str = str(matched_codes)
    features = encode_patient(evidences_str, age, sex, code_to_idx)

    return {
        "features": features,
        "matched_count": len(matched_codes),
        "matched_symptoms": matched_codes,
        "unmatched_terms": unmatched,
    }
