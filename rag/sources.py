"""Knowledge-base source loaders.

Each loader reads one dataset and maps its records onto the shared ``Passage``
schema (id, text, question, title, url, source, qtype). All loaders are
optional and fail soft: if a dataset isn't present (or its optional dependency
isn't installed), the loader warns and returns an empty list so ingestion can
proceed with whatever sources are available.

Add a dataset by writing a ``load_*`` function and registering it in
``LOADERS``. MedQuAD alone is enough for a working system; the others enrich it.
"""
from __future__ import annotations

import csv
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from rag.chunking import Passage, chunk_text, clean, strip_html


def _warn(msg: str) -> None:
    print(f"[sources] {msg}", file=sys.stderr)


# --- MedQuAD --------------------------------------------------------------
def load_medquad(raw_dir: Path) -> list[Passage]:
    """Parse the MedQuAD XML dump (folders of <Document>/<QAPair> files).

    The MedlinePlus-derived subsets (ADAM, Drugs, Herbs) ship with empty
    answers (copyright) and are skipped automatically.
    """
    base = raw_dir / "MedQuAD"
    search_root = base if base.exists() else raw_dir
    xml_files = sorted(search_root.rglob("*.xml"))
    # Exclude MedlinePlus topic dumps, which have a different schema.
    xml_files = [f for f in xml_files if not f.name.startswith("mplus_topics")]
    if not xml_files:
        _warn(
            f"MedQuAD not found under {search_root}. "
            "Run `python scripts/download_medquad.py` first."
        )
        return []

    passages: list[Passage] = []
    for xml_path in tqdm(xml_files, desc="MedQuAD"):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        if root.tag != "Document":
            continue  # not a MedQuAD QA file

        title = clean(root.findtext("Focus") or xml_path.stem)
        url = (root.get("url") or "").strip()
        source = (root.get("source") or "MedQuAD").strip()

        for qapair in root.iter("QAPair"):
            question = clean(qapair.findtext("Question") or "")
            answer = clean(qapair.findtext("Answer") or "")
            if not answer:
                continue
            q_el = qapair.find("Question")
            qtype = (q_el.get("qtype") if q_el is not None else "") or "general"

            for i, chunk in enumerate(chunk_text(answer)):
                passages.append(
                    Passage(
                        id=f"medquad__{xml_path.stem}__{question[:32]}__{i}".replace(" ", "_"),
                        text=chunk,
                        question=question,
                        title=title,
                        url=url,
                        source=source,
                        qtype=qtype,
                    )
                )
    return passages


# --- PubMedQA -------------------------------------------------------------
def load_pubmedqa(raw_dir: Path, config: str = "pqa_labeled") -> list[Passage]:
    """Load PubMedQA from Hugging Face and index the abstract contexts.

    Uses the expert-labeled split by default (``pqa_labeled``, ~1k items). The
    retrievable text is each question's supporting abstract plus its long
    answer, cited back to the PubMed article. Requires ``datasets``.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        _warn("PubMedQA needs `datasets` — `pip install datasets`. Skipping.")
        return []

    try:
        ds = load_dataset("qiaojin/PubMedQA", config, split="train")
    except Exception as exc:  # network / dataset errors
        _warn(f"Could not load PubMedQA ({exc}). Skipping.")
        return []

    passages: list[Passage] = []
    for row in tqdm(ds, desc="PubMedQA"):
        pubid = str(row.get("pubid", ""))
        question = clean(row.get("question", ""))
        contexts = row.get("context", {}).get("contexts", []) or []
        long_answer = clean(row.get("long_answer", ""))
        body = clean(" ".join(contexts) + " " + long_answer).strip()
        if not body:
            continue
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pubid}/" if pubid else ""
        for i, chunk in enumerate(chunk_text(body)):
            passages.append(
                Passage(
                    id=f"pubmedqa__{pubid}__{i}",
                    text=chunk,
                    question=question,
                    title=question or f"PubMed {pubid}",
                    url=url,
                    source="PubMedQA",
                    qtype="research",
                )
            )
    return passages


# --- MedlinePlus ----------------------------------------------------------
def load_medlineplus(raw_dir: Path) -> list[Passage]:
    """Parse MedlinePlus bulk health-topic XML (``mplus_topics_*.xml``).

    Download a topic dump from https://medlineplus.gov/xml.html into
    ``data/raw`` (e.g. ``mplus_topics_2024.xml``). Each ``<health-topic>`` has a
    title, url, and an HTML ``<full-summary>`` we strip to plain text.
    """
    files = sorted(raw_dir.rglob("mplus_topics*.xml"))
    if not files:
        _warn(
            "MedlinePlus topic XML not found (mplus_topics_*.xml). "
            "Download from https://medlineplus.gov/xml.html into data/raw. Skipping."
        )
        return []

    passages: list[Passage] = []
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for topic in tqdm(list(root.iter("health-topic")), desc=f"MedlinePlus:{path.stem}"):
            # English topics only; Spanish topics carry language="Spanish".
            if (topic.get("language") or "English") != "English":
                continue
            title = clean(topic.get("title") or "")
            url = (topic.get("url") or "").strip()
            summary = strip_html(topic.findtext("full-summary") or "")
            if not summary:
                continue
            for i, chunk in enumerate(chunk_text(summary)):
                passages.append(
                    Passage(
                        id=f"medlineplus__{title[:40]}__{i}".replace(" ", "_"),
                        text=chunk,
                        question=f"What is {title}?",
                        title=title,
                        url=url,
                        source="MedlinePlus",
                        qtype="information",
                    )
                )
    return passages


# --- Symptom2Disease ------------------------------------------------------
def load_symptom2disease(raw_dir: Path) -> list[Passage]:
    """Convert the Symptom2Disease CSV into per-disease reference passages.

    Download from
    https://www.kaggle.com/datasets/niyarrbarman/symptom2disease into
    ``data/raw`` (the file is typically ``Symptom2Disease.csv`` with columns
    ``label``,``text``). Rows are aggregated per disease into descriptive
    passages of *typical symptom associations* — used as retrievable reference
    text, NOT as classifier training data.
    """
    candidates = list(raw_dir.rglob("Symptom2Disease.csv")) + list(
        raw_dir.rglob("symptom2disease*.csv")
    )
    if not candidates:
        _warn(
            "Symptom2Disease CSV not found. Download from "
            "https://www.kaggle.com/datasets/niyarrbarman/symptom2disease "
            "into data/raw. Skipping."
        )
        return []

    by_disease: dict[str, list[str]] = defaultdict(list)
    with candidates[0].open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = clean(row.get("label") or row.get("disease") or "")
            text = clean(row.get("text") or row.get("symptoms") or "")
            if label and text:
                by_disease[label].append(text)

    passages: list[Passage] = []
    for disease, descriptions in by_disease.items():
        body = (
            f"Typical symptom descriptions associated with {disease}: "
            + " ".join(descriptions)
        )
        for i, chunk in enumerate(chunk_text(body)):
            passages.append(
                Passage(
                    id=f"symptom2disease__{disease[:40]}__{i}".replace(" ", "_"),
                    text=chunk,
                    question=f"What symptoms are associated with {disease}?",
                    title=disease,
                    url="https://www.kaggle.com/datasets/niyarrbarman/symptom2disease",
                    source="Symptom2Disease",
                    qtype="symptom_association",
                )
            )
    return passages


# --- Registry -------------------------------------------------------------
LOADERS = {
    "medquad": load_medquad,
    "pubmedqa": load_pubmedqa,
    "medlineplus": load_medlineplus,
    "symptom2disease": load_symptom2disease,
}


def load_sources(names: list[str], raw_dir: Path) -> list[Passage]:
    """Run the named loaders and concatenate their passages."""
    passages: list[Passage] = []
    for name in names:
        loader = LOADERS.get(name)
        if loader is None:
            _warn(f"Unknown source '{name}'. Known: {', '.join(LOADERS)}")
            continue
        loaded = loader(raw_dir)
        print(f"  {name}: {len(loaded)} passages")
        passages.extend(loaded)
    return passages
