"""The evaluation set: questions plus gold evidence written as quotes.

Gold is stored as {entity, section, quote}, not chunk ids. Chunk ids change when
the chunker changes; quotes from the label do not. At evaluation time every chunk
of that entity/section containing the quote is gold, so the same sentence on the
tablet label *and* the injection label both count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from furwise.corpus.store import Corpus
from furwise.settings import EVAL_DIR

CATEGORIES = ("label_fact", "plant_fact", "species_trap", "name_variant", "emergency", "decline")
EXPECT = ("answer", "restrict", "emergency", "decline")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%./]+", " ", text.lower())).strip()


@dataclass
class GoldRef:
    entity: str
    section: str  # canonical section, or "*" for any section of the entity
    quote: str


@dataclass
class Question:
    id: str
    category: str
    question: str
    expect: str
    species: str = ""  # "dog", "cat" or "" (not species specific)
    gold: list[GoldRef] = field(default_factory=list)
    reference: str = ""  # reference answer, written from the gold text only
    entity: str = ""  # name-variant questions: the entity the name must resolve to
    note: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.expect in ("answer", "restrict")


def load(path: Path = EVAL_DIR / "questions.yaml") -> list[Question]:
    raw = yaml.safe_load(path.read_text("utf-8"))
    out = []
    for q in raw["questions"]:
        gold = [GoldRef(g["entity"], g.get("section", "*"), g["quote"]) for g in q.get("gold", [])]
        out.append(
            Question(
                id=q["id"],
                category=q["category"],
                question=q["question"].strip(),
                expect=q["expect"],
                species=q.get("species", ""),
                gold=gold,
                reference=q.get("reference", "").strip(),
                entity=q.get("entity", ""),
                note=q.get("note", ""),
            )
        )
    return out


def resolve(ref: GoldRef, corpus: Corpus) -> list[str]:
    """Chunk ids of `ref.entity` / `ref.section` whose text contains the quote."""
    quote = _norm(ref.quote)
    return [
        c["chunk_id"]
        for c in corpus.by_entity.get(ref.entity, [])
        if (ref.section in ("*", c["section"])) and quote in _norm(c["text"])
    ]


def gold_map(q: Question, corpus: Corpus) -> dict[str, list[str]]:
    """Reference id -> equivalent gold chunk ids."""
    return {f"{q.id}#ref{i}": resolve(ref, corpus) for i, ref in enumerate(q.gold)}


def validate(questions: list[Question], corpus: Corpus) -> list[str]:
    """Problems with the test set against the current corpus (empty list = valid)."""
    problems, seen = [], set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.category not in CATEGORIES:
            problems.append(f"{q.id}: unknown category {q.category}")
        if q.expect not in EXPECT:
            problems.append(f"{q.id}: unknown expect {q.expect}")
        if q.expect == "decline" and q.gold:
            problems.append(f"{q.id}: a must-decline question cannot have gold evidence")
        if q.is_answerable and not q.gold:
            problems.append(f"{q.id}: answerable question without gold evidence")
        for i, ref in enumerate(q.gold):
            if ref.entity not in corpus.by_entity:
                problems.append(f"{q.id}: unknown entity {ref.entity}")
            elif not resolve(ref, corpus):
                problems.append(f"{q.id}: gold #{i} quote not found: {ref.quote[:60]!r}")
        if q.entity and q.entity not in corpus.lexicon.entities(q.question):
            problems.append(
                f"{q.id}: name does not resolve to {q.entity}: "
                f"{corpus.lexicon.entities(q.question)}"
            )
    return problems
