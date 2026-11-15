"""Who is a label *for*, and who does its text *warn about*?

A dog NSAID label mentions cats mainly to forbid them ("Do not use in cats.").
Counting species words would call it a cat drug. So mentions are read sentence by
sentence and split into:
  * positive mentions: "indicated for the control of pain in dogs";
  * caution mentions: "do not use in cats", "safety in kittens has not been evaluated";
  * "only" statements: "for oral use in dogs only" (which rules the other species out).

The target species comes from an evidence ladder (Indications, then explicit use
statements, then Dosage, then the title). There is no full-text fallback: a label
we cannot place is dropped rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SPECIES = ("dog", "cat")
_WORDS = {
    "dog": re.compile(r"\b(dogs?|pupp(y|ies)|canines?)\b", re.I),
    "cat": re.compile(r"\b(cats?|kittens?|felines?)\b", re.I),
}
# Parasite and pest names that contain a species word but are not about the pet.
_NOT_PETS = re.compile(
    r"\b(american |brown )?dog ticks?\b|\bdog (tape|round|hook)worms?\b|\bcat fleas?\b"
    r"|\bcat-?scratch\b|\bcanine (distemper|parvovirus|adenovirus)\b",
    re.I,
)
_CAUTION = re.compile(
    r"\b(do not|don't|not for|not to be|should not|must not|never|avoid|contraindicated"
    r"|not (been )?(established|evaluated|approved|recommended|studied|tested|demonstrated|known)"
    r"|not safe|toxic|fatal|death|lethal|keep .{0,20}away|unsafe|caution)\b",
    re.I,
)
_ONLY = re.compile(
    r"\b(?:in|on|for|to)\s+(?:\w+\s+){0,2}?(dogs?|cats?|kittens?|pupp(?:y|ies))"
    r"(?:\s+(?:and|or)\s+(?:\w+\s+){0,2}?(dogs?|cats?|kittens?|pupp(?:y|ies)))?\s+only\b",
    re.I,
)
_USE = re.compile(
    r"\bfor\s+(?:\w+\s+){0,2}?use\s+(?:in|on)\s+(?:\w+\s+){0,2}?(dogs?|cats?|kittens?|pupp(?:y|ies))"
    r"(?:\s*(?:,|and|or)\s*(?:\w+\s+){0,2}?(dogs?|cats?|kittens?|pupp(?:y|ies)))?",
    re.I,
)
_SENT = re.compile(r"(?<=[.!?;])\s+|\n+")


def _species_of(word: str) -> str:
    w = word.lower()
    return "dog" if w.startswith(("dog", "pupp", "canine")) else "cat"


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def mentions(sentence: str) -> set[str]:
    s = _NOT_PETS.sub(" ", sentence)
    return {sp for sp, rx in _WORDS.items() if rx.search(s)}


def is_caution(sentence: str) -> bool:
    return bool(_CAUTION.search(sentence))


def only_statements(text: str) -> set[str]:
    """Species named in "... in dogs only" style statements (negated ones ignored)."""
    out: set[str] = set()
    for sent in sentences(text):
        if re.search(r"\b(not|never)\b", sent, re.I):
            continue
        for m in _ONLY.finditer(sent):
            out |= {_species_of(g) for g in m.groups() if g}
    return out


def use_statements(text: str) -> set[str]:
    out: set[str] = set()
    for sent in sentences(text):
        if re.search(r"\b(not|never)\b", sent, re.I):
            continue
        for m in _USE.finditer(sent):
            out |= {_species_of(g) for g in m.groups() if g}
    return out


def positive_mentions(text: str) -> set[str]:
    out: set[str] = set()
    for sent in sentences(text):
        if not is_caution(sent):
            out |= mentions(sent)
    return out


def caution_mentions(text: str) -> set[str]:
    out: set[str] = set()
    for sent in sentences(text):
        if is_caution(sent):
            out |= mentions(sent)
    return out


def warned_species(text: str, target: set[str]) -> set[str]:
    """Species a chunk warns about: cautioned by name, or ruled out by an "only" statement."""
    warned = caution_mentions(text)
    only = only_statements(text)
    if only:
        warned |= set(SPECIES) - only
    return warned


@dataclass
class SpeciesProfile:
    target: list[str]
    restricted: list[str]  # species the label rules out (and is not for)
    evidence: str  # which rung of the ladder decided the target
    conflict: bool = False  # label text contradicts itself
    notes: list[str] = field(default_factory=list)


def profile(sections: dict[str, str], title: str) -> SpeciesProfile:
    """`sections` maps canonical section id -> full text of that section."""
    indications = sections.get("indications", "")
    header = sections.get("label_header", "")
    dosage = sections.get("dosage", "")
    everything = "\n".join(sections.values())

    ladder = [
        ("indications", positive_mentions(indications)),
        ("use statement", use_statements("\n".join([header, indications, dosage]))),
        ("dosage", positive_mentions(dosage)),
        ("title", mentions(title)),
    ]
    target, evidence = set(), "none"
    for rung, found in ladder:
        if found:
            target, evidence = found, rung
            break

    conflict = False
    only = only_statements("\n".join([title, header, indications, dosage]))
    if target and only and not target <= only:
        # "For use in dogs only" narrows the target, but never removes a species the
        # Indications explicitly name. That case is kept and flagged.
        named = positive_mentions(indications)
        keep = (target & only) | (target & named)
        conflict = bool(target & named - only)
        target = keep or target

    cautioned = caution_mentions(everything)
    others = set(SPECIES) - target
    restricted = others & (cautioned | (set(SPECIES) - only if only else set()))
    return SpeciesProfile(
        target=sorted(target),
        restricted=sorted(restricted),
        evidence=evidence,
        conflict=conflict,
    )
