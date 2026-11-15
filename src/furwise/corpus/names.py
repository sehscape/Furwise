"""Name resolution: brand <-> generic, common <-> scientific, and misspellings.

Owners type "rimadil", "nexguard", "Heartgard", "Lilium" or "tiger lilies". The
lexicon is built only from the sources themselves (DailyMed titles and product
data, ASPCA common / alternate / scientific names) and resolves those to entity ids.

Matching is exact (longest n-gram first) with two careful extensions:
  * fuzzy matching (stdlib difflib) for drug names only. Plant names are ordinary
    words, and fuzzy plant matching turned "garden" into "gardenia";
  * plurals for plant names ("tulips" -> tulip, "lilies" -> lily).
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# Words that are plant or product names somewhere but mean something else in a question.
_STOP = set(
    """a an and are at be can could did do does dog dogs cat cats puppy kitten for from give
    gave had has have he her his how i if in into is it its may me my of on or our safe she
    should so that the their them they this to too use used was we what when which who why will
    with would you your pet pets plant plants tree trees flower flowers bush leaf leaves seed
    seeds root fruit ate eat eaten eating chew chewed lick licked dose doses mg medicine
    medication drug drugs tablet tablets pill pills chewable injection liquid oral spray
    topical dosage label vet vets much many day daily week weekly month monthly year old
    time times also tell about toxic poisonous dangerous okay ok poison other some any
    common plus max ultra one two three bad good new day night water food""".split()
)


def norm(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower().replace("®", " ").replace("™", " ")))


def singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


@dataclass
class Match:
    entity_id: str
    alias: str  # the lexicon name that matched
    text: str  # what the user typed
    kind: str  # "exact", "plural" or "fuzzy"
    score: float = 1.0


class Lexicon:
    def __init__(
        self,
        drug_aliases: dict[str, list[str]],
        plant_aliases: dict[str, list[str]],
        names: dict[str, str],
    ):
        self.drug = drug_aliases  # normalised alias -> entity ids
        self.plant = plant_aliases
        self.names = names  # entity id -> display name
        self._max_n = max((len(a.split()) for a in [*self.drug, *self.plant]), default=1)
        self._fuzzy_pool = [a for a in self.drug if len(a) >= 5]

    # ------------------------------------------------------------ building
    @classmethod
    def build(
        cls,
        drug_names: dict[str, set[str]],
        plant_names: dict[str, set[str]],
        display: dict[str, str],
    ) -> Lexicon:
        def invert(names: dict[str, set[str]]) -> dict[str, list[str]]:
            out: dict[str, list[str]] = {}
            for entity, aliases in names.items():
                for alias in aliases:
                    a = norm(alias)
                    if len(a) < 3 or a in _STOP:
                        continue
                    out.setdefault(a, [])
                    if entity not in out[a]:
                        out[a].append(entity)
            return out

        return cls(invert(drug_names), invert(plant_names), display)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"drug": self.drug, "plant": self.plant, "names": self.names}, ensure_ascii=False
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Lexicon:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["drug"], data["plant"], data["names"])

    # ------------------------------------------------------------ resolving
    def resolve(self, question: str) -> list[Match]:
        tokens = norm(question).split()
        used = [False] * len(tokens)
        matches: list[Match] = []

        for n in range(min(self._max_n, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                if any(used[i : i + n]):
                    continue
                gram = " ".join(tokens[i : i + n])
                plural = " ".join([*tokens[i : i + n - 1], singular(tokens[i + n - 1])])
                hit = None
                if gram in self.drug:
                    hit = (self.drug[gram], gram, "exact")
                elif gram in self.plant:
                    hit = (self.plant[gram], gram, "exact")
                elif plural != gram and plural in self.plant:
                    hit = (self.plant[plural], plural, "plural")
                if hit and not (n == 1 and gram in _STOP):
                    for entity in hit[0]:
                        matches.append(Match(entity, hit[1], gram, hit[2]))
                    used[i : i + n] = [True] * n

        for i, tok in enumerate(tokens):  # misspelled drug names
            if used[i] or len(tok) < 5 or tok in _STOP or tok.isdigit():
                continue
            close = difflib.get_close_matches(tok, self._fuzzy_pool, n=1, cutoff=0.82)
            if close:
                score = difflib.SequenceMatcher(None, tok, close[0]).ratio()
                for entity in self.drug[close[0]]:
                    matches.append(Match(entity, close[0], tok, "fuzzy", round(score, 2)))

        seen, unique = set(), []
        for m in matches:
            if m.entity_id not in seen:
                seen.add(m.entity_id)
                unique.append(m)
        return unique

    def entities(self, question: str) -> list[str]:
        return [m.entity_id for m in self.resolve(question)]

    def describe(self, question: str) -> list[dict]:
        return [
            {**asdict(m), "name": self.names.get(m.entity_id, m.entity_id)}
            for m in self.resolve(question)
        ]


def detect_species(question: str) -> list[str]:
    from furwise.corpus.species import mentions

    return sorted(mentions(question))
