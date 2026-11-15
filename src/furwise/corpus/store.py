"""Load the processed corpus (chunks + lexicon) once and share it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from furwise.corpus.names import Lexicon
from furwise.settings import PROCESSED


@dataclass
class Corpus:
    chunks: list[dict]
    lexicon: Lexicon
    by_id: dict[str, dict] = field(init=False)
    by_entity: dict[str, list[dict]] = field(init=False)

    def __post_init__(self) -> None:
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.by_entity = {}
        for c in self.chunks:
            self.by_entity.setdefault(c["entity_id"], []).append(c)

    @classmethod
    def load(cls, folder: Path = PROCESSED) -> Corpus:
        path = folder / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError("no corpus yet: run `furwise fetch` and `furwise build`")
        chunks = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
        return cls(chunks, Lexicon.load(folder / "lexicon.json"))


@lru_cache(maxsize=1)
def get_corpus() -> Corpus:
    return Corpus.load()
