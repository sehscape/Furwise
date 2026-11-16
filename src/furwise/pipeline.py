"""The four pipeline stages compared in the results table.

Each stage adds one layer on top of the previous one:

    stage0  dense retrieval (bge-small + Chroma), top 5 -> LLM with a cited prompt
    stage1  + BM25, fused with dense by Reciprocal Rank Fusion
    stage2  + BGE cross-encoder reranker
    stage3  + name resolution, entity & species metadata filters, restriction pinning,
              emergency / scope gates and the citation verifier (safety in code)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from furwise import DISCLAIMER
from furwise.corpus.names import detect_species
from furwise.corpus.store import Corpus, get_corpus
from furwise.generate import Generator
from furwise.retrieval.search import Hit, Searcher
from furwise.safety import (
    DECLINE_PREFIX,
    EMERGENCY_LINE,
    EMERGENCY_PREFIX,
    check_scope,
    decline_message,
    is_emergency,
    verify,
)
from furwise.settings import TOP_K

STAGES = {
    "stage0": {
        "hybrid": False,
        "rerank": False,
        "safety": False,
        "label": "0 · Baseline (dense top-k)",
    },
    "stage1": {
        "hybrid": True,
        "rerank": False,
        "safety": False,
        "label": "1 · + Hybrid (BM25 + dense, RRF)",
    },
    "stage2": {"hybrid": True, "rerank": True, "safety": False, "label": "2 · + BGE reranker"},
    "stage3": {
        "hybrid": True,
        "rerank": True,
        "safety": True,
        "label": "3 · + Species filters & safety",
    },
    # The hosted demo: Stage 3 without the cross-encoder, so it fits a free host's ~1 GB
    # of RAM and answers in about a second on a shared CPU. Evaluated separately.
    "stage3_lite": {
        "hybrid": True,
        "rerank": False,
        "safety": True,
        "label": "3-lite · Stage 3 without the reranker (hosted demo)",
        "table": False,
    },
}


@dataclass
class Response:
    question: str
    answer: str
    citations: list[str]  # chunk ids, in the order they are cited
    contexts: list[dict]  # retrieved chunks, best first; [n] in the answer = contexts[n-1]
    emergency: bool = False
    declined: bool = False
    entities: list[dict] = field(default_factory=list)
    species: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


class Pipeline:
    def __init__(
        self,
        stage: str = "stage3",
        corpus: Corpus | None = None,
        generator: Generator | None = None,
    ):
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
        self.stage = stage
        self.cfg = STAGES[stage]
        self.corpus = corpus or get_corpus()
        self.searcher = Searcher(self.corpus, hybrid=self.cfg["hybrid"], rerank=self.cfg["rerank"])
        self._generator = generator

    @property
    def generator(self) -> Generator:
        if self._generator is None:  # created lazily: retrieval-only runs need no API key
            self._generator = Generator()
        return self._generator

    # ---------------------------------------------------------------- retrieval
    def resolve(self, question: str) -> dict:
        """Names and species in the question (Stage 3 only)."""
        matches = self.corpus.lexicon.describe(question)
        return {
            "matches": matches,
            "entities": [m["entity_id"] for m in matches],
            "species": detect_species(question),
        }

    def retrieve(self, question: str, resolved: dict | None = None) -> tuple[list[Hit], dict]:
        """Top-k chunks for a question. Also used alone by the retrieval-only evaluation."""
        if not self.cfg["safety"]:
            res = self.searcher.search(question, TOP_K)
            return res.hits, {"timings": res.timings}
        r = resolved or self.resolve(question)
        names = " ".join(dict.fromkeys(m["name"] for m in r["matches"]))
        query = f"{question} {names}".strip()  # "rimadil" -> "... Rimadyl (carprofen)"
        res = self.searcher.search(query, TOP_K, r["entities"] or None, r["species"] or None)
        hits = self._pin_restrictions(query, res.hits, r["entities"], r["species"])
        return hits, {"timings": res.timings, **r}

    def _pin_restrictions(
        self, query: str, hits: list[Hit], entities: list[str], species: list[str]
    ) -> list[Hit]:
        """If a label rules out the asked species, its restriction must be in the context.
        "Can I give my cat Rimadyl?" is answered by "Do not use in cats", which may not
        be the most *similar* chunk to the question, but it is the one that matters."""
        for s in species:
            for entity in entities:
                restricting = [
                    c["chunk_id"]
                    for c in self.corpus.by_entity.get(entity, [])
                    if s in c["caution_species"] and s not in c["species_target"]
                ]
                if not restricting or any(h.chunk_id in restricting for h in hits):
                    continue
                best = self.searcher.best_of(query, restricting)
                if best:
                    hits = [
                        Hit(best, 1.0, f"pinned: label restricts {s}s"),
                        *[h for h in hits if h.chunk_id != best],
                    ][:TOP_K]
        return hits

    # ---------------------------------------------------------------- answering
    def answer(self, question: str) -> Response:
        start = time.perf_counter()
        if not self.cfg["safety"]:
            return self._answer_plain(question, start)
        return self._answer_safe(question, start)

    def _answer_plain(self, question: str, start: float) -> Response:
        hits, info = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - start) * 1000
        chunks = [self.corpus.by_id[h.chunk_id] for h in hits]
        gen = self.generator.answer(question, chunks)
        answer = gen.answer.strip()
        return Response(
            question=question,
            answer=answer,
            citations=[chunks[n - 1]["chunk_id"] for n in gen.numbers],
            contexts=chunks,
            emergency=answer.startswith(EMERGENCY_PREFIX),
            declined=answer.startswith(DECLINE_PREFIX) or not answer,
            timings={
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(gen.ms, 1),
                "total_ms": round(retrieval_ms + gen.ms, 1),
                **_round(info["timings"]),
            },
            meta={"stage": self.stage, **gen.meta},
        )

    def _answer_safe(self, question: str, start: float) -> Response:
        emergency = is_emergency(question)
        resolved = self.resolve(question)
        species = resolved["species"]
        scope = check_scope(question, resolved["entities"], species)
        base = {
            "question": question,
            "emergency": emergency,
            "entities": resolved["matches"],
            "species": species,
        }

        if not scope.allowed:  # decided before retrieval: nothing to look up
            retrieval_ms = (time.perf_counter() - start) * 1000
            answer = decline_message(scope.reason)
            if emergency:
                answer = f"{EMERGENCY_LINE} {answer}"
            return Response(
                **base,
                answer=answer,
                citations=[],
                contexts=[],
                declined=True,
                timings={
                    "retrieval_ms": round(retrieval_ms, 1),
                    "generation_ms": 0.0,
                    "total_ms": round(retrieval_ms, 1),
                },
                meta={"stage": self.stage, "gate": scope.reason},
            )

        hits, info = self.retrieve(question, resolved)
        retrieval_ms = (time.perf_counter() - start) * 1000
        chunks = [self.corpus.by_id[h.chunk_id] for h in hits]
        gen = self.generator.answer(question, chunks, species, safe=True)
        checked = verify(gen.answer, chunks)
        declined = not checked.answer
        answer = (
            decline_message("the label and plant text I retrieved does not clearly answer that")
            if declined
            else checked.answer
        )
        if emergency:
            answer = f"{EMERGENCY_LINE} {answer}"
        return Response(
            **base,
            answer=answer,
            citations=[chunks[n - 1]["chunk_id"] for n in checked.citations],
            contexts=chunks,
            declined=declined,
            timings={
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(gen.ms, 1),
                "total_ms": round(retrieval_ms + gen.ms, 1),
                **_round(info["timings"]),
            },
            meta={
                "stage": self.stage,
                **gen.meta,
                "dropped": checked.dropped,
                "pinned": [h.chunk_id for h in hits if h.why],
            },
        )


def _round(d: dict) -> dict:
    return {k: round(v, 1) for k, v in d.items()}
