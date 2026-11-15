"""Retrieval for every stage, configured by three switches.

    dense only            Stage 0   bge-small cosine top-k
    + bm25 & RRF          Stage 1   hybrid: exact names/doses (BM25) + meaning (dense)
    + reranker            Stage 2   bge-reranker-base cross-encoder over the fused list
    + filters             Stage 3   entity + species metadata filters, restriction pinning

Filtering is done on an over-fetched candidate list in Python rather than inside
each index. It is exact, identical for both retrievers, and cheap at this corpus
size (a few thousand chunks).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from furwise.corpus.store import Corpus
from furwise.retrieval import index as idx
from furwise.settings import CANDIDATES, RERANK_MODEL, TOP_K

RRF_K = 60  # standard Reciprocal Rank Fusion constant
OVERFETCH = 120  # candidates pulled before filters are applied
# The cross-encoder costs ~0.35 s per passage on a laptop CPU, so it re-scores the top
# 12 fused candidates, with inputs capped at 320 tokens. On a 20-passage sample that
# cap gave the same top 5 as the full 512 tokens at half the time; int8 quantization
# was faster still but reordered the top 5, so it was not used.
RERANK_POOL = 12
RERANK_MAX_TOKENS = 320


@dataclass
class Hit:
    chunk_id: str
    score: float
    why: str = ""  # "pinned: label restricts cat", etc.


@dataclass
class SearchResult:
    hits: list[Hit]
    timings: dict[str, float] = field(default_factory=dict)


def species_ok(chunk: dict, species: list[str]) -> bool:
    """Keep a chunk if the asked species is a target of its label, OR the chunk
    warns about that species. A plain `species == cat` filter would delete
    "Do not use in cats." from a dog label, which is exactly the answer."""
    if not species:
        return True
    return any(s in chunk["species_target"] or s in chunk["caution_species"] for s in species)


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Searcher:
    def __init__(self, corpus: Corpus, hybrid: bool = False, rerank: bool = False):
        self.corpus = corpus
        self.hybrid = hybrid
        self.use_rerank = rerank
        self.vector = idx.open_vector_index(corpus)
        self.bm25 = idx.bm25_retriever(corpus, OVERFETCH) if hybrid else None
        self.reranker = None
        if rerank:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_TOKENS, device="cpu")

    # ------------------------------------------------------------- pieces
    def _dense(self, query: str, n: int, entities: list[str] | None) -> list[tuple[str, float]]:
        from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

        filters = None
        if entities:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="entity_id", value=entities, operator=FilterOperator.IN)
                ]
            )
        retriever = self.vector.as_retriever(similarity_top_k=n, filters=filters)
        return [(r.node.node_id, float(r.score or 0.0)) for r in retriever.retrieve(query)]

    def _sparse(self, query: str) -> list[tuple[str, float]]:
        return [(r.node.node_id, float(r.score or 0.0)) for r in self.bm25.retrieve(query)]

    def _filter(self, ranked, entities, species) -> list[tuple[str, float]]:
        out = []
        for cid, score in ranked:
            chunk = self.corpus.by_id[cid]
            if entities and chunk["entity_id"] not in entities:
                continue
            if not species_ok(chunk, species):
                continue
            out.append((cid, score))
        return out

    def _rerank(self, query: str, ids: list[str], top_n: int) -> list[tuple[str, float]]:
        chunks = [self.corpus.by_id[c] for c in ids]
        pairs = [(query, f"{c['header']}\n{c['text']}") for c in chunks]
        scores = self.reranker.predict(pairs, batch_size=1)
        ranked = sorted(zip(ids, (float(x) for x in scores), strict=True), key=lambda kv: -kv[1])
        return ranked[:top_n]

    # ------------------------------------------------------------- main
    def search(
        self,
        query: str,
        k: int = TOP_K,
        entities: list[str] | None = None,
        species: list[str] | None = None,
    ) -> SearchResult:
        t: dict[str, float] = {}
        filtering = bool(entities or species)
        n = OVERFETCH if filtering else CANDIDATES

        start = time.perf_counter()
        dense = self._filter(self._dense(query, n, entities), entities, species)[:CANDIDATES]
        t["dense_ms"] = (time.perf_counter() - start) * 1000

        if self.hybrid:
            start = time.perf_counter()
            sparse = self._filter(self._sparse(query), entities, species)[:CANDIDATES]
            fused = rrf([[c for c, _ in dense], [c for c, _ in sparse]])
            t["bm25_ms"] = (time.perf_counter() - start) * 1000
        else:
            fused = dense

        if self.reranker is not None and fused:
            start = time.perf_counter()
            ranked = self._rerank(query, [c for c, _ in fused[:RERANK_POOL]], k)
            t["rerank_ms"] = (time.perf_counter() - start) * 1000
        else:
            ranked = fused[:k]
        return SearchResult([Hit(c, s) for c, s in ranked[:k]], t)

    def best_of(self, query: str, chunk_ids: list[str]) -> str | None:
        """Most similar chunk among `chunk_ids` (dense similarity: fast, no reranker)."""
        if not chunk_ids:
            return None
        ranked = self._dense(
            query, OVERFETCH, list({self.corpus.by_id[c]["entity_id"] for c in chunk_ids})
        )
        for cid, _ in ranked:
            if cid in chunk_ids:
                return cid
        return chunk_ids[0]
