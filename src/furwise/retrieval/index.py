"""Build and open the two indexes over the same chunks.

* Dense: bge-small-en-v1.5 embeddings (Sentence Transformers, CPU) in a persistent
  ChromaDB collection, through LlamaIndex.
* Sparse: BM25 (LlamaIndex BM25Retriever, backed by bm25s) built in memory at
  start-up. It takes about a second for this corpus, so it is not persisted.

Each chunk is indexed as "header + text". Metadata (species, section, entity) is
stored next to it for filtering but kept out of the embedded text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

from furwise.corpus.store import Corpus
from furwise.settings import EMBED_MODEL, INDEX

log = logging.getLogger(__name__)

CHROMA_DIR = INDEX / "chroma"
COLLECTION = "furwise"
# bge models expect this instruction on queries, not on passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def fingerprint(corpus: Corpus) -> str:
    h = hashlib.sha256(EMBED_MODEL.encode())
    for c in corpus.chunks:
        h.update(c["chunk_id"].encode())
        h.update(c["header"].encode())
        h.update(c["text"].encode())
    return h.hexdigest()[:16]


def to_node(chunk: dict):
    from llama_index.core.schema import TextNode

    meta = {
        "chunk_id": chunk["chunk_id"],
        "entity_id": chunk["entity_id"],
        "section": chunk["section"],
        "source": chunk["source"],
    }
    return TextNode(
        id_=chunk["chunk_id"],
        text=f"{chunk['header']}\n{chunk['text']}",
        metadata=meta,
        excluded_embed_metadata_keys=list(meta),
        excluded_llm_metadata_keys=list(meta),
    )


def embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        query_instruction=QUERY_INSTRUCTION,
        normalize=True,
        embed_batch_size=32,
        device="cpu",
    )


def _collection(reset: bool = False):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # first build: nothing to delete
            pass
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build_vector_index(corpus: Corpus, full: bool = False) -> dict:
    """Embed the corpus into Chroma. Incremental by default: only chunks whose text
    changed (or is new) are embedded again, and removed chunks are deleted. A full
    embed takes ~13 minutes on a laptop CPU; a typical corpus update takes seconds."""
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    start = time.perf_counter()
    collection = _collection(reset=full)
    store = ChromaVectorStore(chroma_collection=collection)
    stored = collection.get(include=["documents"])
    have = dict(zip(stored["ids"], stored["documents"], strict=True))
    want = {c["chunk_id"]: f"{c['header']}\n{c['text']}" for c in corpus.chunks}

    stale = [cid for cid, doc in have.items() if want.get(cid) != doc]
    for i in range(0, len(stale), 500):
        collection.delete(ids=stale[i : i + 500])
    todo = [c for c in corpus.chunks if have.get(c["chunk_id"]) != want[c["chunk_id"]]]
    if todo:
        index = VectorStoreIndex.from_vector_store(store, embed_model=embed_model())
        index.insert_nodes([to_node(c) for c in todo])

    info = {
        "embed_model": EMBED_MODEL,
        "chunks": collection.count(),
        "embedded": len(todo),
        "removed": len(stale) - sum(1 for cid in stale if cid in want),
        "fingerprint": fingerprint(corpus),
        "seconds": round(time.perf_counter() - start, 1),
    }
    (INDEX / "index_info.json").write_text(json.dumps(info, indent=1), "utf-8")
    return info


def open_vector_index(corpus: Corpus):
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    info_path = INDEX / "index_info.json"
    if not info_path.exists():
        raise RuntimeError("no vector index yet: run `furwise index`")
    if json.loads(info_path.read_text("utf-8"))["fingerprint"] != fingerprint(corpus):
        raise RuntimeError("the corpus changed since the index was built: run `furwise index`")
    store = ChromaVectorStore(chroma_collection=_collection())
    return VectorStoreIndex.from_vector_store(store, embed_model=embed_model())


def bm25_retriever(corpus: Corpus, top_k: int):
    import Stemmer
    from llama_index.retrievers.bm25 import BM25Retriever

    return BM25Retriever.from_defaults(
        nodes=[to_node(c) for c in corpus.chunks],
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
