"""FastAPI service.

uvicorn furwise.api:app --port 8000

GET  /health          corpus size and loaded stage
GET  /lookup?q=...    name + species resolution (no LLM)
POST /ask             {"question": "...", "stage": "stage3"} -> cited answer
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from furwise import DISCLAIMER, __version__
from furwise.corpus.names import detect_species
from furwise.corpus.store import get_corpus
from furwise.llm import LLMError
from furwise.pipeline import STAGES, Pipeline
from furwise.safety import POISON_CONTROL


@lru_cache(maxsize=4)
def get_pipeline(stage: str) -> Pipeline:
    return Pipeline(stage)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline("stage3")  # load models once, before the first request
    yield


app = FastAPI(
    title="Furwise API",
    version=__version__,
    description="Species-aware answers about dog and cat medicines and toxic plants, "
    "grounded in DailyMed labels and the ASPCA plant list. " + DISCLAIMER,
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    stage: str = "stage3"


class Citation(BaseModel):
    n: int
    product: str
    section: str
    source: str
    url: str
    cited: bool
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    emergency: bool
    declined: bool
    species: list[str]
    resolved_names: list[dict]
    sources: list[Citation]
    poison_control: str | None
    timings: dict
    stage: str
    disclaimer: str = DISCLAIMER


@app.get("/health")
def health() -> dict:
    corpus = get_corpus()
    return {
        "status": "ok",
        "version": __version__,
        "chunks": len(corpus.chunks),
        "entities": len(corpus.by_entity),
        "stages": list(STAGES),
    }


@app.get("/lookup")
def lookup(q: str) -> dict:
    return {"species": detect_species(q), "matches": get_corpus().lexicon.describe(q)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.stage not in STAGES:
        raise HTTPException(400, f"stage must be one of {list(STAGES)}")
    try:
        resp = get_pipeline(req.stage).answer(req.question)
    except LLMError as exc:
        raise HTTPException(503, f"the language model is unavailable: {exc}") from exc
    sources = [
        Citation(
            n=i,
            product=c["entity_name"] if c["source"] == "ASPCA" else c["product"],
            section=c["section_name"],
            source=c["source"],
            url=c["url"],
            cited=c["chunk_id"] in resp.citations,
            text=c["text"],
        )
        for i, c in enumerate(resp.contexts, 1)
    ]
    return AskResponse(
        question=resp.question,
        answer=resp.answer,
        emergency=resp.emergency,
        declined=resp.declined,
        species=resp.species,
        resolved_names=resp.entities,
        sources=sources,
        poison_control=POISON_CONTROL if resp.emergency else None,
        timings=resp.timings,
        stage=req.stage,
    )
