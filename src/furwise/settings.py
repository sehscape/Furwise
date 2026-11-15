"""Paths and tunables in one place.

Everything can be overridden with environment variables (or a local .env file),
so the same code runs on a laptop, in CI and in the hosted demo.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("FURWISE_ROOT", Path(__file__).resolve().parents[2]))


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader. Real environment variables always win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")

# Quiet, private model loading: no telemetry, no progress bars in logs.
for _key, _value in {
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "ANONYMIZED_TELEMETRY": "False",
    "RAGAS_DO_NOT_TRACK": "true",
}.items():
    os.environ.setdefault(_key, _value)

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
INDEX = DATA / "index"
MANIFEST = DATA / "manifest"
CONFIGS = ROOT / "configs"
EVAL_DIR = ROOT / "eval"
RUNS = ROOT / "runs"

# Generic user agent on purpose. Set FURWISE_USER_AGENT to add contact details.
USER_AGENT = os.getenv("FURWISE_USER_AGENT", "Mozilla/5.0 (compatible; furwise-research/1.0)")

# Seconds between requests per host. robots.txt Crawl-delay wins if it is larger.
MIN_INTERVAL = {"dailymed.nlm.nih.gov": 0.3, "www.aspca.org": 10.0}

# Chunking: bge-small and bge-reranker truncate at 512 tokens. ~250 words of
# label English plus the contextual header stays under that.
CHUNK_MAX_WORDS = 250
CHUNK_MIN_WORDS = 3  # "Do not use in cats." must survive

# Models (all free, all run on CPU).
EMBED_MODEL = os.getenv("FURWISE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("FURWISE_RERANK_MODEL", "BAAI/bge-reranker-base")

# Retrieval
TOP_K = 5  # chunks given to the generator (and scored by the eval)
CANDIDATES = 30  # per retriever, before fusion / reranking
