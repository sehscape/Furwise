"""Publish the demo to a free Hugging Face Space (Docker SDK, CPU basic).

    pip install huggingface_hub
    huggingface-cli login                     # a free account and a write token
    python scripts/deploy_space.py <your-hf-username>/furwise

Then add GROQ_API_KEY as a secret in the Space settings. The script uploads the code
plus the locally built corpus and vector index; raw downloads (data/raw) are never
uploaded. Every answer in the demo links back to its DailyMed or ASPCA page.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Furwise
emoji: 🐾
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Species-aware pet medication and plant toxicity assistant
---

# 🐾 Furwise

Species-aware answers about dog and cat medicines and toxic plants, grounded in FDA
animal drug labels (DailyMed) and the ASPCA Toxic and Non-Toxic Plant List, with a
citation on every claim.

**For information only. Not a substitute for veterinary advice.**

Code, architecture and evaluation: https://github.com/sehscape/Furwise-Species-Aware-RAG-Assistant-for-Veterinary-Drug-Toxicity-Safety
"""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    from huggingface_hub import HfApi

    repo_id = sys.argv[1]
    for needed in ("data/processed/chunks.jsonl", "data/index/index_info.json"):
        if not (ROOT / needed).exists():
            raise SystemExit(f"missing {needed}: run `furwise build` and `furwise index` first")

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_file(
        path_or_fileobj=SPACE_README.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=[
            "Dockerfile",
            "requirements.txt",
            "pyproject.toml",
            "LICENSE",
            "src/**",
            "app/**",
            "configs/**",
            "eval/results/**",
            "data/manifest/**",
            "data/processed/**",
            "data/index/**",
        ],
        ignore_patterns=["**/__pycache__/**", "data/raw/**"],
    )
    print(
        f"Uploaded. Add GROQ_API_KEY under Settings -> Secrets: "
        f"https://huggingface.co/spaces/{repo_id}/settings"
    )


if __name__ == "__main__":
    main()
