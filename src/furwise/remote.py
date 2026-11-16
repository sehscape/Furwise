"""Ship the prebuilt corpus and index to a hosted demo.

Building the corpus means ~75 minutes of polite ASPCA crawling and ~13 minutes of
embedding, which no free host should repeat on every start. Instead the locally built
`data/processed`, `data/index` and `data/manifest` are pushed once to a **private**
Hugging Face dataset repo (free), and the hosted app downloads them at start-up:

    python -m furwise.remote push <hf-user>/furwise-index     # from your machine
    FURWISE_DATA_REPO=<hf-user>/furwise-index HF_TOKEN=...      # on the host

The repo is private so processed ASPCA content is not republished; every answer in the
demo still links to the original DailyMed or ASPCA page.
"""

from __future__ import annotations

import os
import sys

from furwise.settings import DATA, INDEX, PROCESSED

FOLDERS = ["processed/**", "index/**", "manifest/**"]


def ensure_data() -> bool:
    """Download the corpus and index if they are missing and a data repo is configured.
    Returns True when the local data is (now) present."""
    if (PROCESSED / "chunks.jsonl").exists() and (INDEX / "index_info.json").exists():
        return True
    repo = os.getenv("FURWISE_DATA_REPO")
    if not repo:
        return False
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(DATA),
        allow_patterns=FOLDERS,
        token=os.getenv("HF_TOKEN") or None,
    )
    return (PROCESSED / "chunks.jsonl").exists()


def push(repo: str) -> None:
    from huggingface_hub import HfApi

    for needed in (PROCESSED / "chunks.jsonl", INDEX / "index_info.json"):
        if not needed.exists():
            raise SystemExit(f"missing {needed}: run `furwise build` and `furwise index` first")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(DATA),
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=FOLDERS,
        ignore_patterns=["raw/**"],
        commit_message="Update Furwise corpus and index",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo} (private)")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "push":
        raise SystemExit("usage: python -m furwise.remote push <hf-user>/furwise-index")
    push(sys.argv[2])
