"""Command line: `furwise fetch | build | index | lookup | ask | eval`."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _fetch(args) -> None:
    from furwise.corpus.build import PER_GROUP_DOWNLOADS
    from furwise.ingest import aspca, dailymed
    from furwise.ingest.http import PoliteClient

    client = PoliteClient()
    if args.source in ("dailymed", "all"):
        index = dailymed.fetch_index(client)
        cand, _ = dailymed.candidates(index, dailymed.load_groups(), PER_GROUP_DOWNLOADS)
        todo = [r for rows in cand.values() for r in rows]
        for i, row in enumerate(todo, 1):
            dailymed.download(client, row["setid"])
            if i % 50 == 0:
                logging.info("dailymed labels: %d/%d", i, len(todo))
    if args.source in ("aspca", "all"):
        aspca.fetch(client, with_details=not args.lists_only)


def _build(args) -> None:
    from furwise.corpus.build import build

    print(json.dumps(build(), indent=2))


def _index(args) -> None:
    from furwise.corpus.store import Corpus
    from furwise.retrieval.index import build_vector_index

    print(json.dumps(build_vector_index(Corpus.load()), indent=2))


def _lookup(args) -> None:
    from furwise.corpus.names import detect_species
    from furwise.corpus.store import get_corpus

    lex = get_corpus().lexicon
    print(
        json.dumps(
            {"species": detect_species(args.question), "matches": lex.describe(args.question)},
            indent=2,
        )
    )


def _ask(args) -> None:
    from furwise.pipeline import Pipeline

    resp = Pipeline(args.stage).answer(args.question)
    print(f"\n{resp.answer}\n")
    for n, chunk in enumerate(resp.contexts, 1):
        mark = "*" if chunk["chunk_id"] in resp.citations else " "
        print(f" {mark}[{n}] {chunk['header']}  ({chunk['url']})")
    print(f"\n{resp.disclaimer}")
    print(f"timings: {resp.timings}")


def _eval(args) -> None:
    from furwise.eval import run as ev

    if args.action == "validate":
        ev.validate()
    elif args.action == "run":
        ev.run(args.stage, judge=args.judge, retrieval_only=args.retrieval_only, limit=args.limit)
    elif args.action == "table":
        ev.table()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    p = argparse.ArgumentParser(prog="furwise", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download DailyMed labels and/or the ASPCA plant list")
    f.add_argument("source", choices=["dailymed", "aspca", "all"], nargs="?", default="all")
    f.add_argument("--lists-only", action="store_true", help="ASPCA: skip plant detail pages")
    f.set_defaults(fn=_fetch)

    sub.add_parser("build", help="parse, profile species, chunk, build lexicon").set_defaults(
        fn=_build
    )
    sub.add_parser("index", help="embed chunks into ChromaDB").set_defaults(fn=_index)

    lk = sub.add_parser("lookup", help="show name and species resolution for a question")
    lk.add_argument("question")
    lk.set_defaults(fn=_lookup)

    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("--stage", default="stage3")
    a.set_defaults(fn=_ask)

    e = sub.add_parser("eval", help="evaluate a pipeline stage")
    e.add_argument("action", choices=["validate", "run", "table"])
    e.add_argument("--stage", default="stage3")
    e.add_argument("--judge", action="store_true", help="also run RAGAS faithfulness")
    e.add_argument("--retrieval-only", action="store_true", help="no LLM: retrieval metrics")
    e.add_argument("--limit", type=int, default=None)
    e.set_defaults(fn=_eval)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
