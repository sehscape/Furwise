"""Run the evaluation for one stage and keep the results table up to date.

    furwise eval validate                         test set vs the current corpus
    furwise eval run --stage stage0 --retrieval-only    free: retrieval metrics only
    furwise eval run --stage stage0 --judge       full: answers + RAGAS faithfulness
    furwise eval table                            results.json -> README / results.md

Every run writes runs/<stage>/<timestamp>/{responses,scores}.jsonl + summary.json, and
merges its summary into eval/results/results.json (committed). LLM calls are cached,
so a run stopped by a free-tier daily limit resumes where it stopped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from furwise.corpus.store import get_corpus
from furwise.eval import dataset, metrics
from furwise.settings import EVAL_DIR, ROOT, RUNS

log = logging.getLogger(__name__)
RESULTS = EVAL_DIR / "results" / "results.json"
RETRIEVAL_KEYS = (
    "hit@5",
    "recall@5",
    "mrr",
    "retrieval_mixup",
    "context_precision",
    "context_recall",
    "retrieval_latency_p50_ms",
    "retrieval_latency_p95_ms",
)


def validate() -> None:
    corpus = get_corpus()
    questions = dataset.load()
    problems = dataset.validate(questions, corpus)
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.category] = counts.get(q.category, 0) + 1
    print(f"{len(questions)} questions: {counts}")
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    print("test set OK: every gold quote resolves and every name variant resolves")


def _text(chunk: dict) -> str:
    return f"{chunk['header']}\n{chunk['text']}"


def run(
    stage: str, judge: bool = False, retrieval_only: bool = False, limit: int | None = None
) -> dict:
    from furwise.eval.ragas_eval import FreeJudge, RagasScorer, answer_claims
    from furwise.pipeline import STAGES, Pipeline
    from furwise.retrieval.search import RERANK_MAX_TOKENS, RERANK_POOL

    corpus = get_corpus()
    questions = dataset.load()[:limit]
    sample = dataset.faithfulness_sample()
    problems = dataset.validate(questions, corpus)
    if problems:
        raise SystemExit("test set problems:\n" + "\n".join(problems))

    pipe = Pipeline(stage, corpus)
    scorer = RagasScorer(FreeJudge() if judge and not retrieval_only else None)
    out_dir = RUNS / stage / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe.retrieve("warm up the models")  # first call loads weights; keep it out of latency

    rows, latencies, retrieval_latencies = [], [], []
    with (out_dir / "responses.jsonl").open("w", encoding="utf-8") as fh:
        for i, q in enumerate(questions, 1):
            gold = dataset.gold_map(q, corpus)
            if retrieval_only:
                start = time.perf_counter()
                hits, _ = pipe.retrieve(q.question)
                retrieval_latencies.append((time.perf_counter() - start) * 1000)
                contexts = [corpus.by_id[h.chunk_id] for h in hits]
                answer, citations, record = None, [], {}
            else:
                resp = pipe.answer(q.question)
                contexts, answer, citations = resp.contexts, resp.answer, resp.citations
                latencies.append(resp.timings["total_ms"])
                retrieval_latencies.append(resp.timings["retrieval_ms"])
                record = {
                    "answer": answer,
                    "citations": citations,
                    "timings": resp.timings,
                    "meta": resp.meta,
                    "declined": resp.declined,
                }

            row = metrics.score_question(q, answer, citations, contexts, gold, corpus.by_id)
            if gold and contexts:
                gold_ids = {c for ids in gold.values() for c in ids}
                ref_of = {c: ref for ref, ids in gold.items() for c in ids}
                row.update(
                    asyncio.run(
                        scorer.context_scores(
                            retrieved=[_text(c) for c in contexts],
                            gold_texts=[_text(corpus.by_id[c]) for c in sorted(gold_ids)],
                            retrieved_ref_ids=[
                                ref_of.get(c["chunk_id"], c["chunk_id"]) for c in contexts
                            ],
                            ref_ids=list(gold),
                        )
                    )
                )
            claims = answer_claims(answer or "")
            judged = not sample or q.id in sample
            if (
                scorer.faithfulness is not None
                and judged
                and contexts
                and claims
                and not row.get("declined")
            ):
                try:
                    row["faithfulness"] = asyncio.run(
                        scorer.faithfulness_score(q.question, claims, [_text(c) for c in contexts])
                    )
                except Exception as exc:  # a judge failure must not kill a long run
                    log.warning("%s: faithfulness failed: %s", q.id, exc)
            rows.append(row)
            fh.write(
                json.dumps(
                    {
                        "id": q.id,
                        "question": q.question,
                        **record,
                        "contexts": [c["chunk_id"] for c in contexts],
                        "scores": row,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            if i % 10 == 0:
                log.info("%s: %d/%d questions", stage, i, len(questions))

    summary = metrics.summarize(rows, latencies, retrieval_latencies)
    summary.update(
        {
            "stage": stage,
            "label": STAGES[stage]["label"],
            "mode": "retrieval-only" if retrieval_only else "full",
            "generator": None if retrieval_only else pipe.generator.client.config.label,
            "judge": scorer.judge.label if scorer.judge else None,
            "faithfulness_sample": len(sample) or len(questions),
            "rerank": {"pool": RERANK_POOL, "max_tokens": RERANK_MAX_TOKENS}
            if STAGES[stage]["rerank"]
            else None,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "run_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    if limit is None:  # only complete runs feed the results table
        _merge(stage, summary, retrieval_only)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=2))
    return summary


def _merge(stage: str, summary: dict, retrieval_only: bool) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text("utf-8")) if RESULTS.exists() else {}
    entry = results.get(stage, {})
    if retrieval_only:  # never overwrite answer metrics with a retrieval-only run
        entry.update({k: summary.get(k) for k in RETRIEVAL_KEYS})
        entry.update(
            {"stage": stage, "label": summary["label"], "retrieval_run": summary["run_dir"]}
        )
        entry.setdefault("n", summary["n"])
    else:
        entry = summary
    results[stage] = entry
    RESULTS.write_text(json.dumps(results, indent=2), "utf-8")


# ---------------------------------------------------------------- tables
def _pct(v, invert: bool = False) -> str:
    return "⏳" if v is None else f"{v * 100:.0f}%"


def _num(v) -> str:
    return "⏳" if v is None else f"{v:.2f}"


def _ms(a, b) -> str:
    return "⏳" if a is None else f"{a / 1000:.1f} s / {b / 1000:.1f} s"


def render_tables(results: dict) -> str:
    from furwise.pipeline import STAGES

    main = [
        "| Stage | Context precision | Context recall | Faithfulness | Citation accuracy "
        "| Species mix-up ↓ | Emergency detection | Correct decline | Latency p50 / p95 |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    detail = [
        "| Stage | Hit@5 | MRR | Retrieval mix-up ↓ | Missed species traps ↓ "
        "| False emergency ↓ | Over-decline ↓ | Retrieval p50 / p95 |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for stage, cfg in STAGES.items():
        if not cfg.get("table", True):
            continue
        r = results.get(stage, {})
        main.append(
            f"| {cfg['label']} | {_num(r.get('context_precision'))} | "
            f"{_num(r.get('context_recall'))} | {_num(r.get('faithfulness'))} | "
            f"{_pct(r.get('citation_accuracy'))} | {_pct(r.get('species_mixup'))} | "
            f"{_pct(r.get('emergency_detection'))} | {_pct(r.get('correct_decline'))} | "
            f"{_ms(r.get('latency_p50_ms'), r.get('latency_p95_ms'))} |"
        )
        detail.append(
            f"| {cfg['label']} | {_pct(r.get('hit@5'))} | {_num(r.get('mrr'))} | "
            f"{_pct(r.get('retrieval_mixup'))} | {_pct(r.get('missed_trap'))} | "
            f"{_pct(r.get('false_emergency'))} | {_pct(r.get('over_decline'))} | "
            f"{_ms(r.get('retrieval_latency_p50_ms'), r.get('retrieval_latency_p95_ms'))} |"
        )
    return "\n".join(main) + "\n\n**Retrieval and safety details**\n\n" + "\n".join(detail)


def table(readme: Path = ROOT / "README.md") -> None:
    results = json.loads(RESULTS.read_text("utf-8")) if RESULTS.exists() else {}
    block = render_tables(results)
    (EVAL_DIR / "results" / "results.md").write_text(block + "\n", "utf-8")
    if readme.exists():
        text = readme.read_text("utf-8")
        start, end = "<!-- results:start -->", "<!-- results:end -->"
        if start in text and end in text:
            head, rest = text.split(start, 1)
            _, tail = rest.split(end, 1)
            readme.write_text(f"{head}{start}\n{block}\n{end}{tail}", "utf-8")
    print(block)
