# Evaluation

Every stage of the build is scored by the same harness on the same corpus and the same
104 questions. Only the pipeline changes between rows of the results table, so each gain
can be attributed to one change.

```bash
furwise eval validate                              # test set vs the current corpus
furwise eval run --stage stage0 --retrieval-only   # retrieval metrics, no LLM, free
furwise eval run --stage stage0 --judge            # answers + RAGAS faithfulness
furwise eval table                                 # results.json -> README table
```

## 1. Test set

[`eval/questions.yaml`](../eval/questions.yaml), written by hand against the label and
plant text.

| Category | n | Expected behaviour |
|---|:---:|---|
| `label_fact` | 34 | Answer from the label with citations: indications, dosing, contraindications, warnings, interactions, overdose data |
| `species_trap` | 17 | Dog-only drug asked for a cat (and the reverse), lily toxicity for cats vs dogs. The answer must state the restriction |
| `name_variant` | 14 | Misspellings (*rimadil*, *galiprant*, *nexguard*, *brevecto*), generic names, a generic copy brand (*Novox*), a genus (*Lilium*) |
| `emergency` | 13 | The answer must **start** with *"Contact your vet or a pet poison helpline"*. Six of them are about things the sources do not cover (grapes, chocolate, antifreeze, permethrin, ibuprofen, Tylenol) |
| `decline` | 14 | Human medicines, EPA pesticides, other species, vaccines, diagnosis, euthanasia, products not on DailyMed |
| `plant_fact` | 12 | ASPCA toxicity verdicts and clinical signs, including non-toxic plants |

Species: 66 dog questions, 36 cat questions, 2 with no species (other-animal declines).

**Gold evidence is stored as quotes, not chunk ids.** Each answerable question lists
`{entity, section, quote}`. At run time every chunk of that entity and section that
contains the quote is gold, so the same sentence on the tablet label and the injection
label both count, and gold survives any re-chunking. `furwise eval validate` fails as
soon as a quote stops matching or a name variant stops resolving.

## 2. Metrics

### RAGAS (0.4)

| Metric | RAGAS class | How it is computed here | Cost |
|---|---|---|---|
| Context precision | `NonLLMContextPrecisionWithReference` | Rank-aware average precision of the top 5; a retrieved chunk is relevant when it matches a gold chunk (string similarity ≥ 0.5) | no LLM |
| Context recall | `IDBasedContextRecall` | Share of gold references found in the top 5 (equivalent gold chunks share one id) | no LLM |
| Faithfulness | `Faithfulness` | Claims extracted from the answer and checked against the top-5 contexts by a free judge from a different model family | 2 judge calls per answer |

Why these variants: the context metrics compare against the test set's own gold
evidence, so they are exact and free. Precision is **rank-aware** on purpose: a reranker
mostly *reorders* candidates, and a set-based precision would not see that at all.
Faithfulness is scored on the answer's claims only; the mandated emergency line and the
`[n]` markers are removed first, since fixed policy text is not supposed to be in any
source. Declined answers have no claims and are not scored.

### Deterministic metrics (no LLM)

| Metric | Definition |
|---|---|
| Citation accuracy | Correct citations ÷ all citations, over questions with gold. A citation is correct if it is a gold chunk or from the same label section as one |
| Species mix-up ↓ | Share of dog/cat questions where the answer cites wrong-species evidence (a label not for that species that does not warn about it) **or**, for a trap, answers without stating the restriction |
| Emergency detection | Share of emergency questions whose answer starts with the required line. False alarms on the other questions are reported separately |
| Correct decline | Share of must-decline questions that are declined. Over-declining answerable questions is reported separately |
| Hit@5, MRR | Gold chunk in the top 5; 1 / rank of the first gold chunk |
| Retrieval mix-up ↓ | Share of dog/cat questions whose top 5 contain wrong-species evidence |
| Latency p50 / p95 | End-to-end per question on a laptop CPU, excluding time spent waiting on free-tier rate limits (reported separately) |

## 3. Running at zero cost

All LLM calls go to free tiers: generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b`,
both on Groq. Groq's free tier allows about 200K tokens per model per day, and a full
stage needs roughly 150K generation tokens plus ~250K judge tokens. So:

* retrieval metrics run for every stage in minutes, locally;
* answer metrics need one Groq day per stage or so;
* every reply is cached in `runs/llm_cache/`, so a run stopped by the daily limit resumes
  where it stopped and re-scoring never spends quota twice.

## 4. Harness self-checks

Unit tests (`tests/test_eval.py`) build tiny synthetic cases and check each metric:
a correct trap answer scores 0 mix-up and 1.0 MRR; a dosing answer to a cat trap that
cites the dog dosing chunk scores 1.0 mix-up; an emergency answer that starts with the
required line counts as detected even when the rest is a decline.

## 5. Reading the numbers honestly

* 104 questions are enough to separate stages, not to quote decimals. On 13 emergency
  questions one miss moves the rate by 8 points.
* All stages answer the same questions; per-question results are in
  `runs/<stage>/<timestamp>/responses.jsonl`.
* Gold and reference answers were written by one person from the label text. A second
  reviewer would make them stronger.
* Faithfulness depends on the judge model; its name is stored with every run, and stages
  are only compared under the same judge.
