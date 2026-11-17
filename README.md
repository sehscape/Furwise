<div align="center">

# 🐾 Furwise

### A species-aware RAG assistant for dog & cat medication and plant-toxicity safety

*Hybrid retrieval · BGE reranking · species-aware filtering · a verified citation on every claim · measured stage by stage*

### [🚀 Try the live demo → furwise.streamlit.app](https://furwise.streamlit.app/)

<sub>Try: [a species trap](https://furwise.streamlit.app/?q=Can%20I%20give%20my%20cat%20Rimadyl%20for%20pain%3F) · [an emergency](https://furwise.streamlit.app/?q=My%20cat%20chewed%20an%20Easter%20lily%20leaf%20and%20now%20she%20is%20vomiting.) · [Furwise vs a basic RAG bot](https://furwise.streamlit.app/?q=Can%20I%20put%20Advantage%20Multi%20for%20Dogs%20on%20my%20cat%3F&compare=1) · first load takes about a minute while the free host wakes up</sub>

[![CI](https://github.com/sehscape/Furwise----Species-Aware-RAG-Assistant-for-Veterinary-Drug-Toxicity-Safety/actions/workflows/ci.yml/badge.svg)](https://github.com/sehscape/Furwise----Species-Aware-RAG-Assistant-for-Veterinary-Drug-Toxicity-Safety/actions/workflows/ci.yml)
[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://furwise.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-ChromaDB-6E40C9)
![Models](https://img.shields.io/badge/BGE-embed%20%2B%20rerank-orange)
![Eval](https://img.shields.io/badge/eval-RAGAS%20%2B%20104%20questions-brightgreen)
![Cost](https://img.shields.io/badge/running%20cost-%240-success)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

> [!IMPORTANT]
> **For information only. Not a substitute for veterinary advice.**
> If your pet may have eaten something toxic, contact your veterinarian or a pet poison helpline right away.

<table>
<tr>
<td width="50%"><img src="docs/images/demo-species-trap.jpg" alt="Species trap: Furwise refuses to give a dog-only NSAID to a cat, citing the label"/></td>
<td width="50%"><img src="docs/images/demo-emergency.jpg" alt="Emergency: the vet / poison helpline line comes first, then cited ASPCA facts"/></td>
</tr>
<tr>
<td align="center"><sub><b>Species trap.</b> "rimadyl → Rimadyl (carprofen)", species = cat, answer cites the label's "Do not use in cats".</sub></td>
<td align="center"><sub><b>Emergency.</b> The required line is added in code, the facts are quoted from the ASPCA entry.</sub></td>
</tr>
</table>

<p align="center"><img src="docs/images/demo-compare.jpg" width="80%" alt="Live demo, compare mode: Furwise answers with the label restriction while a basic RAG bot declines"/><br/>
<sub><b>Live demo, compare mode.</b> Same question, same corpus: Furwise pins the label restriction and answers; a basic RAG bot can't find it and declines.</sub></p>

---

## 📌 Contents

[Why Furwise](#-why-furwise) · [Results at a glance](#-results-at-a-glance) · [Architecture](#-architecture) · [Species awareness](#-the-core-idea-species-awareness) · [Data](#-data-sources--processing) · [Evaluation](#-evaluation) · [What each stage changed](#-what-each-stage-changed) · [Examples](#-examples-that-worked-and-that-failed) · [Next safety features](#-two-proposed-safety-features) · [Quickstart](#-quickstart) · [Design decisions](#-design-decisions) · [Credits](#-credits--license)

---

## 💡 Why Furwise

Pet owners ask *"can I give my cat Rimadyl?"* or *"my dog chewed a sago palm, is that bad?"*.
A generic RAG chatbot answers fluently, and in veterinary safety a fluent wrong answer is
worse than no answer. Furwise is built around the specific ways those systems fail:

| How a basic RAG bot fails | Real case from this corpus | What Furwise does |
|---|---|---|
| **Species mix-up**: right drug, wrong animal | Metacam *injection* is labeled for dogs **and** cats; Metacam *oral suspension* says *"Do not use … in cats"* | Species profiled per label **and** per chunk, used as a retrieval filter |
| **A species filter that deletes the warning** | `species == cat` would remove Rimadyl's *"For use in dogs only. Do not use in cats."* | Keep chunks where the species is a **target or is warned about**, and pin the restriction |
| **Name blindness** | "rimadil", "nexguard", "brevecto", "Novox", "Lilium" | Lexicon built from the sources + fuzzy matching for drug names |
| **Confident guessing** | Tylenol, Benadryl, Seresto collars (EPA, not on DailyMed), vaccines | Emergency and scope gates run **before** any model call |
| **Unverifiable citations** | A dose number that is not in the cited label | A verifier drops any sentence its citation does not support |

---

## 📊 Results at a glance

<!-- glance:start -->
<div align="center">

| 🎯 Hit@5 | 🐶🐱 Species traps answered safely | 🧾 Faithfulness | 🚨 Emergencies caught | 🚫 Correct declines | 💸 Cost |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **69% → 87%** | **6 / 17 → 16 / 17** | **0.95** | **11 / 13 → 13 / 13** | **13 / 14 → 14 / 14** | **$0** |
| baseline → final | baseline → final | RAGAS, final pipeline | baseline → final, 0 false alarms | baseline → final | free models + free tiers |

</div>

```mermaid
xychart-beta
    title "Retrieval quality by stage (104 questions)"
    x-axis ["0 Dense", "1 + BM25", "2 + Reranker", "3 + Species filters"]
    y-axis "percent" 0 --> 100
    bar [69, 61, 72, 87]
    line [23, 27, 23, 1]
```
<sub>Bars: hit@5 (a gold passage in the top 5). Line: retrieval species mix-up (top 5 contains evidence about the wrong animal).</sub>
<!-- glance:end -->

---

## 🧩 Architecture

```mermaid
flowchart LR
    subgraph SRC["📥 Sources (free, credited)"]
        DM["DailyMed REST API<br/>3,532 animal labels"]
        AS["ASPCA plant list<br/>987 plants"]
    end

    subgraph ING["🧱 Offline corpus build"]
        P["SPL parser<br/>lxml · tables · lists"]
        S["Section normaliser<br/>LOINC → title → inline heading"]
        SP["Species profiler<br/>target vs caution"]
        D["Dedup<br/>text similarity"]
        C["Chunker<br/>contextual headers"]
        L["Lexicon<br/>brand ↔ generic<br/>common ↔ scientific"]
    end

    subgraph IDX["🗂️ Indexes"]
        V[("ChromaDB<br/>bge-small-en-v1.5")]
        B[("BM25")]
    end

    subgraph QP["🧠 Query pipeline (Stage 3)"]
        G1{"Emergency &<br/>scope gates"}
        ER["Name + species<br/>resolution"]
        HR["Hybrid retrieval<br/>RRF + metadata filters"]
        RR["BGE reranker"]
        PIN["Restriction<br/>pinning"]
        GEN["Free LLM<br/>claims + [n] citations"]
        CV["Citation<br/>verifier"]
    end

    UI["⚡ FastAPI · Streamlit"]
    EM["🚨 'Contact your vet or a<br/>pet poison helpline'<br/>prepended in code"]

    DM --> P --> S --> SP --> D --> C
    AS --> C
    D --> L
    AS --> L
    C --> V & B
    UI --> G1 --> ER --> HR --> RR --> PIN --> GEN --> CV --> UI
    G1 -. emergency .-> EM -.-> UI
    L -. aliases .-> ER
    V & B --> HR

    classDef src fill:#eef2ff,stroke:#6366f1,color:#1e1b4b
    classDef safe fill:#fef2f2,stroke:#ef4444,color:#7f1d1d
    classDef core fill:#ecfdf5,stroke:#10b981,color:#064e3b
    class DM,AS src
    class G1,CV,EM,PIN safe
    class ER,HR,RR,GEN core
```

### How one question flows

```mermaid
sequenceDiagram
    autonumber
    actor U as Pet owner
    participant G as Safety gates
    participant X as Lexicon
    participant R as Retriever + reranker
    participant M as Free LLM
    participant V as Verifier
    U->>G: "Can I give my cat rimadil for pain?"
    G->>G: not an emergency · in scope
    G->>X: resolve names and species
    X-->>G: rimadil → Rimadyl (carprofen) · species = cat
    G->>R: query + filters {entity: carprofen, species: cat}
    R-->>G: top 5 incl. "For use in dogs only. Do not use in cats." [Rimadyl · Warnings]
    G->>M: question + numbered sources + species note
    M-->>V: claims with [n] citations
    V->>V: drop sentences their citation doesn't support
    V-->>U: cited answer + sources + disclaimer
```

---

## 🎯 The core idea: species awareness

A label is *for* some species, but its text *talks about* others, mostly to forbid them.
Counting species words would call Rimadyl a cat drug. Furwise reads labels sentence by
sentence and stores two fields on every chunk:

| Field | Meaning | Rimadyl · Warnings chunk |
|---|---|---|
| `species_target` | Who the label is for: Indications → explicit use statements → Dosage → title, **no full-text guessing** | `["dog"]` |
| `caution_species` | Who this chunk warns about ("do not use in cats", "safety in kittens has not been established", or ruled out by "dogs only") | `["cat"]` |

```mermaid
flowchart TD
    Q["❓ Can I give my cat Galliprant?"] --> E["entity = grapiprant · species = cat"]
    E --> F{"keep chunk if<br/>cat ∈ species_target<br/>OR cat ∈ caution_species"}
    F -->|kept| K1["✅ Galliprant · Label Header<br/><i>For oral use in dogs only</i>"]
    F -->|dropped| K2["⛔ Galliprant · Dosage<br/>dog dosing table"]
    K1 --> P{"label restricts cats<br/>but not in top 5?"}
    P -->|pin it| A["Answer: Galliprant is labeled for oral use in dogs only [1]"]
```

Real profiles from the current build:

| Product | Form | Labeled for | Rules out | Decided by |
|---|---|---|---|---|
| Rimadyl (carprofen) | chewable tablet | 🐶 dogs | 🐱 cats | Indications + *"For use in dogs only. Do not use in cats."* |
| Metacam (meloxicam) | injection | 🐶🐱 dogs & cats | – | Indications (separate dog and cat paragraphs, chunked apart) |
| Metacam (meloxicam) | oral suspension | 🐶 dogs | 🐱 cats | *"Do not use METACAM Oral Suspension in cats."* |
| Onsior (robenacoxib) | tablet | 🐱 cats | 🐶 dogs | *"For Oral Use In Cats Only"* |
| Veraflox (pradofloxacin) | oral suspension | 🐱 cats | 🐶 dogs | *"Do not use in dogs."* |
| Advantage Multi for Dogs | topical | 🐶 dogs | 🐱 cats | *"Do not use this product (containing 2.5% moxidectin) on cats."* |
| Bravecto (fluralaner) | chew / topical | separate 🐶 and 🐱 labels | – | Indications |

> [!TIP]
> **The lily trap.** The Easter lily is *toxic to cats* but *non-toxic to dogs* on the ASPCA
> list. A single answer for "pets" is wrong for one of the two, so dog and cat toxicity are
> separate facts in every plant chunk, and the test set asks both.

---

## 📦 Data sources & processing

Only two sources, both credited. No copyrighted veterinary manuals.

| Source | Used for | Collection |
|---|---|---|
| **[DailyMed](https://dailymed.nlm.nih.gov/)** (U.S. National Library of Medicine) | FDA animal drug labels (SPL XML) | Public REST API, no key; cached; public domain |
| **[ASPCA Toxic & Non-Toxic Plant List](https://www.aspca.org/pet-care/animal-poison-control/toxic-and-non-toxic-plants)** | Dog / cat toxicity, toxic principles, clinical signs | 2 list pages + 428 detail pages, **one request every 10 s** (robots.txt crawl-delay), cached, never committed |

<!-- corpus:start -->
```mermaid
flowchart LR
    A["3,532<br/>animal labels<br/>(Rx + OTC)"] -->|"121 drug groups<br/>brand / ingredient match"| B["112 groups<br/>found"]
    B -->|"brand first, pet titles first<br/>≤ 8 per group"| C["416<br/>downloaded"]
    C -->|"−75 not for dogs/cats<br/>−162 generic copies<br/>−14 group cap"| D["165 labels<br/>106 drugs"]
    D --> E["2,025<br/>label chunks"]
    F["987 plants<br/>(dog + cat lists)"] --> G["987<br/>plant chunks"]
```

| 🏷️ Labels | 💊 Drugs | 🐶 Dog-only | 🐱 Cat-only | 🐶🐱 Both | ⛔ Rule out the other species | 🌿 Plants | 🧩 Chunks |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **165** | **106** | 95 | 28 | 42 | **79** | 987 (416 toxic to dogs · 423 to cats) | **3,012** |

```mermaid
pie showData
    title Label chunks by section (n = 2,025)
    "Adverse Reactions" : 388
    "Dosage & Administration" : 371
    "Overdose / Target Animal Safety" : 367
    "Precautions" : 213
    "Indications" : 184
    "Contraindications" : 131
    "Warnings" : 124
    "Label Header (banner)" : 122
    "Information for Owners" : 110
    "Drug Interactions" : 15
```
<!-- corpus:end -->

**Processing** (offline and deterministic once the raw files are cached):

1. **Parse SPL XML** with `lxml`: products, active ingredients, sections; tables → `cell | cell` rows, lists, line breaks.
2. **Normalise sections**: LOINC code → title rules → inline bold headings (common in animal labels) → parent section. The six required sections plus four that carry safety facts: the top-of-label **banner** ("For oral use in dogs only"), **Precautions** (where most interactions live), **Overdose / Target Animal Safety** and **Information for Owners**. Federal-law boilerplate is removed sentence by sentence, because it often shares a paragraph with "Do not use in cats".
3. **Profile species** (target vs caution, see above). Labels indicated only for other animals are dropped, even if a dosing table mentions "prairie dogs" (a real bug found and fixed).
4. **Deduplicate** generic copies by 5-word-shingle Jaccard ≥ 0.30 within the same form and species. Copies still donate their brand names (*Novox* → carprofen).
5. **Chunk** ≤ 250 words, never across sections; "Dogs:" / "Cats:" sub-headings start a new chunk; long tables keep their header row; every chunk carries a header like `Rimadyl (carprofen) tablet | for dogs | Warnings`.
6. **Lexicon**: brand ↔ generic ↔ ingredient, common ↔ scientific ↔ genus, all from the sources.
7. **Manifests**: [`data/manifest/`](data/manifest) records the exact label set id + version and every plant page used.

---

## 📈 Evaluation

### Test set: [`eval/questions.yaml`](eval/questions.yaml)

**104 hand-written questions.** Every answerable one points at the exact label or plant text
that supports it as `{entity, section, quote}`; quotes are resolved to chunks at run time, so
gold survives re-chunking, and `furwise eval validate` fails if a quote stops matching.

| Category | n | What it tests | Expected behaviour |
|---|:---:|---|---|
| 🏷️ Label facts | 34 | Indications, dosing, contraindications, warnings, interactions, overdose data | Answer with citations |
| 🪤 Species traps | 17 | Dog-only drugs asked for cats (and the reverse); Easter lily for cats vs dogs | State the restriction, citing it |
| 🔤 Name variants | 14 | *rimadil*, *galiprant*, *nexguard*, *brevecto*, *Novox*, *Lilium*, generic names | Same answer as the canonical name |
| 🚨 Emergencies | 13 | Overdoses, toxic plants, seizures; grapes, chocolate, antifreeze (not in the sources) | Start with *"Contact your vet or a pet poison helpline"* |
| 🚫 Declines | 14 | Human medicines, EPA pesticides, other species, vaccines, diagnosis | Decline, never guess |
| 🌿 Plant facts | 12 | Toxic **and** non-toxic plants, clinical signs | Answer with citations |

```mermaid
flowchart LR
    T["104 questions<br/>+ gold quotes"] --> P["Pipeline<br/>stage 0 / 1 / 2 / 3"]
    P --> R["answer · citations ·<br/>top-5 chunks · timings"]
    R --> D["Deterministic metrics<br/>citations · species mix-up ·<br/>emergency · decline · latency"]
    R --> J["RAGAS<br/>context precision & recall (no LLM)<br/>faithfulness (free judge)"]
    D & J --> S["eval/results/results.json<br/>→ this table"]
```

| Metric | How it is computed |
|---|---|
| **Context precision** | RAGAS `NonLLMContextPrecisionWithReference`: rank-aware average precision of the top 5 against the gold chunks (rank-aware on purpose: a reranker mostly *reorders*) |
| **Context recall** | RAGAS `IDBasedContextRecall`: share of gold references found in the top 5 |
| **Faithfulness** | RAGAS `Faithfulness`: answer claims checked against the retrieved text by a free judge from a *different* model family (`gpt-oss-120b`; the generator is `qwen3.8-27b`) |
| **Citation accuracy** | Cited chunk is gold, or from the same label section as a gold chunk |
| **Species mix-up ↓** | Answer cites wrong-species evidence, or misses a species trap |
| **Emergency detection** | Answer *starts with* the required line |
| **Correct decline** | Must-decline questions declined |
| **Latency p50 / p95** | End to end on a laptop CPU (Core Ultra 7 155U), excluding free-tier rate-limit waits |

Every stage runs on the **same corpus and the same questions** and adds **one layer**, so each row's change is attributable. The table is generated by `furwise eval table` from [`eval/results/results.json`](eval/results/results.json).

<!-- results:start -->
| Stage | Context precision | Context recall | Faithfulness | Citation accuracy | Species mix-up ↓ | Emergency detection | Correct decline | Latency p50 / p95 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 · Baseline (dense top-k) | 0.53 | 0.69 | 0.99 | 54% | 0% | 85% | 93% | 0.5 s / 1.1 s |
| 1 · + Hybrid (BM25 + dense, RRF) | 0.48 | 0.61 | 1.00 | 52% | 0% | 85% | 93% | 0.6 s / 1.1 s |
| 2 · + BGE reranker | 0.58 | 0.72 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 3 · + Species filters & safety | 0.74 | 0.87 | 0.95 | 63% | 0% | 100% | 100% | 3.4 s / 13.6 s |

**Retrieval and safety details**

| Stage | Hit@5 | MRR | Retrieval mix-up ↓ | Missed species traps ↓ | False emergency ↓ | Over-decline ↓ | Retrieval p50 / p95 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 · Baseline (dense top-k) | 69% | 0.51 | 23% | 0% | 3% | 26% | 0.1 s / 0.1 s |
| 1 · + Hybrid (BM25 + dense, RRF) | 61% | 0.45 | 27% | 0% | 3% | 27% | 0.1 s / 0.1 s |
| 2 · + BGE reranker | 72% | 0.59 | 23% | ⏳ | ⏳ | ⏳ | 4.2 s / 11.3 s |
| 3 · + Species filters & safety | 87% | 0.70 | 1% | 0% | 0% | 4% | 3.0 s / 13.3 s |
<!-- results:end -->

<sub>Retrieval metrics use all 104 questions. Answer metrics need a free-tier LLM; RAGAS faithfulness is judged on a fixed stratified sample of 24 questions (the same for every stage) because the free judge has a daily token cap. ⏳ = not run yet. Method and caveats: <a href="docs/evaluation.md">docs/evaluation.md</a>.</sub>

---

## 🔬 What each stage changed

<!-- stages:start -->
Each stage was evaluated before the next one was built.

#### Stage 0 · Baseline: dense retrieval, top 5, cited prompt
A strong-ish baseline on purpose: the same section-aware chunks with contextual headers as every other stage,
so later gains are not just "better chunking". It finds the right passage for **69%** of questions, but **23%**
of species questions get evidence about the wrong animal in the top 5: "can I give my cat Rimadyl?" retrieves
Rimadyl's dog dosing, because nothing tells the retriever that the cat is the point of the question.

The answers show what that costs. Its species mix-up looks perfect (0%), but only because the model **refuses**
when the evidence is wrong: it declined **26%** of answerable questions, including **10 of 17 species traps**
("Can my cat take Galliprant?" → *"I can't answer that from my sources."*), because the "dogs only" banner was
never retrieved. Its safety behaviour is prompt-only and it shows: it missed 2 of 13 emergencies (an empty reply
to a cat that licked permethrin), raised 3 false alarms, and answered a diagnosis question it should have declined.

#### Stage 1 · + BM25 hybrid with Reciprocal Rank Fusion: **worse** (69% → 61%)
Kept in the table because it is the most instructive result. BM25 finds the right **drug** every time (exact
names are its strength) but the wrong **section**: owners say *"used for"*, labels say *"indicated for"*, and very
short chunks such as a one-line Contraindications paragraph get a large BM25 length bonus. Of the 10 questions
that changed, hybrid lost 8 and won 2. Lesson: lexical search fixes *which product*, not *which part of the label*.
At the answer level nothing improved either: 5 of 17 species traps answered safely (baseline 6), 27% over-declines,
the same 2 missed emergencies and 3 false alarms. Better *lexical* recall does not give the model the evidence it
needs when the question is really about the other animal.

#### Stage 2 · + BGE cross-encoder reranker (61% → 72%, MRR 0.45 → 0.59)
The reranker reads question and passage together, so it recovers exactly what BM25 broke: it moves the
Indications paragraph above the Contraindications line for "what is X used for". The cost is real on a laptop
CPU: retrieval goes from ~0.1 s to **4.2 s p50 / 11.3 s p95**. Input was capped at 320 tokens and the pool at 12
candidates after measuring (int8 quantization was faster but reordered the top 5, so it was rejected).
Species mix-up is unchanged (23%): ranking better does not tell the system which animal the question is about.

#### Stage 3 · + Name resolution, species/entity filters, pinning, safety gates, verifier (72% → 87%)
The biggest jump, and the one that matters for safety:
* **Entity filter**: "rimadil" → Rimadyl (carprofen) restricts candidates to carprofen labels, so the reranker only
  has to choose the section.
* **Species filter** (target **or** cautioned) drops dog dosing for cat questions but keeps "Do not use in cats".
  Retrieval mix-up falls from **23% to 1%** (one question that names both a cat and a dog).
* **Restriction pinning** guarantees the "dogs only" banner is in the context for cross-species questions.
* **Gates** answer all 14 must-decline questions and the 6 emergencies outside the sources in ~1 ms, without a
  model call. **Emergency detection 13/13**, no false alarms on the other 91 questions.
* **Verifier**: every answer is cited; sentences with unsupported citations are removed. Answer-level species
  mix-up is **0%**, faithfulness **0.95**. The price is a 4% over-decline rate (3 of 77 answerable questions); two of them
  are explained under failures below.

Against the baseline, on the same 104 questions: species traps answered safely **6/17 → 16/17**, answerable
questions answered with no refusal and no species mix-up **57/77 → 74/77**, emergencies **11/13 → 13/13**, false alarms
**3 → 0**, declines **13/14 → 14/14**.

> [!NOTE]
> **Citation accuracy (63%) is scored strictly**: a citation counts only if it is the gold chunk or from the same
> label section. Many "wrong" citations are correct facts from a sibling section (e.g. Galliprant vomiting cited from
> Adverse Reactions instead of Precautions); RAGAS faithfulness (0.95) is the better measure of support.
>
> **Answer metrics for Stage 2** are still being generated on Groq's free tier (~200K tokens per model per day,
> shared with the live demo). Stages 0, 1 and 3 are complete; Stage 2 is 37/104 questions in and resumes from cache
> on the next quota window (`furwise eval run --stage stage2 --judge`). The ⏳ cells fill in with `furwise eval table`.
<!-- stages:end -->

---

## 🧪 Examples that worked and that failed

<!-- examples:start -->
Real outputs of the final pipeline (Stage 3) from the evaluation run, unedited except for
length. `[n]` points at the numbered source shown under each answer in the app.

### ✅ Worked

<details open>
<summary><b>🪤 Species trap: "Can I give my cat Rimadyl for pain?"</b></summary>

> Rimadyl should be given to dogs only, and cats should not be given Rimadyl [1]. The label explicitly states to do not use in cats [4].

**Why it worked:** the species filter kept only Rimadyl chunks that are for cats *or warn about cats*, so the
dog dosing tables never reached the model, and the Warnings chunk *"For use in dogs only. Do not use in cats."*
was in the context. The Stage 0 baseline retrieved Rimadyl's indications and dosing for dogs instead.
</details>

<details>
<summary><b>🔤 Misspelled brand: "is rimadil safe for my dogs arthritis"</b></summary>

> Rimadyl is indicated for the relief of pain and inflammation associated with osteoarthritis in dogs [1]. It is not recommended for use in dogs with bleeding disorders, as safety has not been established in these animals [5]. …

**Why it worked:** the lexicon fuzzy-matched *rimadil* → Rimadyl (carprofen) (score 0.86), and the query was
expanded with the canonical name before retrieval.
</details>

<details>
<summary><b>🌿 The lily trap, both ways</b></summary>

> **Cat:** Easter Lily is toxic to cats [1]. Clinical signs include vomiting, inappetence, lethargy, kidney failure, and death is possible [1]. Cats are the only species known to be affected [1].
>
> **Dog:** Easter Lily is non-toxic to dogs [1].

**Why it worked:** plant chunks keep dog and cat verdicts as separate ASPCA facts, so the same entry gives
opposite, correct answers per species.
</details>

<details>
<summary><b>🚨 Emergency with label evidence: "My dog chewed up part of a sago palm."</b></summary>

> Contact your vet or a pet poison helpline right away. Sago Palm is toxic to dogs [1]. The toxic principle is cycasin [1]. Clinical signs include vomiting, melena, icterus, increased thirst, hemorrhagic gastroenteritis, bruising, coagulopathy, liver damage, liver failure, death [1].

**Why it worked:** the emergency gate fired on "chewed up" and the required line was prepended in code;
the rest is quoted from the ASPCA entry. The app also shows the ASPCA Poison Control number printed on that page.
</details>

<details>
<summary><b>🚨 Emergency outside the sources: "My dog just ate a handful of grapes"</b></summary>

> Contact your vet or a pet poison helpline right away. I can't answer that from my sources: I could not find that medicine or plant in my sources (FDA animal drug labels on DailyMed and the ASPCA plant list). …

**Why it worked:** grapes are in neither source. Furwise still escalates, but refuses to invent toxicity
details. No model call was made (0 ms generation).
</details>

<details>
<summary><b>🚫 Decline: "Can I give my dog Benadryl for his allergies?"</b></summary>

> I can't answer that from my sources: it is a human or unlabeled medicine with no FDA animal label in DailyMed. …

**Why it worked:** the scope gate recognised a human medicine with no animal label and declined before
retrieval. All 14 must-decline questions were declined this way, in ~1 ms each.
</details>

<details>
<summary><b>💊 Interaction stated in the label: "Can my dog take Rimadyl together with a corticosteroid like prednisone?"</b></summary>

> Rimadyl should not be given with steroids (for example, cortisone, dexamethasone, prednisone, triamcinolone) [3]. Consider appropriate washout times when switching from corticosteroid use to NSAID use [1].

**Why it worked:** both sentences are quoted from the Rimadyl label (owner information and Precautions).
The verifier only lets interaction claims through when the cited label section states an interaction.
</details>

### ❌ Failed (and why)

<details open>
<summary><b>Over-decline on a generic-copy brand: "Is Novox safe for cats?"</b></summary>

> I can't answer that from my sources: the label and plant text I retrieved does not clearly answer that. …

**Why it failed:** *Novox* correctly resolved to carprofen and the retrieved chunks contained *"Do not use in
cats"*, but they say **Rimadyl**, not Novox. The model could not connect the two names and returned nothing.
Safe, but unhelpful. **Fix:** add the resolved name mapping ("Novox is a carprofen product; the sources are
carprofen labels") to the prompt, the same way the species note is added.
</details>

<details>
<summary><b>Unfaithful paraphrase: "Can I put Advantage Multi for Dogs on my cat?"</b></summary>

> No, Advantage Multi for Dogs is indicated for dogs only [1].

**Why it failed:** the conclusion is right, but the label actually says *"Do not use this product (containing
2.5% moxidectin) on cats"*, and the cited chunk was an adverse-reactions passage. RAGAS faithfulness scored it
0.0, and it passed the verifier because the word overlap was high. This is the case where an LLM judge catches
what a lexical verifier cannot.
</details>

<details>
<summary><b>Verifier too strict: "Can I give my dog Onsior tablets?"</b></summary>

> I can't answer that from my sources: …

**Why it failed:** the model's only sentence had no `[n]` citation, so the verifier removed it and the answer
became a decline. The banner *"For Oral Use In Cats Only"* was retrievable. Dropping uncited text is the right
default for safety, at the cost of 4% over-declines.
</details>

<details>
<summary><b>Two animals in one question: "My cat has arthritis. Can I use the Metacam oral suspension I have for my dog?"</b></summary>

The answer was correct (*"Do not give METACAM Oral Suspension to cats"*), but retrieval let in a dog-only dosing
chunk: the question mentions both **cat** and **dog**, so the species filter allowed both. It is the only
retrieval mix-up left in Stage 3. **Fix:** decide which animal is the *patient* ("my cat has arthritis") rather
than filtering on every species mentioned.
</details>
<!-- examples:end -->

---

## 🛡️ Two proposed safety features

<!-- features:start -->
Chosen from what the evaluation exposed, not from a wish list. Each one reuses data Furwise already has.

| # | Feature | What it does | Why this one |
|:---:|---|---|---|
| 1 | **Age & weight eligibility check** | Extract the pet's age or weight from the question ("my 10-week-old kitten", "a 3 lb puppy") and compare it, in code, with the minimums stated on the label (*"8 weeks of age and older, and weighing 2.0 pounds or greater"*), citing that line | 7 of the 34 label questions are about a minimum age, and they are a classic owner mistake (a puppy given an adult flea chew). Today the LLM reads the minimum correctly, but nothing *guarantees* it is compared with the pet; a deterministic check makes it a rule, like the emergency line |
| 2 | **Clarify ambiguous products before answering** | When a name maps to products with different species or verdicts (Advantage Multi **for Dogs** vs **for Cats**, Bravecto chew vs topical, "lily" → true lilies that are deadly to cats vs peace and calla lilies that are not), ask one short question instead of answering | Several of the failures above involve product variants (Advantage Multi, Metacam suspension vs injection, Onsior tablet vs injection), and the lexicon already knows when a name is ambiguous. Asking costs one turn; mixing up a 2.5% moxidectin dog product with the cat version can kill a cat |
<!-- features:end -->

---

## 🚀 Quickstart

```bash
git clone https://github.com/sehscape/Furwise----Species-Aware-RAG-Assistant-for-Veterinary-Drug-Toxicity-Safety.git furwise
cd furwise
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . --no-deps
cp .env.example .env                                  # paste a free Groq key
```

```bash
furwise fetch dailymed      # ~5 min, cached under data/raw/
furwise fetch aspca         # ~75 min the first time (10 s crawl delay), resumable
furwise build               # offline, ~30 s -> data/processed/ + data/manifest/
furwise index               # embeds chunks into ChromaDB (incremental after the first run)
furwise ask "Can I give my cat Rimadyl?"
```

```bash
streamlit run app/streamlit_app.py           # demo UI, with a "compare with basic RAG" toggle
uvicorn furwise.api:app --port 8000          # REST API: /ask, /lookup, /health
furwise eval run --stage stage3 --judge      # evaluate a stage (cached, resumable)
pytest                                       # 45 unit tests, no network or models needed
```

### ☁️ Free hosting: Streamlit Community Cloud, lite mode ([live](https://furwise.streamlit.app/))

Free app hosts give about **1 GB of RAM**. The full Stage 3 pipeline needs ~2.6 GB (the cross-encoder alone is
~1.1 GB) and takes 3–14 s per question on a shared CPU, so the hosted demo runs **`stage3_lite`**: Stage 3 with every
safety layer but **without the reranker**. Measured on the same 104 questions:

| | Full Stage 3 (local) | Hosted lite | Baseline |
|---|:---:|:---:|:---:|
| Hit@5 | 87% | **78%** | 69% |
| Retrieval species mix-up | 1% | **3%** | 23% |
| Retrieval latency p50 | 3.0 s | **0.04 s** | 0.1 s |
| Memory (app + compare mode) | ~2.6 GB | **~0.85 GB** | – |

Emergency detection and declines are identical in both, because those gates run before retrieval.

```bash
python -m furwise.remote push <hf-user>/furwise-index   # prebuilt corpus + index -> private HF dataset (free)
```
Then on [share.streamlit.io](https://share.streamlit.io): **New app** → this repo → main file `app/streamlit_app.py`,
Python 3.11, and these secrets:

```toml
GROQ_API_KEY = "..."
HF_TOKEN = "..."                       # read access to the private dataset
FURWISE_DATA_REPO = "<hf-user>/furwise-index"
FURWISE_DEMO_STAGE = "stage3_lite"
```

The dataset is private, so processed ASPCA content is not republished; every answer links to the original page.
The [`Dockerfile`](Dockerfile) runs the full pipeline anywhere with ~3 GB of RAM.

<details>
<summary><b>🧭 Project structure</b></summary>

```text
Furwise/
├── app/streamlit_app.py         # demo UI (in-process pipeline, compare mode)
├── configs/drugs.yaml           # names only: what counts as a "common" dog/cat drug
├── data/manifest/               # committed: exact label versions + plant pages + stats
├── docs/                        # architecture decisions, evaluation method
├── eval/
│   ├── questions.yaml           # 104 questions with gold quotes
│   └── results/                 # results.json + generated table
├── scripts/finish_eval.bat      # resume the remaining evaluation runs
├── src/furwise/
│   ├── ingest/                  # polite HTTP client, DailyMed, SPL parser, ASPCA
│   ├── corpus/                  # sections, species, chunker, names, build
│   ├── retrieval/               # Chroma + BM25 indexes, search (RRF, filters, reranker)
│   ├── eval/                    # dataset, deterministic metrics, RAGAS, runner
│   ├── safety.py                # emergency + scope gates, citation verifier
│   ├── generate.py              # cited JSON answers, species note
│   ├── llm.py                   # free LLM client (Groq / Gemini / Ollama), cache, pacing
│   ├── pipeline.py              # stage 0-3
│   ├── api.py                   # FastAPI
│   ├── remote.py                # ship the prebuilt index to a hosted demo
│   └── cli.py                   # furwise fetch | build | index | ask | eval
├── tests/                       # synthetic fixtures only
└── Dockerfile                   # full pipeline in a container
```
</details>

---

## 🧠 Design decisions

Full write-up with alternatives considered: **[docs/architecture.md](docs/architecture.md)**.

| Decision | Why |
|---|---|
| Section-aware chunks with a contextual header | "Do not use in cats." means nothing unless the chunk says which drug it belongs to |
| Species stored as *target* vs *caution* | A plain species filter deletes the most important warnings |
| Restriction pinning | The most *similar* chunk is rarely the one that says "dogs only" |
| Safety in code, not in the prompt | The emergency line and declines must be guaranteed, not "usually" there |
| Gates before retrieval | Out-of-scope questions cost nothing and cannot be hallucinated |
| Citation verifier | Unsupported sentences, invented numbers and unstated interactions are removed |
| `bge-small` + `bge-reranker-base` on CPU | Free and local; reranker input capped at 320 tokens after measuring latency |
| RRF for hybrid fusion | No weights to tune between BM25 and cosine scores |
| Same corpus for every stage | Gains in the table come from the pipeline, not from re-chunking |
| Different model family as RAGAS judge | Reduces self-preference; every safety metric is deterministic anyway |
| Free models only, cached | $0 to run; evaluation reruns never spend quota twice |

---

## 🙏 Credits & license

- Drug label content: **DailyMed**, U.S. National Library of Medicine (public domain).
- Plant toxicity content: **ASPCA Animal Poison Control Center**, *Toxic and Non-Toxic Plant List*,
  collected lightly under its robots.txt and linked from every answer. Raw pages are not committed.
- Models: BAAI `bge-small-en-v1.5` and `bge-reranker-base`; generation and judging on Groq's free tier.
- Code: [MIT](LICENSE) © Sehal Chodankar.

<div align="center">

**For information only. Not a substitute for veterinary advice.**

</div>
