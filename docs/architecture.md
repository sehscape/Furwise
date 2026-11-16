# Architecture and design decisions

Furwise answers questions like *"can I give my cat Rimadyl?"* or *"my dog chewed a sago
palm"* from two sources only: FDA animal drug labels on DailyMed and the ASPCA Toxic and
Non-Toxic Plant List. A fluent wrong answer is worse than no answer here, so most of the
design is about **when not to answer**, **which animal the evidence is about**, and
**making every claim checkable**.

This page explains each decision and the alternative it beat. Numbers come from the
evaluation in [evaluation.md](evaluation.md) and the results table in the README.

---

## 1. Pipeline at a glance

```mermaid
flowchart LR
    Q([Question]) --> G{Emergency and<br/>scope gates}
    G -- out of scope --> D[Templated decline]
    G -- in scope --> N[Name + species<br/>resolution]
    N --> H[Hybrid retrieval<br/>dense + BM25, RRF<br/>entity and species filters]
    H --> R[BGE reranker]
    R --> P[Restriction pinning]
    P --> L[LLM: claims with [n] citations]
    L --> V[Citation verifier]
    V --> A([Answer + sources + disclaimer])
    G -. emergency .-> E[Prepend: Contact your vet or<br/>a pet poison helpline right away.]
    E -.-> A
```

Each box is one module:

| Step | Module | What it does |
|---|---|---|
| Gates | `safety.py` | Emergency detection (rules, high recall) and the scope gate (human medicines, other species, diagnosis, topics outside the sources) |
| Resolution | `corpus/names.py` | Brand ↔ generic, common ↔ scientific, misspellings, plurals; dog/cat detection |
| Retrieval | `retrieval/search.py` | bge-small dense search in ChromaDB + BM25, Reciprocal Rank Fusion, metadata filters |
| Reranking | `retrieval/search.py` | `BAAI/bge-reranker-base` cross-encoder over the top 12 fused candidates |
| Pinning | `pipeline.py` | If a label rules out the asked species, that restriction is always in the context |
| Generation | `generate.py` | JSON answer whose sentences cite numbered sources |
| Verification | `safety.py` | Drops sentences without a supporting citation, with numbers that are not in the cited text, or with interaction claims the label does not make |

---

## 2. Data decisions

### 2.1 Only two sources, both credited
DailyMed SPL labels are public domain U.S. government documents. The ASPCA list is
scraped lightly: two list pages plus detail pages for plants that are toxic to a dog or
a cat, one request every 10 seconds as `robots.txt` asks, cached forever, never
committed. No copyrighted veterinary manuals are used.

### 2.2 Choosing "common dog and cat drugs"
[`configs/drugs.yaml`](../configs/drugs.yaml) lists 121 drug groups by **name only**
(ingredients and brand names, no facts). A DailyMed label joins the most specific group
(Heartgard Plus → ivermectin + pyrantel, not ivermectin). Downloads are ranked
original-brand first, then pet-looking titles before livestock ones: without that rule,
ivermectin's download budget went to cattle pour-ons that the species filter then threw
away.

### 2.3 Sections
Animal labels are messier than human ones: most sections are coded *SPL UNCLASSIFIED* and
the real heading is a bold phrase inside a paragraph (`<b>WARNINGS:</b>`). The section
classifier tries, in order: LOINC code → title keywords → inline bold / ALL-CAPS heading
→ parent section. Besides the six required sections (Indications, Dosage,
Contraindications, Warnings, Adverse Reactions, Drug Interactions) four more are kept
because they carry safety facts:

| Extra section | Why |
|---|---|
| `label_header` | The top-of-label banner, e.g. *"For oral use in dogs only"*. For several drugs it is the **only** place the other species is ruled out |
| `precautions` | Most interaction statements live here, not in a formal Drug Interactions section (only 15 chunks come from one) |
| `overdose_safety` | Target animal safety and overdose studies: the evidence for "my dog ate three tablets" |
| `owner_info` | Plain-language side-effect lists written for owners |

Boilerplate (*"Federal law restricts this drug..."*) is removed **sentence by sentence**,
never block by block, because it often shares a paragraph with *"Do not use in cats."*

### 2.4 Species profiles: target vs caution
This is the core idea. A dog NSAID label mentions cats mostly to forbid them, so counting
species words would call Rimadyl a cat drug. Every sentence is classified as a *positive*
mention ("indicated ... in dogs"), a *caution* mention ("do not use in cats", "safety in
kittens has not been evaluated") or an *only* statement ("for oral use in dogs only").

* **Label target species** come from an evidence ladder: Indications → explicit use
  statements → Dosage → title. There is deliberately **no full-text fallback**: a label
  that cannot be placed is excluded rather than guessed (75 were, mostly livestock or other species).
* An *only* statement narrows the target but never removes a species the Indications
  explicitly name. When the two disagree, the label is kept with both and flagged
  (5 labels).
* Each **chunk** stores `species_target` (who the label is for) and `caution_species`
  (who this chunk warns about, including species ruled out by an *only* statement).

### 2.5 Deduplication
DailyMed has many near-identical generic copies (dozens of carprofen labels). Copies fill
the top-k with the same text, so a candidate is dropped when it has the same dosage form
and species as a kept label **and** a 5-word-shingle Jaccard similarity ≥ 0.30. Copies
still donate their brand names to the lexicon, so "Novox" resolves to carprofen without
indexing Novox's duplicate text. At most 3 labels are kept per group.

### 2.6 Chunking
Section-aware, ≤ 250 words, never across sections. "Dogs:" / "Cats:" sub-headings start a
new chunk, so a dog dose and a cat dose never share one. Long tables are split by rows
with the header repeated. Every chunk gets a contextual header,
`Rimadyl (carprofen) tablet, chewable | for dogs | Warnings`, which is embedded with it:
a chunk that only says *"Do not use in cats."* is meaningless without it.

---

## 3. Retrieval decisions

| Choice | Alternative | Why this one |
|---|---|---|
| `BAAI/bge-small-en-v1.5` embeddings | bge-base / bge-large, API embeddings | 33M parameters: the full corpus embeds in ~13 min on a laptop CPU, queries in ~50 ms. Free and local |
| ChromaDB (persistent) | FAISS, pgvector | Metadata filters built in, zero ops, a folder on disk |
| BM25 next to dense | dense only | Drug names, doses and label jargon are exact-match problems ("Galliprant", "0.09 mg/lb") |
| Reciprocal Rank Fusion | weighted score fusion | No tuning, robust to the very different score scales of BM25 and cosine |
| `BAAI/bge-reranker-base` | no reranker, bigger rerankers | Cross-encoders read query and passage together. Capped at 320 tokens and the top 12 candidates because it costs ~0.35 s per passage on this CPU (measured; int8 quantization was faster but reordered the top 5, so it was rejected) |
| Filters in Python on an over-fetched list | filters inside each index | Identical semantics for dense and BM25, exact, and cheap at ~3k chunks |
| LlamaIndex as a thin layer | a full LlamaIndex query engine | Its vector store, embeddings and BM25 retriever are used; fusion, filters and safety are plain code because they need full control |

### 3.1 The species filter that keeps warnings
A naive `species == cat` filter would **delete** *"For use in dogs only. Do not use in
cats."* from the Rimadyl label, which is exactly the answer to *"can I give my cat
Rimadyl?"*. Furwise keeps a chunk when the asked species is a target of its label **or**
the chunk warns about that species.

### 3.2 Restriction pinning
The most similar chunk to *"can I give my cat Galliprant for pain?"* is the Indications
paragraph about pain, not the banner that says *"For oral use in dogs only"*. So when a
resolved drug has chunks that rule out the asked species, the best of them is pinned into
the context. Retrieval similarity decides what is *relevant*; the pin decides what is
*necessary*.

---

## 4. Generation and safety decisions

| Decision | Why |
|---|---|
| Safety in code, not in the prompt | "Start with: contact your vet" in a prompt is a suggestion. The emergency line is prepended by code, so it is always there and always first |
| Gates before retrieval | Questions about Benadryl, horses or vaccines are declined before any model call. Cheaper, faster, and nothing can be hallucinated |
| JSON answer with numbered citations | Citations become ids that can be checked mechanically against what was retrieved |
| Citation verifier | Sentences need ≥ 35% content-word overlap with the cited text, every dose number must literally appear in it, and interaction claims must cite a label section that states an interaction. Anything else is removed; if nothing survives, the answer becomes a decline |
| Species note in the prompt | Built from label metadata, e.g. "[2] is from a label for dogs only and its text warns about cats". The model is told which evidence is about the other animal |
| Doses quoted, never computed | The prompt and the verifier forbid numbers that are not in the label |
| A different model family as judge | Generator `qwen/qwen3.8-27b`, RAGAS judge `openai/gpt-oss-120b`, both on Groq's free tier, so the judge is not grading its own style |
| Free models only | Groq free tier for generation and judging; Ollama works offline. LLM replies are cached on disk, so evaluation reruns cost nothing |

---

## 5. What was rejected

* **A knowledge graph of drug interactions.** The labels state very few interactions
  explicitly; a graph would mostly encode things the sources do not say.
* **Query rewriting with an LLM.** Name resolution is deterministic and testable; an LLM
  rewrite adds latency and a new way to hallucinate a drug name.
* **Bigger embedding and reranker models.** Measured CPU latency already dominates; the
  evaluation shows where retrieval fails, and it is not model size.
* **Fuzzy matching for plant names.** It turned "garden" into "gardenia". Plants use exact
  and plural matching only.

---

## 6. Hosting

The Streamlit app runs the whole pipeline in-process. The plan was a free Hugging Face
Space, but Docker and Gradio Spaces on the free CPU tier now require a paid plan, so the
demo runs on Streamlit Community Cloud's free tier instead, which allows about 1 GB of RAM.

That limit forced a measured trade-off. The full Stage 3 pipeline peaks at ~2.6 GB, and the
bge-reranker accounts for ~1.1 GB of that. The hosted demo therefore runs `stage3_lite`,
which is Stage 3 without the reranker: ~0.85 GB including compare mode (one embedding model
is shared by both pipelines), 40 ms retrieval, hit@5 78% against 87% for the full pipeline
and 69% for the baseline. The safety gates run before retrieval, so they behave the same.

Building the corpus means ~75 minutes of polite crawling and ~13 minutes of embedding, so the
host never builds it. `furwise.remote` pushes the locally built corpus and index to a
**private** Hugging Face dataset (free), and the app downloads it at start-up with a read
token. The [`Dockerfile`](../Dockerfile) runs the full pipeline on any machine with ~3 GB of
RAM, and the FastAPI service exposes the same pipeline for programmatic use.
