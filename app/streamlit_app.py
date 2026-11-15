"""Furwise demo UI.

    streamlit run app/streamlit_app.py

Runs the pipeline in-process (no separate API needed), so it can be hosted for free
as a single Hugging Face Space. Optional side-by-side mode shows what a plain RAG
bot (Stage 0) would have answered to the same question.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import streamlit as st

from furwise import DISCLAIMER
from furwise.api import POISON_CONTROL
from furwise.llm import LLMError
from furwise.pipeline import STAGES, Pipeline, Response
from furwise.settings import EVAL_DIR

st.set_page_config(page_title="Furwise · pet medication & toxicity", page_icon="🐾", layout="wide")

EXAMPLES = {
    "🪤 Species traps": [
        "Can I give my cat Rimadyl for pain?",
        "Can I put Advantage Multi for Dogs on my cat?",
        "Can I give Veraflox to my dog?",
    ],
    "🚨 Emergencies": [
        "My cat chewed an Easter lily leaf and now she is vomiting.",
        "My dog ate a whole bottle of Rimadyl chewables. What should I do?",
    ],
    "🔤 Name variants": [
        "is rimadil safe for my dogs arthritis",
        "Are Lilium plants toxic to cats?",
        "how often do i give nexguard to my dog",
    ],
    "🚫 Out of scope": [
        "Can I give my dog Benadryl for his allergies?",
        "What vaccines does my new puppy need?",
    ],
}

CSS = """
<style>
.fw-hero h1 {margin-bottom: 0;}
.fw-hero p {color: #64748b; margin-top: .25rem;}
.fw-card {border-radius: 14px; padding: 1rem 1.2rem; margin: .4rem 0 1rem 0;
          border: 1px solid rgba(100,116,139,.25); line-height: 1.6;}
.fw-ok {border-left: 6px solid #10b981;}
.fw-emergency {border-left: 6px solid #ef4444; background: rgba(239,68,68,.06);}
.fw-decline {border-left: 6px solid #f59e0b; background: rgba(245,158,11,.06);}
.fw-cite {font-size: .75em; font-weight: 600; color: #2563eb; vertical-align: super;}
.fw-chip {display: inline-block; padding: .1rem .55rem; margin: 0 .3rem .3rem 0;
          border-radius: 999px; font-size: .8rem; background: rgba(37,99,235,.1); color: #1d4ed8;}
.fw-muted {color: #64748b; font-size: .85rem;}
</style>
"""


@st.cache_resource(show_spinner="Loading the corpus and models (first run only)...")
def pipeline(stage: str) -> Pipeline:
    return Pipeline(stage)


def render_answer(text: str) -> str:
    safe = html.escape(text)
    return re.sub(r"\[(\d{1,2})\]", r'<span class="fw-cite">[\1]</span>', safe)


def answer_card(resp: Response, title: str) -> None:
    kind = "fw-emergency" if resp.emergency else "fw-decline" if resp.declined else "fw-ok"
    st.markdown(f"**{title}**")
    st.markdown(
        f'<div class="fw-card {kind}">{render_answer(resp.answer)}</div>', unsafe_allow_html=True
    )
    if resp.emergency:
        st.error(f"🚨 {POISON_CONTROL}, or your nearest emergency vet.")


def sources(resp: Response) -> None:
    if not resp.contexts:
        st.caption("No sources were retrieved for this question.")
        return
    for n, c in enumerate(resp.contexts, 1):
        cited = "✅ cited" if c["chunk_id"] in resp.citations else "· retrieved"
        name = c["entity_name"] if c["source"] == "ASPCA" else c["product"]
        with st.expander(f"[{n}] {name} · {c['section_name']} · {c['source']}  {cited}"):
            st.caption(c["header"])
            st.text(c["text"][:1500])
            st.markdown(f"[Open the source ↗]({c['url']})")


def trace(resp: Response) -> None:
    chips = "".join(
        f'<span class="fw-chip">{html.escape(m["text"])} → '
        f"{html.escape(m['name'])} ({m['kind']})</span>"
        for m in resp.entities
    )
    species = "".join(
        f'<span class="fw-chip">{"🐶" if s == "dog" else "🐱"} {s}</span>' for s in resp.species
    )
    st.markdown(
        f"**Recognised:** {chips or '—'} &nbsp; **Species:** {species or '—'}",
        unsafe_allow_html=True,
    )
    pinned = resp.meta.get("pinned") or []
    if pinned:
        st.markdown("📌 A label restriction for this species was pinned into the context.")
    dropped = resp.meta.get("dropped") or []
    if dropped:
        st.markdown("🧹 **Removed by the citation verifier:**")
        for d in dropped:
            st.markdown(
                f"- ~~{html.escape(d['sentence'])}~~ <span class='fw-muted'>"
                f"({html.escape(d['reason'])})</span>",
                unsafe_allow_html=True,
            )
    t = resp.timings
    st.markdown(
        f"<span class='fw-muted'>retrieval {t.get('retrieval_ms', 0):.0f} ms · "
        f"generation {t.get('generation_ms', 0):.0f} ms · "
        f"model {resp.meta.get('model', '—')}</span>",
        unsafe_allow_html=True,
    )


def results_table() -> None:
    path = EVAL_DIR / "results" / "results.md"
    if path.exists():
        st.markdown(path.read_text("utf-8"))
    else:
        st.caption("Run `furwise eval run` to fill the results table.")


def sidebar() -> bool:
    with st.sidebar:
        st.markdown("### 🐾 Furwise")
        st.markdown(
            "Species-aware answers about **dog & cat medicines** and **toxic plants**, "
            "with a citation on every claim."
        )
        compare = st.toggle(
            "Compare with a basic RAG bot",
            value=False,
            help="Also answer with Stage 0 (dense retrieval + LLM, no safety "
            "layer) to see what Furwise adds.",
        )
        st.markdown("---")
        st.markdown("**Sources**")
        st.markdown(
            "- [DailyMed](https://dailymed.nlm.nih.gov/) FDA animal drug labels "
            "(U.S. National Library of Medicine)\n"
            "- [ASPCA Toxic and Non-Toxic Plant List](https://www.aspca.org/pet-care/"
            "animal-poison-control/toxic-and-non-toxic-plants)"
        )
        stats = Path(__file__).resolve().parents[1] / "data" / "manifest" / "stats.json"
        if stats.exists():
            s = json.loads(stats.read_text("utf-8"))
            st.markdown(
                f"{s['labels']} labels · {s['drug_groups']} drugs · {s['plants']} "
                f"plants · {s['chunks']:,} chunks"
            )
        st.markdown("---")
        st.caption(DISCLAIMER)
    return compare


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    compare = sidebar()
    st.markdown(
        '<div class="fw-hero"><h1>🐾 Furwise</h1><p>Ask about a dog or cat medicine, '
        "or a plant your pet got into. Answers come only from FDA drug labels and the "
        "ASPCA plant list.</p></div>",
        unsafe_allow_html=True,
    )
    st.warning(
        f"**{DISCLAIMER}** If your pet may have eaten something toxic, contact your "
        "veterinarian or a pet poison helpline right away.",
        icon="⚠️",
    )

    tab_ask, tab_results = st.tabs(["Ask", "Evaluation results"])
    with tab_ask:
        cols = st.columns(len(EXAMPLES))
        for col, (group, questions) in zip(cols, EXAMPLES.items(), strict=True):
            with col:
                st.caption(group)
                for q in questions:
                    if st.button(q, key=q, use_container_width=True):
                        st.session_state["question"] = q

        question = st.text_input(
            "Your question", key="question", placeholder="e.g. Can I give my cat Rimadyl?"
        )
        if not question:
            return
        try:
            if compare:
                left, right = st.columns(2)
                with left:
                    resp = pipeline("stage3").answer(question)
                    answer_card(resp, "🐾 Furwise (Stage 3)")
                    trace(resp)
                    sources(resp)
                with right:
                    base = pipeline("stage0").answer(question)
                    answer_card(base, f"🤖 Basic RAG ({STAGES['stage0']['label']})")
                    sources(base)
            else:
                resp = pipeline("stage3").answer(question)
                answer_card(resp, "Answer")
                trace(resp)
                st.markdown("#### Sources")
                sources(resp)
        except LLMError as exc:
            st.error(f"The language model is not available right now: {exc}")
    with tab_results:
        results_table()


main()
