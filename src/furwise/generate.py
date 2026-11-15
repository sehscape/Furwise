"""Answer generation with numbered, checkable citations.

The model sees the retrieved chunks as [1]..[k] and returns JSON:
    {"answer": "... [2] ...", "citations": [2]}
Numbers map back to chunk ids, so every citation can be checked against what was
actually retrieved. Stages 0-2 share one plain prompt (only retrieval changes
between them). Stage 3 adds a short, code-generated species note built from label
metadata, and its safety behaviour is enforced in code, not by the prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from furwise.llm import LLMClient, extract_json
from furwise.safety import DECLINE_PREFIX, EMERGENCY_PREFIX

BASE_PROMPT = f"""You answer pet owners' questions about dog and cat medicines and plant toxicity.
Use ONLY the numbered sources: FDA animal drug labels (DailyMed) and the ASPCA plant list.

Rules:
- State only facts written in the sources. No outside knowledge, no doses that are not in the sources.
- Put a citation like [2] at the end of every sentence.
- Mind the species: a dog label does not cover cats, and the reverse.
- If the sources do not answer the question, start with exactly: "{DECLINE_PREFIX}."
- If the pet may have been poisoned, overdosed or is unwell after an exposure, start with exactly: "{EMERGENCY_PREFIX} right away."
- At most 4 short sentences.

Reply with JSON only: {{"answer": "<answer with [n] citations>", "citations": [<numbers used>]}}"""

SAFE_PROMPT = """You answer pet owners' questions about dog and cat medicines and plant toxicity.
Use ONLY the numbered sources: FDA animal drug labels (DailyMed) and the ASPCA plant list.

Rules:
- State only facts written in the sources. No outside knowledge. Quote doses exactly as written; never calculate a dose for a specific pet.
- Put a citation like [2] at the end of every sentence. Cite the source that states the fact.
- Read the species note. If a label is not for the pet's species, or warns against it, say that first and cite it.
- Mention a drug interaction only if a source states it.
- Do not add emergency or "ask your vet" advice; the app adds it.
- If the sources do not answer the question, reply with an empty answer.
- At most 4 short sentences.

Reply with JSON only: {"answer": "<answer with [n] citations>", "citations": [<numbers used>]}"""

_MARKER = re.compile(r"\[(\d{1,2})\]")


def format_sources(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {c['header']}\n{c['text']}" for i, c in enumerate(chunks, 1))


def species_note(chunks: list[dict], species: list[str]) -> str:
    """Plain statements read from label metadata, e.g.
    'The question is about a cat. [1] is from a label for dogs only that warns about cats.'"""
    if not species:
        return ""
    lines = [f"The question is about a {' and a '.join(species)}."]
    for i, c in enumerate(chunks, 1):
        if c["source"] != "DailyMed":
            continue
        target = c["species_target"]
        for s in species:
            if s not in target:
                warn = " and its text warns about " + s + "s" if s in c["caution_species"] else ""
                lines.append(
                    f"[{i}] is from a label for {' and '.join(t + 's' for t in target)} only{warn}."
                )
                break
    return " ".join(lines)


@dataclass
class Generated:
    answer: str
    numbers: list[int]
    meta: dict = field(default_factory=dict)
    ms: float = 0.0


class Generator:
    def __init__(self, client: LLMClient | None = None, max_tokens: int = 700):
        self.client = client or LLMClient.from_env("GEN")
        self.max_tokens = max_tokens

    def answer(
        self,
        question: str,
        chunks: list[dict],
        species: list[str] | None = None,
        safe: bool = False,
    ) -> Generated:
        system = SAFE_PROMPT if safe else BASE_PROMPT
        user = f"Sources:\n\n{format_sources(chunks)}\n\n"
        if safe and species:
            user += f"Species note: {species_note(chunks, species)}\n\n"
        user += f"Question: {question}"
        reply = self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            self.max_tokens,
            json_mode=True,
        )
        answer, numbers, ok = parse_reply(reply.text)
        return Generated(
            answer=answer,
            numbers=[n for n in numbers if 1 <= n <= len(chunks)],
            meta={
                "model": self.client.config.label,
                "json_ok": ok,
                "cached": reply.cached,
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "wait_ms": round(reply.wait_ms),
            },
            ms=reply.ms,
        )


def parse_reply(text: str) -> tuple[str, list[int], bool]:
    """(answer, cited numbers, JSON contract followed). Falls back to the raw text and
    its [n] markers, so a model that ignores the format loses nothing silently."""
    try:
        data = json.loads(extract_json(text))
        answer = str(data.get("answer", "")).strip()
        numbers = [int(n) for n in data.get("citations", []) if str(n).strip().isdigit()]
        numbers += [int(n) for n in _MARKER.findall(answer)]
        return answer, list(dict.fromkeys(numbers)), True
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        return text.strip(), list(dict.fromkeys(int(n) for n in _MARKER.findall(text))), False
