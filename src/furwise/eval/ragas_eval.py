"""RAGAS metrics, run at zero cost.

* Context precision: RAGAS `NonLLMContextPrecisionWithReference`. Rank-aware average
  precision, with relevance decided by string similarity to the gold chunks instead
  of an LLM. Rank-aware matters here: a reranker mostly *reorders*, and a set-based
  precision would not see that at all.
* Context recall: RAGAS `IDBasedContextRecall`. Share of gold references found in the
  top 5. Equivalent gold chunks (the same sentence on two labels) share one id.
* Faithfulness: RAGAS `Faithfulness` (claim extraction + verification), judged by a
  free model from a different family than the generator, through `FreeJudge` below.
"""

from __future__ import annotations

import asyncio
import re

from pydantic import ValidationError
from ragas.llms.base import InstructorBaseRagasLLM

from furwise.llm import LLMClient, LLMError, extract_json
from furwise.safety import EMERGENCY_LINE

_CITE = re.compile(r"\s*\[\d{1,2}\]")


def answer_claims(answer: str) -> str:
    """The part of an answer that should be grounded: no fixed policy line, no [n] markers."""
    return _CITE.sub("", answer.replace(EMERGENCY_LINE, "")).strip()


class FreeJudge(InstructorBaseRagasLLM):
    """RAGAS's structured-output LLM interface on top of our free client. RAGAS's
    own prompts and scoring are unchanged; we only supply the model call."""

    def __init__(self, client: LLMClient | None = None, max_tokens: int = 1500):
        self.client = client or LLMClient.from_env("JUDGE")
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(1)  # one call at a time: free tiers are per-minute

    @property
    def label(self) -> str:
        return self.client.config.label

    def generate(self, prompt: str, response_model):
        schema = response_model.model_json_schema()
        messages = [
            {"role": "system", "content": "You are a careful evaluator. Reply with JSON only."},
            {
                "role": "user",
                "content": f"{prompt}\n\nReturn one JSON object matching this schema:\n{schema}",
            },
        ]
        for _ in range(3):
            reply = self.client.chat(messages, self.max_tokens, json_mode=True)
            try:
                return response_model.model_validate_json(extract_json(reply.text))
            except ValidationError as exc:
                messages += [
                    {"role": "assistant", "content": reply.text},
                    {
                        "role": "user",
                        "content": f"Invalid for the schema "
                        f"({exc.errors()[:2]}). Reply with "
                        "the corrected JSON object only.",
                    },
                ]
        raise LLMError(f"{self.label} returned no valid {response_model.__name__}")

    async def agenerate(self, prompt: str, response_model):
        async with self._sem:
            return await asyncio.to_thread(self.generate, prompt, response_model)


class RagasScorer:
    def __init__(self, judge: FreeJudge | None = None):
        from ragas.metrics import IDBasedContextRecall, NonLLMContextPrecisionWithReference

        self.precision = NonLLMContextPrecisionWithReference()
        self.recall = IDBasedContextRecall()
        self.judge = judge
        self.faithfulness = None
        if judge is not None:
            from ragas.metrics.collections import Faithfulness

            self.faithfulness = Faithfulness(llm=judge)

    async def context_scores(
        self,
        retrieved: list[str],
        gold_texts: list[str],
        retrieved_ref_ids: list[str],
        ref_ids: list[str],
    ) -> dict:
        from ragas.dataset_schema import SingleTurnSample

        p = await self.precision.single_turn_ascore(
            SingleTurnSample(retrieved_contexts=retrieved, reference_contexts=gold_texts)
        )
        r = await self.recall.single_turn_ascore(
            SingleTurnSample(retrieved_context_ids=retrieved_ref_ids, reference_context_ids=ref_ids)
        )
        return {"context_precision": round(float(p), 4), "context_recall": round(float(r), 4)}

    async def faithfulness_score(self, question: str, answer: str, contexts: list[str]) -> float:
        result = await self.faithfulness.ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        return round(float(result.value), 4) if result.value is not None else None
