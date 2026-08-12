"""Judge verifier — a stronger model grades the answer.

The most reliable gate and the most expensive: it spends a call on a *higher*
tier than the one that produced the answer. Use it sparingly, typically only on
the penultimate tier where a false accept is costly. Returns a rubric-scored
confidence.
"""

from __future__ import annotations

import re

from cascade.llm import LLM
from cascade.types import LLMResponse, Query, Verdict
from cascade.verifiers.base import Verifier

_SYSTEM = (
    "You are an expert grader. Score the candidate answer's correctness from 0 to 10. "
    "Reply as: 'SCORE: <0-10>, REASON: <short>'."
)
_SCORE_RE = re.compile(r"SCORE:\s*(\d{1,2}(?:\.\d+)?)", re.IGNORECASE)


class JudgeVerifier(Verifier):
    """Grade with a stronger judge model."""

    name = "judge"

    def __init__(self, judge_model: LLM, accept_score: float = 7.0) -> None:
        self.judge = judge_model
        self.accept_score = accept_score

    def verify(self, query: Query, response: LLMResponse) -> Verdict:
        prompt = f"Question:\n{query.text}\n\nCandidate answer:\n{response.text}\n\nScore it."
        review = self.judge.complete(prompt + _carry_gold(query), system=_SYSTEM)
        score = _parse_score(review.text)
        conf = max(0.0, min(1.0, score / 10.0))
        passed = score >= self.accept_score
        return self._tag(Verdict(passed, conf, f"judge score {score:.1f}/10", cost_usd=review.cost_usd))


def _carry_gold(query: Query) -> str:
    m = re.search(r"\[GOLD:.*?\]", query.text, re.DOTALL)
    return f"\n{m.group(0)}" if m else ""


def _parse_score(text: str) -> float:
    m = _SCORE_RE.search(text)
    if not m:
        return 5.0
    return max(0.0, min(10.0, float(m.group(1))))
