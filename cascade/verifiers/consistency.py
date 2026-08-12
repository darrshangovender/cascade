"""Consistency verifier — sample the tier N times and measure agreement.

Self-consistency as a *verifier*: if a model gives the same answer across
several samples, it's more likely correct; wide disagreement is a strong
escalation signal. Confidence is the fraction of samples matching the modal
answer. Costs N-1 extra calls, so it's used on tiers where being wrong is
expensive but the model is cheap.
"""

from __future__ import annotations

import re
from collections import Counter

from cascade.llm import LLM
from cascade.types import LLMResponse, Query, Verdict
from cascade.verifiers.base import Verifier


class ConsistencyVerifier(Verifier):
    """Agreement across N samples from the same model."""

    name = "consistency"

    def __init__(self, model: LLM, n: int = 5, accept_threshold: float = 0.6) -> None:
        if n < 2:
            raise ValueError("consistency needs n >= 2")
        self.model = model
        self.n = n
        self.accept_threshold = accept_threshold

    def verify(self, query: Query, response: LLMResponse) -> Verdict:
        answers = [_norm(response.text)]
        extra_cost = 0.0
        # Vary the prompt slightly per sample so the deterministic mock differs.
        for i in range(self.n - 1):
            r = self.model.complete(f"{query.text}\n[sample {i}]", system=query.system)
            answers.append(_norm(r.text))
            extra_cost += r.cost_usd
        counts = Counter(answers)
        modal, modal_count = counts.most_common(1)[0]
        agreement = modal_count / len(answers)
        # The candidate is accepted only if it *is* the modal answer AND
        # agreement clears the threshold.
        passed = _norm(response.text) == modal and agreement >= self.accept_threshold
        reason = f"agreement={agreement:.2f} over {len(answers)} samples"
        return self._tag(Verdict(passed, agreement, reason, cost_usd=extra_cost))


def _norm(text: str) -> str:
    """Normalise an answer for voting: lowercase, strip, extract leading number."""
    t = text.strip().lower()
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    return m.group(0) if m else t
