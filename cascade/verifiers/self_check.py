"""Self-check verifier — ask the same model whether its answer is right.

One extra call to the *same* tier. Cheap relative to escalating, and
surprisingly effective: a model is often better at spotting its own error than
at avoiding it (the "generation-verification gap"). Returns a confidence parsed
from a YES/NO + certainty reply.
"""

from __future__ import annotations

import re

from cascade.llm import LLM
from cascade.types import LLMResponse, Query, Verdict
from cascade.verifiers.base import Verifier

_SYSTEM = (
    "You are a strict grader. You are shown a question and a candidate answer. "
    "Decide if the answer is correct. Reply on one line as: "
    "'VERDICT: YES|NO, CONFIDENCE: <0-100>'."
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(YES|NO)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE:\s*(\d{1,3})", re.IGNORECASE)


class SelfCheckVerifier(Verifier):
    """Second opinion from the same model that produced the answer."""

    name = "self_check"

    def __init__(self, model: LLM) -> None:
        self.model = model

    def verify(self, query: Query, response: LLMResponse) -> Verdict:
        prompt = (
            f"Question:\n{query.text}\n\n"
            f"Candidate answer:\n{response.text}\n\n"
            "Is the candidate answer correct?"
        )
        # Preserve any gold sentinel so the mock model can grade deterministically.
        review = self.model.complete(prompt + _carry_gold(query), system=_SYSTEM)
        passed, conf = _parse(review.text)
        return self._tag(Verdict(passed, conf, review.text.strip()[:120], cost_usd=review.cost_usd))


def _carry_gold(query: Query) -> str:
    # The router embeds [GOLD:...] in the original prompt for offline runs.
    import re as _re

    m = _re.search(r"\[GOLD:.*?\]", query.text, _re.DOTALL)
    return f"\n{m.group(0)}" if m else ""


def _parse(text: str) -> tuple[bool, float]:
    vm = _VERDICT_RE.search(text)
    cm = _CONF_RE.search(text)
    passed = bool(vm and vm.group(1).upper() == "YES")
    conf = (int(cm.group(1)) / 100.0) if cm else (0.7 if passed else 0.3)
    conf = max(0.0, min(1.0, conf))
    # If the grader says NO, confidence is confidence-in-rejection; expose it as
    # low acceptance confidence so the policy escalates.
    return passed, conf if passed else min(conf, 0.4)
