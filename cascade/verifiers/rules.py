"""Rule-based verifier — zero-cost, deterministic structural checks.

Cheapest possible gate. Catches the failure modes that don't need a model to
detect: empty answers, refusals, format violations, schema mismatches. Runs
first in most tiers because rejecting here saves a verifier LLM call.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from cascade.types import LLMResponse, Query, Verdict
from cascade.verifiers.base import Verifier

_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "as an ai",
    "i'm not able",
    "i am unable",
    "i don't know",
)


class RuleVerifier(Verifier):
    """Structural / format checks with no model call.

    Parameters
    ----------
    pattern
        Optional regex the answer must match (e.g. r"^\\d+$" for a bare integer).
    predicate
        Optional callable (query, answer_text) -> bool for custom checks.
    min_len
        Reject answers shorter than this many characters.
    """

    name = "rules"

    def __init__(
        self,
        pattern: str | None = None,
        predicate: Callable[[Query, str], bool] | None = None,
        min_len: int = 1,
    ) -> None:
        self.pattern = re.compile(pattern) if pattern else None
        self.predicate = predicate
        self.min_len = min_len

    def verify(self, query: Query, response: LLMResponse) -> Verdict:
        text = response.text.strip()
        low = text.lower()
        if len(text) < self.min_len:
            return self._tag(Verdict(False, 0.0, "answer too short"))
        if any(m in low for m in _REFUSAL_MARKERS):
            return self._tag(Verdict(False, 0.05, "refusal detected"))
        if self.pattern and not self.pattern.search(text):
            return self._tag(Verdict(False, 0.1, "pattern mismatch"))
        if self.predicate and not self.predicate(query, text):
            return self._tag(Verdict(False, 0.1, "predicate failed"))
        # Structural checks passed. Rules can't confirm *correctness*, only
        # well-formedness, so confidence is capped modestly.
        return self._tag(Verdict(True, 0.6, "structural checks passed"))
