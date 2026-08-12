"""Verifier interface.

A verifier inspects a candidate answer and returns a ``Verdict`` — an
accept/reject plus a calibrated confidence. The cascade escalates to the next
tier when ``verdict.confidence`` is below the tier's threshold (or the verdict
fails outright).

Verifiers are the heart of the system: a cascade is only as good as its ability
to *know when the cheap model was wrong*. Different verifiers trade cost against
reliability, and you mix them per tier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cascade.types import LLMResponse, Query, Verdict


class Verifier(ABC):
    """Judge a candidate answer to a query."""

    name: str = "verifier"

    @abstractmethod
    def verify(self, query: Query, response: LLMResponse) -> Verdict:
        """Return a Verdict for ``response`` as an answer to ``query``."""

    def _tag(self, verdict: Verdict) -> Verdict:
        verdict.verifier = self.name
        return verdict
