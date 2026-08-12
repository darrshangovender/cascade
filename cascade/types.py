"""Core data types shared across the cascade system.

Everything that crosses a module boundary is one of these frozen dataclasses /
Pydantic models, so a route can be serialised, logged, and replayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Difficulty(str, Enum):
    """Estimated difficulty band for a query. Drives the entry tier."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True)
class Query:
    """An inbound question plus optional routing hints."""

    text: str
    system: str | None = None
    # Optional caller hint: force a minimum tier (0 = cheapest).
    min_tier: int = 0
    # Arbitrary metadata carried through the trace for later analysis.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """One model completion with its cost accounting."""

    text: str
    model: str
    tier: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float = 0.0


@dataclass
class Verdict:
    """A verifier's judgement of a candidate answer.

    ``passed`` is the accept/reject decision. ``confidence`` is a calibrated
    [0, 1] score used by the escalation policy — the cascade escalates when
    confidence falls below the tier's threshold.
    """

    passed: bool
    confidence: float
    reason: str = ""
    verifier: str = ""
    # Cost of running the verifier itself (verifiers that call an LLM — self-check,
    # consistency, judge — spend real tokens; the cascade charges this to the budget).
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class Step:
    """One tier attempt inside a cascade: the model call + its verdict."""

    tier: int
    response: LLMResponse
    verdict: Verdict
    escalated: bool


@dataclass
class RouteResult:
    """The final outcome of routing a single query through the cascade."""

    query: Query
    answer: str
    final_tier: int
    steps: list[Step]
    total_cost_usd: float
    total_latency_ms: float
    difficulty: Difficulty | None = None

    @property
    def n_escalations(self) -> int:
        return sum(1 for s in self.steps if s.escalated)

    @property
    def accepted_by(self) -> str:
        """Name of the verifier that accepted the winning answer (or 'exhausted')."""
        if self.steps and self.steps[-1].verdict.passed:
            return self.steps[-1].verdict.verifier or "unknown"
        return "exhausted"
