"""The cascade executor.

Given an ordered list of tiers (each: a model + its verifier), run a query
through them starting at ``entry_tier``:

    for tier in tiers[entry:]:
        answer  = tier.model.complete(query)
        verdict = tier.verifier.verify(query, answer)
        if policy.accept(tier, verdict): return answer
        # else escalate to the next tier

Budget breaches stop escalation early and return the best answer so far. Every
attempt is recorded as a ``Step`` for the trace.
"""

from __future__ import annotations

from dataclasses import dataclass

from cascade.budget import Budget, BudgetExceeded
from cascade.policy import Policy
from cascade.types import Query, RouteResult, Step
from cascade.verifiers.base import Verifier
from cascade.llm import LLM


@dataclass
class Tier:
    """One rung of the cascade: a model plus the verifier that gates it."""

    model: LLM
    verifier: Verifier

    @property
    def name(self) -> str:
        return self.model.model


class Cascade:
    """Executes a query through an ordered list of tiers under a policy."""

    def __init__(self, tiers: list[Tier], policy: Policy) -> None:
        if not tiers:
            raise ValueError("cascade needs at least one tier")
        self.tiers = tiers
        self.policy = policy

    def run(
        self,
        query: Query,
        entry_tier: int = 0,
        budget: Budget | None = None,
    ) -> RouteResult:
        budget = budget or Budget()
        entry = max(entry_tier, query.min_tier)
        entry = min(entry, len(self.tiers) - 1)

        steps: list[Step] = []
        best_step: Step | None = None

        for idx in range(entry, len(self.tiers)):
            tier = self.tiers[idx]
            is_last = idx == len(self.tiers) - 1
            try:
                resp = tier.model.complete(query.text, system=query.system)
                budget.charge(resp.cost_usd, resp.latency_ms)
                verdict = tier.verifier.verify(query, resp)
                # Charge the verifier's own LLM spend (0 for rule-based).
                budget.charge(verdict.cost_usd, 0.0, count=False)
            except BudgetExceeded:
                break

            accepted = self.policy.accept(idx, verdict, is_last)
            step = Step(tier=idx, response=resp, verdict=verdict, escalated=not accepted)
            steps.append(step)
            if best_step is None or verdict.confidence > best_step.verdict.confidence:
                best_step = step

            if accepted:
                return self._result(query, steps, winning=step)

        # Escalation exhausted or budget hit — return the highest-confidence step.
        winning = best_step or steps[-1]
        return self._result(query, steps, winning=winning)

    @staticmethod
    def _result(query: Query, steps: list[Step], winning: Step) -> RouteResult:
        return RouteResult(
            query=query,
            answer=winning.response.text,
            final_tier=winning.tier,
            steps=steps,
            total_cost_usd=sum(s.response.cost_usd + s.verdict.cost_usd for s in steps),
            total_latency_ms=sum(s.response.latency_ms for s in steps),
        )
