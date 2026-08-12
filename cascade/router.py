"""Top-level Router — the public entry point.

Ties difficulty estimation, the cascade, and budgets together:

    router = Router(tiers, policy, difficulty=HeuristicDifficulty())
    result = router.route("What is 2 + 2?")

The difficulty estimate picks the entry tier so easy questions skip nothing and
hard questions don't waste a doomed cheap call.
"""

from __future__ import annotations

from cascade.budget import Budget
from cascade.cascade import Cascade, Tier
from cascade.difficulty import HeuristicDifficulty, entry_tier
from cascade.policy import Policy
from cascade.types import Query, RouteResult


class Router:
    def __init__(
        self,
        tiers: list[Tier],
        policy: Policy,
        difficulty=None,
        default_budget: Budget | None = None,
    ) -> None:
        self.cascade = Cascade(tiers, policy)
        self.difficulty = difficulty or HeuristicDifficulty()
        self.default_budget = default_budget

    def route(
        self,
        text: str,
        system: str | None = None,
        min_tier: int = 0,
        budget: Budget | None = None,
    ) -> RouteResult:
        query = Query(text=text, system=system, min_tier=min_tier)
        band = self.difficulty.classify(query)
        entry = entry_tier(band, len(self.cascade.tiers))
        # Fresh budget per route unless a caller supplies one.
        b = budget or (Budget(**_budget_kwargs(self.default_budget)) if self.default_budget else Budget())
        result = self.cascade.run(query, entry_tier=entry, budget=b)
        result.difficulty = band
        return result


def _budget_kwargs(b: Budget) -> dict:
    return {
        "max_cost_usd": b.max_cost_usd,
        "max_latency_ms": b.max_latency_ms,
        "max_calls": b.max_calls,
    }
