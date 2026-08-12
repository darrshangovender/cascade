"""Hard budget guards for a route.

A cascade can, in the worst case, call every tier plus a verifier per tier.
The Budget caps that blast radius on three axes: cumulative cost, cumulative
wall-clock, and number of model calls. Any breach raises ``BudgetExceeded`` so
the router can stop escalating and return the best answer so far.
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(Exception):
    """Raised when a route would exceed one of its caps."""


@dataclass
class Budget:
    max_cost_usd: float = 1.0
    max_latency_ms: float = 30_000.0
    max_calls: int = 12

    _spent_usd: float = 0.0
    _elapsed_ms: float = 0.0
    _calls: int = 0

    def charge(self, cost_usd: float, latency_ms: float, count: bool = True) -> None:
        """Record spend against the budget. Raises if any cap is now breached.

        ``count=False`` adds the cost/latency but does not increment the model-call
        counter — used for verifier overhead, which is spend but not a *generation*
        call the ``max_calls`` cap is meant to bound.
        """
        self._spent_usd += cost_usd
        self._elapsed_ms += latency_ms
        if count:
            self._calls += 1
        self.check()

    def check(self) -> None:
        if self._spent_usd > self.max_cost_usd:
            raise BudgetExceeded(f"cost {self._spent_usd:.4f} > {self.max_cost_usd:.4f} USD")
        if self._elapsed_ms > self.max_latency_ms:
            raise BudgetExceeded(f"latency {self._elapsed_ms:.0f} > {self.max_latency_ms:.0f} ms")
        if self._calls > self.max_calls:
            raise BudgetExceeded(f"calls {self._calls} > {self.max_calls}")

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def calls(self) -> int:
        return self._calls
