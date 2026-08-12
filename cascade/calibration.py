"""Offline threshold calibration — the analytical heart of the system.

Given a labelled calibration set, we want per-tier escalation thresholds that hit
a *target accuracy* at *minimum cost*. This is a constrained optimisation over
the accuracy/cost Pareto frontier.

Method
------
1. Sweep a grid of candidate thresholds per tier.
2. For each threshold vector, simulate the cascade over the calibration set and
   record (accuracy, mean_cost).
3. Keep only Pareto-optimal points (no other point is both cheaper AND more
   accurate).
4. Among points meeting the accuracy target, pick the cheapest. If none meet it,
   pick the most accurate (best effort).

Because the MockLLM is deterministic, the simulation is exact and the chosen
thresholds are reproducible.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from cascade.cascade import Cascade
from cascade.policy import ThresholdPolicy
from cascade.types import Query


@dataclass
class CalibrationItem:
    query: Query
    gold: str


@dataclass
class OperatingPoint:
    thresholds: tuple[float, ...]
    accuracy: float
    mean_cost_usd: float
    mean_calls: float


@dataclass
class CalibrationResult:
    chosen: OperatingPoint
    frontier: list[OperatingPoint]
    target_accuracy: float
    met_target: bool


def _grade(answer: str, gold: str) -> bool:
    """Loose correctness check used for calibration + benchmarking."""
    a = answer.strip().lower()
    g = gold.strip().lower()
    return g in a or a in g or a == g


def evaluate_thresholds(
    cascade_factory,
    items: list[CalibrationItem],
    thresholds: tuple[float, ...],
) -> OperatingPoint:
    """Run the cascade with a given threshold vector over the calibration set."""
    cascade: Cascade = cascade_factory(ThresholdPolicy(list(thresholds)))
    n_correct = 0
    total_cost = 0.0
    total_calls = 0
    for item in items:
        result = cascade.run(item.query)
        if _grade(result.answer, item.gold):
            n_correct += 1
        total_cost += result.total_cost_usd
        total_calls += len(result.steps)
    n = len(items)
    return OperatingPoint(
        thresholds=thresholds,
        accuracy=n_correct / n,
        mean_cost_usd=total_cost / n,
        mean_calls=total_calls / n,
    )


def pareto_frontier(points: list[OperatingPoint]) -> list[OperatingPoint]:
    """Keep points not dominated on (accuracy↑, cost↓)."""
    frontier: list[OperatingPoint] = []
    for p in points:
        dominated = any(
            (q.accuracy >= p.accuracy and q.mean_cost_usd <= p.mean_cost_usd)
            and (q.accuracy > p.accuracy or q.mean_cost_usd < p.mean_cost_usd)
            for q in points
        )
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda x: x.mean_cost_usd)


def calibrate(
    cascade_factory,
    items: list[CalibrationItem],
    n_tiers: int,
    target_accuracy: float,
    grid: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 0.85, 0.95),
) -> CalibrationResult:
    """Find cheapest thresholds meeting ``target_accuracy`` on ``items``.

    The last tier has no threshold (nothing to escalate to), so we sweep
    thresholds for the first ``n_tiers - 1`` tiers only.
    """
    n_sweep = max(1, n_tiers - 1)
    points: list[OperatingPoint] = []
    for combo in itertools.product(grid, repeat=n_sweep):
        thresholds = combo + (0.0,)  # last tier accepts unconditionally
        points.append(evaluate_thresholds(cascade_factory, items, thresholds))

    frontier = pareto_frontier(points)
    meeting = [p for p in frontier if p.accuracy >= target_accuracy]
    if meeting:
        chosen = min(meeting, key=lambda p: p.mean_cost_usd)
        met = True
    else:
        chosen = max(frontier, key=lambda p: p.accuracy)
        met = False
    return CalibrationResult(
        chosen=chosen,
        frontier=frontier,
        target_accuracy=target_accuracy,
        met_target=met,
    )
