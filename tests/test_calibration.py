from cascade.cascade import Cascade, Tier
from cascade.calibration import (
    CalibrationItem,
    calibrate,
    evaluate_thresholds,
    pareto_frontier,
    OperatingPoint,
)
from cascade.llm import MockLLM
from cascade.verifiers import RuleVerifier, SelfCheckVerifier, JudgeVerifier

from tests.conftest import gold_query


def _factory():
    cheap = MockLLM("gpt-4o-mini", 0, 0.35)
    mid = MockLLM("claude-sonnet-4-5", 1, 0.65)
    strong = MockLLM("claude-opus-4-7", 2, 0.9)

    def make(policy):
        tiers = [
            Tier(cheap, RuleVerifier()),
            Tier(mid, SelfCheckVerifier(mid)),
            Tier(strong, JudgeVerifier(strong)),
        ]
        return Cascade(tiers, policy)

    return make


def _items(n=30):
    items = []
    for i in range(n):
        diff = (i % 10) / 10.0
        items.append(CalibrationItem(gold_query(f"Question {i}?", str(i), diff), gold=str(i)))
    return items


def test_evaluate_thresholds_returns_point():
    point = evaluate_thresholds(_factory(), _items(20), thresholds=(0.5, 0.5, 0.0))
    assert 0.0 <= point.accuracy <= 1.0
    assert point.mean_cost_usd >= 0.0
    assert point.mean_calls >= 1.0


def test_pareto_frontier_filters_dominated():
    pts = [
        OperatingPoint((0.0,), accuracy=0.8, mean_cost_usd=0.10, mean_calls=1),
        OperatingPoint((0.5,), accuracy=0.8, mean_cost_usd=0.20, mean_calls=2),  # dominated
        OperatingPoint((0.9,), accuracy=0.9, mean_cost_usd=0.30, mean_calls=3),
    ]
    front = pareto_frontier(pts)
    costs = {p.mean_cost_usd for p in front}
    assert 0.20 not in costs  # dominated point removed
    assert 0.10 in costs and 0.30 in costs


def test_calibrate_hits_or_reports_target():
    result = calibrate(_factory(), _items(30), n_tiers=3, target_accuracy=0.7)
    assert result.chosen is not None
    assert len(result.frontier) >= 1
    # Cheapest tier-0-only is on the frontier; chosen must be at least as good.
    if result.met_target:
        assert result.chosen.accuracy >= 0.7


def test_higher_target_costs_at_least_as_much():
    factory, items = _factory(), _items(30)
    low = calibrate(factory, items, n_tiers=3, target_accuracy=0.6)
    high = calibrate(factory, items, n_tiers=3, target_accuracy=0.9)
    if low.met_target and high.met_target:
        assert high.chosen.mean_cost_usd >= low.chosen.mean_cost_usd - 1e-9
