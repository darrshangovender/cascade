from cascade.calibration import (
    CalibrationItem,
    OperatingPoint,
    calibrate,
    evaluate_thresholds,
    pareto_frontier,
)
from cascade.cascade import Cascade, Tier
from cascade.llm import MockLLM
from cascade.verifiers import JudgeVerifier, RuleVerifier, SelfCheckVerifier
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


def test_empty_answer_is_not_graded_correct():
    """'' is a substring of every gold, so the loose grader used to count a
    model that returned nothing as a hit — inflating calibration accuracy."""
    from cascade.calibration import _grade

    assert not _grade("", "42")
    assert not _grade("   ", "Paris")
    assert _grade("The answer is 42", "42")
    assert _grade("42", "42 degrees")


def test_mock_wrong_answers_do_not_grade_correct():
    """The mock's deliberate wrong answer used to be the gold with a suffix,
    which the substring grader scored as correct — so a non-numeric calibration
    set measured ~100% accuracy no matter how bad the model was."""
    from cascade.calibration import _grade
    from cascade.llm import _plausible_wrong

    for gold in ("Paris", "QED", "x", "the mitochondria"):
        for seed in (1, 2, 3, 7, 99):
            assert not _grade(_plausible_wrong(gold, seed), gold), (gold, seed)
