from cascade.budget import Budget
from cascade.cascade import Cascade, Tier
from cascade.llm import MockLLM
from cascade.policy import AlwaysAcceptPolicy, ThresholdPolicy
from cascade.types import Query
from cascade.verifiers import RuleVerifier, SelfCheckVerifier

from tests.conftest import gold_query


def test_always_accept_stops_at_entry_tier(three_tiers):
    casc = Cascade(three_tiers, AlwaysAcceptPolicy())
    result = casc.run(gold_query("What is 2+2?", "4", 0.1))
    assert len(result.steps) == 1
    assert result.final_tier == 0


def test_high_threshold_forces_escalation(three_tiers):
    # Threshold 0.99 at tiers 0 and 1 → almost always escalate to the top tier.
    casc = Cascade(three_tiers, ThresholdPolicy([0.99, 0.99, 0.0]))
    result = casc.run(gold_query("Prove a hard theorem", "QED", 0.9))
    assert result.final_tier == 2
    assert len(result.steps) == 3


def test_min_tier_hint_respected(three_tiers):
    casc = Cascade(three_tiers, AlwaysAcceptPolicy())
    q = Query("easy [GOLD:4||0.1]", min_tier=1)
    result = casc.run(q, entry_tier=0)
    assert result.steps[0].tier == 1  # min_tier overrides entry


def test_budget_stops_escalation():
    # Two tiers, tiny budget that only affords one call.
    tiers = [
        Tier(MockLLM("gpt-4o-mini", 0, 0.3), RuleVerifier()),
        Tier(MockLLM("claude-opus-4-7", 1, 0.9), RuleVerifier()),
    ]
    casc = Cascade(tiers, ThresholdPolicy([0.99, 0.0]))
    budget = Budget(max_calls=1)
    result = casc.run(gold_query("q", "a", 0.5), budget=budget)
    # Only one model call fit inside the budget.
    assert len(result.steps) == 1


def test_result_accounting(three_tiers):
    casc = Cascade(three_tiers, ThresholdPolicy([0.99, 0.99, 0.0]))
    result = casc.run(gold_query("hard", "x", 0.9))
    # Total cost includes both the model responses AND the verifier LLM calls.
    expected = sum(s.response.cost_usd + s.verdict.cost_usd for s in result.steps)
    assert result.total_cost_usd == expected
    assert result.total_cost_usd > 0
    assert result.n_escalations >= 1


def test_verifier_cost_is_charged(three_tiers):
    # The self-check + judge verifiers spend tokens; total must exceed the bare
    # model-response cost.
    casc = Cascade(three_tiers, ThresholdPolicy([0.99, 0.99, 0.0]))
    result = casc.run(gold_query("hard", "x", 0.9))
    response_only = sum(s.response.cost_usd for s in result.steps)
    assert result.total_cost_usd > response_only


def test_empty_cascade_raises():
    import pytest

    with pytest.raises(ValueError):
        Cascade([], AlwaysAcceptPolicy())
