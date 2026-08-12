from cascade.router import Router
from cascade.policy import ThresholdPolicy, AlwaysAcceptPolicy
from cascade.types import Difficulty

from tests.conftest import three_tiers  # noqa: F401 (fixture)


def test_router_returns_result(three_tiers):
    router = Router(three_tiers, AlwaysAcceptPolicy())
    result = router.route("What is the capital of France? [GOLD:Paris||0.1]")
    assert result.answer
    assert result.difficulty in set(Difficulty)


def test_router_easy_question_enters_low(three_tiers):
    router = Router(three_tiers, AlwaysAcceptPolicy())
    result = router.route("Hi [GOLD:hello||0.05]")
    assert result.difficulty is Difficulty.EASY
    assert result.steps[0].tier == 0


def test_router_hard_question_enters_higher(three_tiers):
    router = Router(three_tiers, AlwaysAcceptPolicy())
    hard = (
        "Prove step by step and explain why, comparing the trade-offs, deriving "
        "the result: [GOLD:proof||0.95]"
    )
    result = router.route(hard)
    assert result.difficulty is Difficulty.HARD
    assert result.steps[0].tier >= 1  # skipped the cheapest tier


def test_router_fresh_budget_each_route(three_tiers):
    router = Router(three_tiers, ThresholdPolicy([0.99, 0.99, 0.0]))
    r1 = router.route("q1 [GOLD:a||0.9]")
    r2 = router.route("q2 [GOLD:b||0.9]")
    # Both routes get full budget; both can escalate to the top tier.
    assert r1.final_tier == 2 and r2.final_tier == 2
