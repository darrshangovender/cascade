from cascade.difficulty import HeuristicDifficulty, entry_tier
from cascade.types import Difficulty, Query


def test_short_factual_is_easy():
    d = HeuristicDifficulty()
    assert d.classify(Query("What is the capital of France?")) is Difficulty.EASY


def test_multistep_prompt_scores_higher():
    d = HeuristicDifficulty()
    easy = d.score(Query("What is 2+2?"))
    hard = d.score(
        Query(
            "Prove step by step why the derivative of x^2 is 2x, then explain why "
            "this generalises, comparing the power rule to first principles."
        )
    )
    assert hard > easy


def test_entry_tier_mapping():
    assert entry_tier(Difficulty.EASY, 3) == 0
    assert entry_tier(Difficulty.HARD, 3) == 1  # n_tiers-2
    assert entry_tier(Difficulty.MEDIUM, 3) == 1
    # single-tier cascade always enters at 0
    assert entry_tier(Difficulty.HARD, 1) == 0


def test_scores_are_bounded():
    d = HeuristicDifficulty()
    for text in ["hi", "x" * 5000, "calculate, compare, derive; prove step by step"]:
        s = d.score(Query(text))
        assert 0.0 <= s <= 1.0
