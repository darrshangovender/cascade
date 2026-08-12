"""Shared fixtures: a 3-tier mock cascade used across the test suite."""

import pytest

from cascade.cascade import Tier
from cascade.llm import MockLLM
from cascade.verifiers import RuleVerifier, SelfCheckVerifier, JudgeVerifier


@pytest.fixture
def cheap_model():
    return MockLLM("gpt-4o-mini", tier=0, skill=0.35)


@pytest.fixture
def mid_model():
    return MockLLM("claude-sonnet-4-5", tier=1, skill=0.65)


@pytest.fixture
def strong_model():
    return MockLLM("claude-opus-4-7", tier=2, skill=0.9)


@pytest.fixture
def three_tiers(cheap_model, mid_model, strong_model):
    """cheap→rules, mid→self-check, strong→judge(by strong)."""
    return [
        Tier(cheap_model, RuleVerifier(min_len=1)),
        Tier(mid_model, SelfCheckVerifier(mid_model)),
        Tier(strong_model, JudgeVerifier(strong_model)),
    ]


def gold_query(text: str, answer: str, difficulty: float = 0.5):
    """Build a query string carrying the gold sentinel the MockLLM reads."""
    from cascade.types import Query

    return Query(text=f"{text} [GOLD:{answer}||{difficulty}]")
