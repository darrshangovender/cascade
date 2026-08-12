from cascade.llm import MockLLM
from cascade.types import LLMResponse, Query, Verdict
from cascade.verifiers import (
    ConsistencyVerifier,
    JudgeVerifier,
    RuleVerifier,
    SelfCheckVerifier,
)


def _resp(text: str, tier: int = 0) -> LLMResponse:
    return LLMResponse(text, "m", tier, 10, 5, 0.0)


def test_rule_verifier_rejects_empty():
    v = RuleVerifier(min_len=1)
    verdict = v.verify(Query("q"), _resp("   "))
    assert not verdict.passed
    assert verdict.verifier == "rules"


def test_rule_verifier_rejects_refusal():
    v = RuleVerifier()
    verdict = v.verify(Query("q"), _resp("I cannot help with that."))
    assert not verdict.passed


def test_rule_verifier_pattern():
    v = RuleVerifier(pattern=r"^\d+$")
    assert v.verify(Query("q"), _resp("42")).passed
    assert not v.verify(Query("q"), _resp("forty-two")).passed


def test_rule_verifier_predicate():
    v = RuleVerifier(predicate=lambda q, a: a.startswith("YES"))
    assert v.verify(Query("q"), _resp("YES it is")).passed
    assert not v.verify(Query("q"), _resp("NO")).passed


def test_verdict_confidence_bounds():
    import pytest

    with pytest.raises(ValueError):
        Verdict(True, 1.5)


def test_self_check_returns_verdict():
    model = MockLLM("gpt-4o-mini", tier=0, skill=0.5)
    v = SelfCheckVerifier(model)
    verdict = v.verify(Query("What is 2+2? [GOLD:4||0.1]"), _resp("4"))
    assert isinstance(verdict, Verdict)
    assert verdict.verifier == "self_check"


def test_consistency_needs_min_two():
    import pytest

    model = MockLLM("gpt-4o-mini", tier=0, skill=0.5)
    with pytest.raises(ValueError):
        ConsistencyVerifier(model, n=1)


def test_consistency_agreement_is_confidence():
    model = MockLLM("gpt-4o-mini", tier=0, skill=0.99)  # near-perfect → consistent
    v = ConsistencyVerifier(model, n=5)
    q = Query("What is 2+2? [GOLD:4||0.05]")
    verdict = v.verify(q, _resp("4"))
    assert 0.0 <= verdict.confidence <= 1.0


def test_judge_scores():
    judge = MockLLM("claude-opus-4-7", tier=2, skill=0.95)
    v = JudgeVerifier(judge)
    verdict = v.verify(Query("What is 2+2? [GOLD:4||0.05]"), _resp("4"))
    assert verdict.verifier == "judge"
