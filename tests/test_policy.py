from cascade.policy import (
    AlwaysAcceptPolicy,
    TargetAccuracyPolicy,
    ThresholdPolicy,
)
from cascade.types import Verdict


def v(passed, conf):
    return Verdict(passed, conf)


def test_threshold_accepts_above():
    p = ThresholdPolicy([0.5, 0.0])
    assert p.accept(0, v(True, 0.6), is_last_tier=False)
    assert not p.accept(0, v(True, 0.4), is_last_tier=False)


def test_threshold_rejects_failed_verdict():
    p = ThresholdPolicy([0.1, 0.0])
    assert not p.accept(0, v(False, 0.9), is_last_tier=False)


def test_last_tier_always_accepts():
    p = ThresholdPolicy([0.99])
    assert p.accept(0, v(False, 0.0), is_last_tier=True)


def test_always_accept():
    p = AlwaysAcceptPolicy()
    assert p.accept(0, v(False, 0.0), is_last_tier=False)


def test_threshold_needs_values():
    import pytest

    with pytest.raises(ValueError):
        ThresholdPolicy([])


def test_target_accuracy_remembers_target():
    p = TargetAccuracyPolicy([0.5, 0.0], target_accuracy=0.9)
    assert p.target_accuracy == 0.9
    assert p.accept(0, v(True, 0.7), is_last_tier=False)
