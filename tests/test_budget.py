import pytest

from cascade.budget import Budget, BudgetExceeded


def test_cost_cap_raises():
    b = Budget(max_cost_usd=0.01)
    with pytest.raises(BudgetExceeded, match="cost"):
        b.charge(cost_usd=0.02, latency_ms=1.0)


def test_latency_cap_raises():
    b = Budget(max_latency_ms=100)
    with pytest.raises(BudgetExceeded, match="latency"):
        b.charge(cost_usd=0.0, latency_ms=200)


def test_call_cap_raises():
    b = Budget(max_calls=2)
    b.charge(0.0, 1.0)
    b.charge(0.0, 1.0)
    with pytest.raises(BudgetExceeded, match="calls"):
        b.charge(0.0, 1.0)


def test_accumulates_spend():
    b = Budget(max_cost_usd=1.0)
    b.charge(0.1, 1.0)
    b.charge(0.2, 1.0)
    assert abs(b.spent_usd - 0.3) < 1e-9
    assert b.calls == 2
