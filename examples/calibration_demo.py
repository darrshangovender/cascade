"""Calibration demo: fit escalation thresholds to a target accuracy.

Shows the accuracy/cost Pareto frontier and the cheapest operating point that
hits the target.

    python examples/calibration_demo.py
"""

from cascade.cascade import Cascade, Tier
from cascade.calibration import CalibrationItem, calibrate
from cascade.llm import MockLLM
from cascade.verifiers import ConsistencyVerifier, RuleVerifier, SelfCheckVerifier


def factory():
    cheap = MockLLM("gpt-4o-mini", 0, 0.40)
    mid = MockLLM("claude-sonnet-4-5", 1, 0.65)
    strong = MockLLM("claude-opus-4-7", 2, 0.92)

    def make(policy):
        return Cascade(
            [
                Tier(cheap, ConsistencyVerifier(cheap, n=5, accept_threshold=0.6)),
                Tier(mid, SelfCheckVerifier(mid)),
                Tier(strong, RuleVerifier()),
            ],
            policy,
        )

    return make


def calibration_set(n: int = 80):
    items = []
    import random

    rng = random.Random(7)
    for i in range(n):
        diff = round(rng.uniform(0.05, 0.95), 2)
        gold = str(i * 7 % 100)
        q = f"Question {i}: compute the value. [GOLD:{gold}||{diff}]"
        items.append(CalibrationItem(query=_as_query(q), gold=gold))
    return items


def _as_query(text: str):
    from cascade.types import Query

    return Query(text)


def main() -> None:
    items = calibration_set()
    for target in (0.6, 0.75, 0.9):
        result = calibrate(factory(), items, n_tiers=3, target_accuracy=target)
        c = result.chosen
        status = "MET" if result.met_target else "BEST-EFFORT"
        print(
            f"target={target:.2f} [{status}]  "
            f"thresholds={tuple(round(t, 2) for t in c.thresholds)}  "
            f"acc={c.accuracy:.3f}  cost=${c.mean_cost_usd:.5f}/q  calls={c.mean_calls:.2f}"
        )

    print("\nPareto frontier (cheapest -> most accurate):")
    front = calibrate(factory(), items, n_tiers=3, target_accuracy=0.99).frontier
    for p in front:
        print(f"  acc={p.accuracy:.3f}  cost=${p.mean_cost_usd:.5f}/q  thresholds={tuple(round(t,2) for t in p.thresholds)}")


if __name__ == "__main__":
    main()
