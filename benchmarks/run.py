"""Reproducible benchmark: the cascade vs single-model baselines.

Shows the headline result — the cascade matches the strongest model's accuracy
at a fraction of the cost — by comparing:

  * cheap-only   : every query answered by the cheapest model (one call)
  * mid-only     : every query answered by the mid model (one call)
  * strong-only  : every query answered by the strongest model (the accuracy
                   ceiling, and the most expensive — one call)
  * cascade      : calibrated thresholds; cheap/mid models resolve the easy
                   queries and only the hard ones escalate to the strong model

Run:
    python benchmarks/run.py

Fully offline (MockLLM). Writes results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from cascade.cascade import Cascade, Tier
from cascade.calibration import CalibrationItem, calibrate, evaluate_thresholds, _grade
from cascade.llm import MockLLM
from cascade.verifiers import ConsistencyVerifier, RuleVerifier, SelfCheckVerifier

from benchmarks.dataset import make_dataset

# Skills span a wide range so each tier adds real accuracy on harder items.
CHEAP = lambda: MockLLM("gpt-4o-mini", 0, skill=0.40)
MID = lambda: MockLLM("claude-sonnet-4-5", 1, skill=0.65)
STRONG = lambda: MockLLM("claude-opus-4-7", 2, skill=0.92)


def factory():
    cheap, mid, strong = CHEAP(), MID(), STRONG()

    def make(policy):
        return Cascade(
            [
                # Cheap tier: consistency (5 cheap samples — nearly free) gives
                # a strong agreement signal for stopping on easy queries.
                Tier(cheap, ConsistencyVerifier(cheap, n=5, accept_threshold=0.6)),
                # Mid tier: self-check (ONE extra mid call). Consistency here
                # would cost ~5 mid calls ≈ one strong call, wiping out the
                # savings — a single self-check is the right cost/signal trade.
                Tier(mid, SelfCheckVerifier(mid)),
                Tier(strong, RuleVerifier()),  # last tier — always accepts
            ],
            policy,
        )

    return make


def single_model_baseline(items: list[CalibrationItem], model_factory) -> dict:
    """True single-model baseline: one call per query."""
    model = model_factory()
    n_correct, total_cost = 0, 0.0
    for item in items:
        resp = model.complete(item.query.text)
        if _grade(resp.text, item.gold):
            n_correct += 1
        total_cost += resp.cost_usd
    n = len(items)
    return {"accuracy": round(n_correct / n, 4), "cost_per_q": round(total_cost / n, 6)}


def main() -> int:
    train = make_dataset(n=150, seed=1)
    test = make_dataset(n=200, seed=2)

    rows: dict[str, dict] = {
        "cheap-only": single_model_baseline(test, CHEAP),
        "mid-only": single_model_baseline(test, MID),
        "strong-only": single_model_baseline(test, STRONG),
    }

    # Calibrate the cascade on train to (nearly) match strong-only accuracy,
    # then evaluate on the held-out test set.
    target = rows["strong-only"]["accuracy"] * 0.98
    cal = calibrate(factory(), train, n_tiers=3, target_accuracy=target)
    test_point = evaluate_thresholds(factory(), test, cal.chosen.thresholds)
    rows["cascade"] = {
        "accuracy": round(test_point.accuracy, 4),
        "cost_per_q": round(test_point.mean_cost_usd, 6),
        "thresholds": [round(t, 2) for t in cal.chosen.thresholds],
        "mean_calls": round(test_point.mean_calls, 2),
    }

    strong_cost = rows["strong-only"]["cost_per_q"]
    casc_cost = rows["cascade"]["cost_per_q"]
    savings = 1 - casc_cost / strong_cost if strong_cost else 0.0
    acc_gap = rows["strong-only"]["accuracy"] - rows["cascade"]["accuracy"]

    print("\n=== cascade benchmark (200-item test set, offline MockLLM) ===\n")
    print(f"{'strategy':14} {'accuracy':>10} {'cost/q':>13}")
    for name, r in rows.items():
        print(f"{name:14} {r['accuracy']:>10.1%} {r['cost_per_q']:>13.6f}")
    print(
        f"\ncascade reaches within {acc_gap:.1%} of strong-only accuracy "
        f"at {savings:.0%} lower cost."
    )
    print(
        f"cascade thresholds: {rows['cascade']['thresholds']}  "
        f"(mean {rows['cascade']['mean_calls']} tiers/query)"
    )

    out = Path(__file__).parent / "results.json"
    out.write_text(
        json.dumps(
            {"rows": rows, "cost_savings_vs_strong": round(savings, 4), "acc_gap": round(acc_gap, 4)},
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
