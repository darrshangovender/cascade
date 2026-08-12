"""Quickstart: a 3-tier cascade routed by difficulty.

Runs fully offline with MockLLM — no API keys required.

    python examples/quickstart.py
"""

from cascade import Router, Tier, ThresholdPolicy
from cascade.llm import MockLLM
from cascade.verifiers import ConsistencyVerifier, RuleVerifier, SelfCheckVerifier


def build_router() -> Router:
    cheap = MockLLM("gpt-4o-mini", tier=0, skill=0.40)
    mid = MockLLM("claude-sonnet-4-5", tier=1, skill=0.65)
    strong = MockLLM("claude-opus-4-7", tier=2, skill=0.92)

    tiers = [
        # Cheap tier: consistency (cheap samples are near-free) → stop on easy.
        Tier(cheap, ConsistencyVerifier(cheap, n=5, accept_threshold=0.6)),
        # Mid tier: single self-check call — cheap verification, real signal.
        Tier(mid, SelfCheckVerifier(mid)),
        Tier(strong, RuleVerifier()),  # top tier — always accepts
    ]
    # Escalate from tier 0 if agreement < 0.7, from tier 1 if self-check < 0.6.
    policy = ThresholdPolicy([0.7, 0.6, 0.0])
    return Router(tiers, policy)


def main() -> None:
    router = build_router()

    questions = [
        ("What is the capital of France?", "Paris", 0.1),
        ("What is 17 * 23?", "391", 0.45),
        ("Prove step by step why sqrt(2) is irrational.", "irrational", 0.9),
    ]

    for text, gold, diff in questions:
        result = router.route(f"{text} [GOLD:{gold}||{diff}]")
        print(f"\nQ: {text}")
        print(f"  difficulty : {result.difficulty.value}")
        print(f"  answer     : {result.answer}")
        print(f"  final tier : {result.final_tier}  ({result.steps[-1].response.model})")
        print(f"  tiers used : {[s.response.model for s in result.steps]}")
        print(f"  cost       : ${result.total_cost_usd:.5f}")
        print(f"  accepted by: {result.accepted_by}")


if __name__ == "__main__":
    main()
