# Verifiers

A verifier answers one question: *was the cheap model's answer good enough, or
should we escalate?* It returns a `Verdict(passed, confidence, cost_usd)`. The
policy escalates when `confidence` falls below the tier's threshold.

The verifier is the single most important design choice in a cascade. A perfect
verifier gives you the strong model's accuracy at the cheap model's cost. A
useless verifier gives you the strong model's cost with extra latency. Real
verifiers live in between, and you pick one per tier by trading its cost against
its reliability.

| Verifier | Extra calls | Signal | Best used on |
|---|---|---|---|
| `RuleVerifier` | 0 | Structural: non-empty, no refusal, matches a regex/schema/predicate | Every tier as a pre-filter; the top tier (always-accept) |
| `SelfCheckVerifier` | 1 (same tier) | The model grades its own answer YES/NO + confidence | Mid tiers — cheap, exploits the generation-verification gap |
| `ConsistencyVerifier` | N−1 (same tier) | Agreement across N samples; wide disagreement ⇒ escalate | Cheap tiers where N extra calls are still near-free |
| `JudgeVerifier` | 1 (stronger tier) | A stronger model scores the answer 0–10 | The penultimate tier, where a false accept is costly |

## Cost intuition

With per-token prices roughly `mini : sonnet : opus = 1 : 20 : 100`:

- Consistency at N=5 on the **cheap** model ≈ 5 cheap calls ≈ **free** relative to one strong call.
- Consistency at N=5 on the **mid** model ≈ 5 mid calls ≈ **one strong call** — so it wipes out the savings. Use self-check (one call) on mid tiers instead.
- Judge on the **strong** model to grade a **mid** answer costs a full strong call — only worth it when a wrong accept is expensive.

This is why the shipped benchmark uses **consistency on the cheap tier** and
**self-check on the mid tier**: it's the cost/signal sweet spot.

## The generation–verification gap

Self-check works because a model is often better at *spotting* an error than at
*avoiding* it — verifying "is this answer correct?" is an easier task than
producing the answer. The gap is model- and task-dependent; calibration measures
it empirically for your setup and sets thresholds accordingly.

## Writing your own

Subclass `Verifier` and implement `verify(query, response) -> Verdict`. Populate
`cost_usd` if you make LLM calls, so the budget accounting stays honest.
