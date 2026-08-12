# Calibration

The escalation thresholds are the cascade's tuning knobs. Set them too low and
you accept wrong cheap answers (accuracy drops). Set them too high and you
escalate everything to the strong model (cost balloons). `calibration.py` finds
the thresholds that hit a **target accuracy at minimum cost**, offline, against a
labelled set.

## The method

It's a constrained optimisation over the accuracy/cost Pareto frontier:

1. **Sweep** a grid of candidate thresholds per tier (the last tier has none —
   there's nothing above it to escalate to).
2. **Simulate** the full cascade over the calibration set for each threshold
   vector, recording `(accuracy, mean_cost, mean_calls)`.
3. **Filter to the Pareto frontier** — drop any point that another beats on both
   accuracy and cost.
4. **Choose**: among frontier points meeting the accuracy target, take the
   cheapest. If none meet it, take the most accurate (best effort) and flag it.

Because the offline `MockLLM` is deterministic, the simulation is exact and the
chosen thresholds are fully reproducible.

## What the frontier looks like

From `examples/calibration_demo.py` (offline, 80-item set):

```
target=0.60 [MET]  thresholds=(0.00, 0.00)  acc=0.662  cost=$0.00016/q  calls=1.60
target=0.75 [MET]  thresholds=(0.70, 0.00)  acc=0.800  cost=$0.00027/q  calls=2.09
target=0.90 [MET]  thresholds=(0.85, 0.70)  acc=0.900  cost=$0.00036/q  calls=2.44
```

Accuracy and cost rise together, monotonically — exactly the trade-off you'd
expect. You pick the operating point your product needs: a customer-facing
feature might demand 0.90; a batch enrichment job might happily take 0.75 at 25%
less cost.

## Recalibrating in production

Thresholds drift as models change (a provider ships a better cheap model, your
prompt evolves). Re-run `calibrate()` on a fresh labelled sample whenever:

- you swap any model in the cascade,
- your traffic's difficulty distribution shifts,
- your accuracy or cost target changes.

Keep a small (100–300 item) labelled calibration set in version control and
recalibrate in CI, the same way you'd re-run an eval suite.
