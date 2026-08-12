# Architecture

`cascade` has four moving parts. Each is a small, independently-testable unit.

```
                       ┌───────────────────────────┐
   query ─────────────▶│   DifficultyEstimator     │  picks the entry tier
                       └─────────────┬─────────────┘
                                     ▼
     ┌──────────────────────────────────────────────────────────────┐
     │                          Cascade                              │
     │                                                               │
     │   tier 0 (cheap)     tier 1 (mid)        tier 2 (strong)      │
     │   ┌──────────┐       ┌──────────┐        ┌──────────┐         │
     │   │  model   │       │  model   │        │  model   │         │
     │   └────┬─────┘       └────┬─────┘        └────┬─────┘         │
     │        ▼                  ▼                   ▼               │
     │   ┌──────────┐       ┌──────────┐        ┌──────────┐        │
     │   │ verifier │       │ verifier │        │ verifier │        │
     │   └────┬─────┘       └────┬─────┘        └────┬─────┘        │
     │        │ Verdict          │ Verdict           │             │
     │        ▼                  ▼                    ▼             │
     │   ┌───────────────────────────────────────────────────┐    │
     │   │  Policy: accept  ──▶ return                        │    │
     │   │          escalate ──▶ next tier                    │    │
     │   └───────────────────────────────────────────────────┘    │
     │                    (Budget guards every call)               │
     └──────────────────────────────────────────────────────────────┘
                                     ▼
                             RouteResult + trace
```

## 1. DifficultyEstimator (`difficulty.py`)

Maps a query to `EASY | MEDIUM | HARD`, which sets the cascade's **entry tier**.
Easy questions start at the cheapest model; hard ones skip a doomed cheap call
and enter mid. Two implementations: a zero-cost heuristic and a one-call LLM
classifier.

## 2. Cascade (`cascade.py`)

Runs the query through tiers starting at the entry tier. Each tier is a
`(model, verifier)` pair. The model answers, the verifier judges, the policy
decides accept-or-escalate. Records every attempt as a `Step`.

## 3. Verifier (`verifiers/`)

The escalation gate. Returns a `Verdict(passed, confidence, cost)`. The whole
system rests on the verifier's ability to *know when the cheap model was wrong*.
Four implementations trade cost against reliability — see [verifiers.md](verifiers.md).

## 4. Policy (`policy.py`) + calibration (`calibration.py`)

The policy accepts when `verdict.confidence >= tier_threshold`. Thresholds are
the tuning knob, and `calibration.py` fits them offline against a labelled set to
hit a target accuracy at minimum cost — see [calibration.md](calibration.md).

## Cost accounting

Every model call **and every verifier call** is charged to a `Budget` (cost,
latency, call-count caps). Verifier overhead is real money — a consistency check
at n=5 costs ~5 model calls — so the accounting includes it. This is what lets
calibration reason about true cost, not just model-answer cost.
