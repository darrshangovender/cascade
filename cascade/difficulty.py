"""Difficulty estimation.

The router uses a difficulty estimate to pick the *entry* tier of the cascade —
an easy question starts (and usually ends) at the cheapest model, a hard one may
skip straight to a mid tier to avoid a wasted cheap call.

Two estimators ship:

- ``HeuristicDifficulty`` — zero-cost, offline. Scores length, multi-step cue
  words ("prove", "step by step", "explain why"), numeric density, and clause
  count. Fast, deterministic, no dependencies.
- ``LLMDifficulty`` — one cheap classifier call. More accurate on adversarial
  phrasing, but costs a call; use when the entry-tier decision is expensive to
  get wrong.
"""

from __future__ import annotations

import re

from cascade.llm import LLM
from cascade.types import Difficulty, Query

_MULTISTEP_CUES = (
    "step by step",
    "prove",
    "explain why",
    "derive",
    "how many",
    "calculate",
    "reason",
    "compare",
    "trade-off",
    "tradeoff",
    "design",
)


class HeuristicDifficulty:
    """Cheap, dependency-free difficulty scorer."""

    def __init__(self, easy_max: float = 0.33, hard_min: float = 0.60) -> None:
        self.easy_max = easy_max
        self.hard_min = hard_min

    def score(self, query: Query) -> float:
        """Return a difficulty score in [0, 1]."""
        text = query.text.lower()
        n_words = len(text.split())
        length_score = min(1.0, n_words / 60.0)
        cue_hits = sum(1 for c in _MULTISTEP_CUES if c in text)
        cue_score = min(1.0, cue_hits / 3.0)
        numeric_density = len(re.findall(r"\d", text)) / max(1, len(text))
        numeric_score = min(1.0, numeric_density * 20.0)
        clause_score = min(1.0, text.count(",") / 5.0 + text.count(";") / 2.0)
        # Weighted blend. Multi-step cue words ("prove", "derive", "step by
        # step") are the strongest difficulty signal, so they carry the most
        # weight — length alone is a weak proxy.
        return (
            0.20 * length_score
            + 0.50 * cue_score
            + 0.15 * numeric_score
            + 0.15 * clause_score
        )

    def classify(self, query: Query) -> Difficulty:
        s = self.score(query)
        if s <= self.easy_max:
            return Difficulty.EASY
        if s >= self.hard_min:
            return Difficulty.HARD
        return Difficulty.MEDIUM


class LLMDifficulty:
    """One-call difficulty classifier using a cheap model."""

    SYSTEM = (
        "You rate how much reasoning a question needs. "
        "Reply with exactly one word: EASY, MEDIUM, or HARD."
    )

    def __init__(self, model: LLM) -> None:
        self.model = model

    def classify(self, query: Query) -> Difficulty:
        resp = self.model.complete(query.text, system=self.SYSTEM)
        token = resp.text.strip().upper()
        for d in Difficulty:
            if d.value.upper() in token:
                return d
        return Difficulty.MEDIUM


def entry_tier(difficulty: Difficulty, n_tiers: int) -> int:
    """Map a difficulty band to a cascade entry tier index."""
    if n_tiers <= 1:
        return 0
    if difficulty is Difficulty.EASY:
        return 0
    if difficulty is Difficulty.HARD:
        # Start one below the top tier (leave room to escalate once).
        return max(0, n_tiers - 2)
    return min(1, n_tiers - 1)
