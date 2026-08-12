"""Synthetic labelled benchmark set with a controlled difficulty distribution.

Deterministic (seeded), so the whole benchmark is reproducible offline.
Each item embeds its gold answer + difficulty via the [GOLD:...] sentinel the
MockLLM reads.
"""

from __future__ import annotations

import random

from cascade.calibration import CalibrationItem
from cascade.types import Query


def make_dataset(n: int = 200, seed: int = 42) -> list[CalibrationItem]:
    rng = random.Random(seed)
    items: list[CalibrationItem] = []
    for i in range(n):
        # Realistic production traffic is dominated by easy queries with a
        # long tail of hard ones — roughly 60% easy / 25% medium / 15% hard.
        # This skew is *why* cascades save money: the cheap model handles the
        # bulk, and the expensive model is reserved for the residual tail.
        roll = rng.random()
        if roll < 0.60:
            difficulty = rng.uniform(0.05, 0.30)
        elif roll < 0.85:
            difficulty = rng.uniform(0.30, 0.60)
        else:
            difficulty = rng.uniform(0.60, 0.95)
        difficulty = round(difficulty, 2)
        gold = str(rng.randint(0, 999))
        prompt = f"Benchmark item {i} (difficulty {difficulty}). Answer with a number. [GOLD:{gold}||{difficulty}]"
        items.append(CalibrationItem(query=Query(prompt), gold=gold))
    return items
