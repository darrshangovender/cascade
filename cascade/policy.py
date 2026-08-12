"""Escalation policies.

A policy answers one question at each tier: *given this verdict, do we accept
the answer or escalate to the next tier?* The threshold per tier is what the
calibration module tunes.

Three policies ship:

- ``ThresholdPolicy``  — escalate when verdict.confidence < tier_threshold.
  The workhorse; thresholds come from calibration.
- ``AlwaysAcceptPolicy`` / ``NeverEscalatePolicy`` — degenerate baselines used
  in tests and to measure "cheapest model only".
- ``TargetAccuracyPolicy`` — wraps ThresholdPolicy but exposes the target the
  thresholds were calibrated for (documentation / introspection).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cascade.types import Verdict


class Policy(ABC):
    @abstractmethod
    def accept(self, tier: int, verdict: Verdict, is_last_tier: bool) -> bool:
        """Return True to accept the answer, False to escalate."""


class ThresholdPolicy(Policy):
    """Accept when confidence clears the per-tier threshold (or verdict passed
    outright on the last tier)."""

    def __init__(self, thresholds: list[float]) -> None:
        if not thresholds:
            raise ValueError("need at least one threshold")
        self.thresholds = thresholds

    def accept(self, tier: int, verdict: Verdict, is_last_tier: bool) -> bool:
        if is_last_tier:
            return True  # nowhere left to escalate
        thr = self.thresholds[min(tier, len(self.thresholds) - 1)]
        return verdict.passed and verdict.confidence >= thr


class AlwaysAcceptPolicy(Policy):
    """Accept the first tier's answer unconditionally (cheapest-only baseline)."""

    def accept(self, tier: int, verdict: Verdict, is_last_tier: bool) -> bool:
        return True


class NeverEscalatePolicy(AlwaysAcceptPolicy):
    """Alias for the cheapest-only baseline."""


class TargetAccuracyPolicy(ThresholdPolicy):
    """ThresholdPolicy that remembers the accuracy target it was tuned for."""

    def __init__(self, thresholds: list[float], target_accuracy: float) -> None:
        super().__init__(thresholds)
        self.target_accuracy = target_accuracy
