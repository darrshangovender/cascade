"""Provider-portable LLM clients.

Every model in a cascade is an ``LLM`` — a small protocol with one method,
``complete``. Three implementations ship:

- ``MockLLM``  — deterministic, offline, zero-cost-to-run. Used by the test
  suite AND the benchmark, so the whole repo is reproducible with no API keys.
- ``AnthropicLLM`` / ``OpenAILLM`` — thin wrappers over the real SDKs, lazily
  imported so importing ``cascade`` never requires the provider libraries.

The MockLLM is the interesting one. It models a *capability tier*: given a
benchmark item whose gold answer is embedded in the prompt via a
``[GOLD:...]`` sentinel, the mock returns the correct answer with a probability
that rises with the model's ``skill`` and falls with the item's ``difficulty``.
The randomness is seeded by hashing the prompt, so a given (model, prompt) pair
is deterministic — tests are stable and benchmarks are reproducible.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

from cascade.types import LLMResponse

# Rough public per-1k-token prices (USD), used for cost accounting.
PRICES: dict[str, tuple[float, float]] = {
    # model: (prompt_per_1k, completion_per_1k)
    "claude-haiku-4-5": (0.0008, 0.004),
    "claude-sonnet-4-5": (0.003, 0.015),
    "claude-opus-4-7": (0.015, 0.075),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}


@runtime_checkable
class LLM(Protocol):
    """Anything that can turn a prompt into a completion with cost accounting."""

    model: str
    tier: int

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse: ...


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 chars/token. Good enough for accounting."""
    return max(1, len(text) // 4)


def price_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICES.get(model, (0.001, 0.002))
    return (prompt_tokens / 1000) * p_in + (completion_tokens / 1000) * p_out


_GOLD_RE = re.compile(r"\[GOLD:(.*?)\]", re.DOTALL)


class MockLLM:
    """Deterministic offline model used by tests + benchmarks.

    Parameters
    ----------
    model, tier
        Identity + position in the cascade (0 = cheapest).
    skill
        Capability in [0, 1]. Higher = answers hard questions more often.
    latency_ms
        Simulated per-call latency for the trace.
    """

    def __init__(self, model: str, tier: int, skill: float, latency_ms: float = 50.0) -> None:
        self.model = model
        self.tier = tier
        self.skill = skill
        self.latency_ms = latency_ms

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        # When handed a grader-style system prompt, role-play a grader instead
        # of an answerer, so the self-check / judge verifiers work offline too.
        if system and _is_grading(system):
            text = self._grade(prompt, system)
        else:
            text = self._answer(prompt)
        pt = _estimate_tokens(prompt)
        ct = _estimate_tokens(text)
        return LLMResponse(
            text=text,
            model=self.model,
            tier=self.tier,
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=price_of(self.model, pt, ct),
            latency_ms=self.latency_ms,
        )

    def _answer(self, prompt: str) -> str:
        gold, difficulty = self._parse(prompt)
        seed = self._seed(prompt)
        # Probability of getting it right: logistic in (skill - difficulty).
        p_correct = _logistic(6.0 * (self.skill - difficulty))
        roll = (seed % 10_000) / 10_000.0
        if gold is not None and roll < p_correct:
            return gold
        if gold is not None:
            return _plausible_wrong(gold, seed)
        return f"[{self.model} answer]"

    def _grade(self, prompt: str, system: str) -> str:
        """Role-play a grader: judge whether the embedded candidate matches gold.

        The grader's *own* judgement is correct with probability ``skill`` — a
        stronger model is a more reliable verifier. Emits the format the calling
        verifier parses (VERDICT/CONFIDENCE for self-check, SCORE for judge).
        """
        gold, _ = self._parse(prompt)
        candidate = _extract_candidate(prompt)
        truth = _loose_match(candidate, gold) if gold is not None else True
        seed = self._seed(prompt)
        grader_right = (seed % 10_000) / 10_000.0 < self.skill
        verdict = truth if grader_right else (not truth)
        if "SCORE" in system.upper():  # judge rubric
            score = 9 if verdict else 2
            return f"SCORE: {score}, REASON: {'consistent with expected' if verdict else 'appears incorrect'}"
        yn = "YES" if verdict else "NO"
        conf = 88 if grader_right else 55
        return f"VERDICT: {yn}, CONFIDENCE: {conf}"

    @staticmethod
    def _parse(prompt: str) -> tuple[str | None, float]:
        m = _GOLD_RE.search(prompt)
        if not m:
            return None, 0.5
        payload = m.group(1)
        # payload format: "answer||difficulty"
        if "||" in payload:
            answer, diff = payload.split("||", 1)
            try:
                return answer.strip(), float(diff)
            except ValueError:
                return answer.strip(), 0.5
        return payload.strip(), 0.5

    def _seed(self, prompt: str) -> int:
        h = hashlib.sha256(f"{self.model}:{prompt}".encode()).hexdigest()
        return int(h[:8], 16)


def _is_grading(system: str) -> bool:
    s = system.lower()
    return "grader" in s or "score the candidate" in s or "candidate answer" in s


_CANDIDATE_RE = re.compile(r"Candidate answer:\s*(.+?)(?:\n\n|\n\[GOLD|\Z)", re.DOTALL)


def _extract_candidate(prompt: str) -> str:
    m = _CANDIDATE_RE.search(prompt)
    return m.group(1).strip() if m else ""


def _loose_match(a: str, gold: str) -> bool:
    a, g = a.strip().lower(), gold.strip().lower()
    return bool(g) and (g in a or a in g or a == g)


def _logistic(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


def _plausible_wrong(gold: str, seed: int) -> str:
    """Produce a deterministic wrong-but-plausible answer."""
    if gold.strip().lstrip("-").isdigit():
        return str(int(gold) + (1 if seed % 2 else -1))
    return f"{gold} (incorrect variant {seed % 97})"


class AnthropicLLM:
    """Wrapper over the Anthropic SDK (lazily imported)."""

    def __init__(self, model: str, tier: int, max_tokens: int = 1024) -> None:
        self.model = model
        self.tier = tier
        self.max_tokens = max_tokens
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic  # lazy

            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        client = self._ensure()
        import time

        t0 = time.perf_counter()
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.perf_counter() - t0) * 1000
        text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
        pt = msg.usage.input_tokens
        ct = msg.usage.output_tokens
        return LLMResponse(text, self.model, self.tier, pt, ct, price_of(self.model, pt, ct), latency)


class OpenAILLM:
    """Wrapper over the OpenAI SDK (lazily imported)."""

    def __init__(self, model: str, tier: int, max_tokens: int = 1024) -> None:
        self.model = model
        self.tier = tier
        self.max_tokens = max_tokens
        self._client = None

    def _ensure(self):
        if self._client is None:
            import openai  # lazy

            self._client = openai.OpenAI()
        return self._client

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        client = self._ensure()
        import time

        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        latency = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        pt = resp.usage.prompt_tokens
        ct = resp.usage.completion_tokens
        return LLMResponse(text, self.model, self.tier, pt, ct, price_of(self.model, pt, ct), latency)
