"""cascade — route each LLM call to the cheapest model that will get it right.

Public surface:

    from cascade import Router, Cascade, Tier, Budget
    from cascade.llm import MockLLM, AnthropicLLM, OpenAILLM
    from cascade.verifiers import RuleVerifier, SelfCheckVerifier, ConsistencyVerifier, JudgeVerifier
    from cascade.policy import ThresholdPolicy
    from cascade.calibration import calibrate, CalibrationItem
"""

from cascade.budget import Budget, BudgetExceeded
from cascade.cascade import Cascade, Tier
from cascade.difficulty import HeuristicDifficulty, LLMDifficulty
from cascade.policy import AlwaysAcceptPolicy, Policy, TargetAccuracyPolicy, ThresholdPolicy
from cascade.router import Router
from cascade.types import Difficulty, LLMResponse, Query, RouteResult, Verdict

__version__ = "0.1.0"

__all__ = [
    "AlwaysAcceptPolicy",
    "Budget",
    "BudgetExceeded",
    "Cascade",
    "Difficulty",
    "HeuristicDifficulty",
    "LLMDifficulty",
    "LLMResponse",
    "Policy",
    "Query",
    "RouteResult",
    "Router",
    "TargetAccuracyPolicy",
    "ThresholdPolicy",
    "Tier",
    "Verdict",
]
