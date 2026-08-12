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
from cascade.policy import Policy, ThresholdPolicy, AlwaysAcceptPolicy, TargetAccuracyPolicy
from cascade.router import Router
from cascade.types import Difficulty, Query, RouteResult, Verdict, LLMResponse

__version__ = "0.1.0"

__all__ = [
    "Router",
    "Cascade",
    "Tier",
    "Budget",
    "BudgetExceeded",
    "Policy",
    "ThresholdPolicy",
    "AlwaysAcceptPolicy",
    "TargetAccuracyPolicy",
    "HeuristicDifficulty",
    "LLMDifficulty",
    "Difficulty",
    "Query",
    "RouteResult",
    "Verdict",
    "LLMResponse",
]
