"""eval-bridge: capture LLM failures, scrub PII, replay offline as eval fixtures."""

from .errors import (
    EvalBridgeError,
    FixtureError,
    ProviderError,
    ScrubberError,
)
from .errors import EvalBridgeError
from .fixture import Fixture, FixtureSet, load_fixture, load_fixture_dir
from .incident import COUNTER_FILE, next_incident_id
from .runner import (
    AssertionReport,
    Runner,
    RunnerConfig,
    RunnerReport,
    TestCaseResult,
    jaccard,
)
from .scoring import (
    BUILTIN_SCORERS,
    BIAS,
    FAITHFULNESS,
    HALLUCINATION,
    Judge,
    LLMJudge,
    OfflineJudge,
    ScoreResult,
    Scorer,
    TOXICITY,
    ANSWER_RELEVANCE,
    default_scorer_set,
    g_eval,
)
from .scrubber import (
    DEFAULT_PATTERNS,
    Pattern,
    Scrubber,
    ScrubberConfig,
    ScrubResult,
    find_residual_secrets,
)
from .config import Config, load_config
from .importer import import_langfuse, import_otel
from .mutator import mutate_fixture
from .provider import (
    FixtureProvider,
    OpenAICompatProvider,
    Provider,
    provider_from_config,
)

__all__ = [
    "ANSWER_RELEVANCE",
    "AssertionReport",
    "BUILTIN_SCORERS",
    "BIAS",
    "Config",
    "DEFAULT_PATTERNS",
    "EvalBridgeError",
    "FAITHFULNESS",
    "Fixture",
    "FixtureError",
    "FixtureProvider",
    "FixtureSet",
    "HALLUCINATION",
    "IncidentIDError",
    "Judge",
    "LLMJudge",
    "OfflineJudge",
    "OpenAICompatProvider",
    "Pattern",
    "Provider",
    "ProviderError",
    "Runner",
    "RunnerConfig",
    "RunnerReport",
    "ScoreResult",
    "Scorer",
    "ScrubResult",
    "Scrubber",
    "ScrubberConfig",
    "ScrubberError",
    "TOXICITY",
    "TestCaseResult",
    "default_scorer_set",
    "find_residual_secrets",
    "g_eval",
    "import_langfuse",
    "import_otel",
    "jaccard",
    "load_config",
    "load_fixture",
    "load_fixture_dir",
    "mutate_fixture",
    "provider_from_config",
    "COUNTER_FILE",
    "next_incident_id",
]

__version__ = "0.5.0"
