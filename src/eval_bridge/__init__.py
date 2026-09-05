"""eval-bridge: capture LLM failures, scrub PII, replay offline as eval fixtures."""

from .errors import (
    EvalBridgeError,
    FixtureError,
    ProviderError,
    ScrubberError,
)
from .fixture import Fixture, FixtureSet, load_fixture, load_fixture_dir
from .runner import (
    AssertionReport,
    Runner,
    RunnerConfig,
    RunnerReport,
    TestCaseResult,
    jaccard,
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
from .mutator import mutate_fixture
from .provider import (
    FixtureProvider,
    OpenAICompatProvider,
    Provider,
    provider_from_config,
)

__all__ = [
    "AssertionReport",
    "Config",
    "DEFAULT_PATTERNS",
    "EvalBridgeError",
    "Fixture",
    "FixtureError",
    "FixtureProvider",
    "FixtureSet",
    "OpenAICompatProvider",
    "Pattern",
    "Provider",
    "ProviderError",
    "Runner",
    "RunnerConfig",
    "RunnerReport",
    "ScrubResult",
    "Scrubber",
    "ScrubberConfig",
    "ScrubberError",
    "TestCaseResult",
    "find_residual_secrets",
    "jaccard",
    "load_config",
    "load_fixture",
    "load_fixture_dir",
    "mutate_fixture",
    "provider_from_config",
]

__version__ = "0.2.0"
