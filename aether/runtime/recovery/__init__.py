"""Provider recovery, failover, and request-shaping helpers."""

from .error_classifier import FailoverReason, classify_api_error
from .fallback_chain import FallbackChain, ProviderSlot
from .provider_errors import ProviderInvocationError, ResponseInvalidError
from .rate_guard import RateGuard, RateGuardCheck, default_rate_guard_dir, provider_rate_guard_key
from .strategies import (
    AttemptState,
    ClassifiedRecoveryStrategy,
    GenericBackoffStrategy,
    NoRetryStrategy,
    RecoveryDecision,
    RecoveryStrategy,
)

__all__ = [
    "FailoverReason",
    "classify_api_error",
    "FallbackChain",
    "ProviderSlot",
    "ProviderInvocationError",
    "ResponseInvalidError",
    "RateGuard",
    "RateGuardCheck",
    "default_rate_guard_dir",
    "provider_rate_guard_key",
    "AttemptState",
    "ClassifiedRecoveryStrategy",
    "GenericBackoffStrategy",
    "NoRetryStrategy",
    "RecoveryDecision",
    "RecoveryStrategy",
]
