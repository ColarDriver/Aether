"""Provider, auth, and model-selection services."""

from aether.services.providers.auth import AuthService
from aether.services.providers.contracts import (
    AuxiliarySlotStatus,
    CredentialSetStatus,
    CredentialStatus,
    ModelDiscoveryStatus,
    ModelSummary,
    ProviderModelList,
    ProviderPreflightStatus,
    ProviderRuntimeStatus,
    ProviderSelectionRequest,
    ProviderSelectionResult,
    ProviderSummary,
)
from aether.services.providers.model_selection import ModelSelectionService
from aether.services.providers.service import (
    DISCOVERY_PATH_PROBES,
    MODEL_CATALOG,
    PROVIDER_DISPLAY,
    ProviderService,
    candidate_urls,
    extract_model_ids,
    suggest_base_url,
)

__all__ = [
    "AuxiliarySlotStatus",
    "AuthService",
    "CredentialSetStatus",
    "CredentialStatus",
    "DISCOVERY_PATH_PROBES",
    "MODEL_CATALOG",
    "ModelDiscoveryStatus",
    "ModelSelectionService",
    "ModelSummary",
    "PROVIDER_DISPLAY",
    "ProviderModelList",
    "ProviderPreflightStatus",
    "ProviderRuntimeStatus",
    "ProviderSelectionRequest",
    "ProviderSelectionResult",
    "ProviderService",
    "ProviderSummary",
    "candidate_urls",
    "extract_model_ids",
    "suggest_base_url",
]
