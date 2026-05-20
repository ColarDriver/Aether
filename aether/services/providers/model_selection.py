"""Provider/model selection service."""

from __future__ import annotations

from collections.abc import Mapping

from aether.config.provider_runtime import resolve_main_provider_runtime
from aether.services.common import ServiceValidationError
from aether.services.config import PrefsService
from aether.services.providers.auth import AuthService
from aether.services.providers.contracts import (
    ProviderSelectionRequest,
    ProviderSelectionResult,
)


class ModelSelectionService:
    def __init__(
        self,
        *,
        prefs: PrefsService | None = None,
        auth: AuthService | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._prefs = prefs or PrefsService()
        self._auth = auth or AuthService(environ=environ)
        self._environ = environ

    def select(self, request: ProviderSelectionRequest) -> ProviderSelectionResult:
        try:
            initial_runtime = resolve_main_provider_runtime(
                environ=self._environ,
                provider=request.provider,
                model=None,
                base_url=request.base_url,
            )
            preferred_model = (
                request.model
                or self._prefs.get_last_model(initial_runtime.provider_name)
                or initial_runtime.model
            )
            runtime = resolve_main_provider_runtime(
                environ=self._environ,
                provider=request.provider,
                model=preferred_model,
                base_url=request.base_url,
            )
        except ValueError as exc:
            raise ServiceValidationError(str(exc)) from exc
        credential = self._auth.first_credential(runtime)
        missing = [
            key_name
            for key_name in runtime.api_key_env_names
            if credential is None or credential.name != key_name
        ]
        if request.persist_last_model:
            self._prefs.set_last_model(runtime.provider_name, runtime.model)
        return ProviderSelectionResult(
            provider=runtime.provider_name,
            family=runtime.family,
            model=runtime.model,
            base_url=runtime.base_url,
            ready=credential is not None or not runtime.api_key_env_names,
            missing_credentials=[] if credential is not None else missing,
            credential=credential,
        )


__all__ = ["ModelSelectionService"]
