"""Provider authentication readiness service."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aether.config.provider_runtime import ProviderRuntimeConfig, resolve_main_provider_runtime
from aether.runtime.credentials import default_credential_lookup
from aether.services.common import ServiceValidationError
from aether.services.providers.contracts import CredentialSetStatus, CredentialStatus


class AuthService:
    def __init__(self, *, environ: Mapping[str, str] | None = None) -> None:
        self._environ = environ

    def credentials_status(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> CredentialSetStatus:
        try:
            runtime = resolve_main_provider_runtime(
                environ=self._environ,
                provider=provider,
                model=model,
                base_url=base_url,
            )
        except ValueError as exc:
            raise ServiceValidationError(str(exc)) from exc
        return self.runtime_credentials_status(runtime)

    def runtime_credentials_status(self, runtime: ProviderRuntimeConfig) -> CredentialSetStatus:
        lookup = default_credential_lookup(environ=self._environ)
        return CredentialSetStatus(
            family=runtime.family,
            provider=runtime.provider_name,
            credentials=[
                _credential_status(key_name, lookup.get_first((key_name,)))
                for key_name in runtime.api_key_env_names
            ],
        )

    def first_credential(self, runtime: ProviderRuntimeConfig) -> CredentialStatus | None:
        credential = default_credential_lookup(environ=self._environ).get_first(
            runtime.api_key_env_names
        )
        if credential is None:
            return None
        return _credential_status(credential.key_name, credential)


def _credential_status(key_name: str, credential: Any | None) -> CredentialStatus:
    if credential is None:
        return CredentialStatus(
            source="env",
            name=key_name,
            configured=False,
            redacted="",
        )
    public = credential.public_metadata()
    return CredentialStatus(
        source=str(public.get("source") or "env"),
        name=str(public.get("name") or key_name),
        configured=bool(public.get("configured")),
        redacted=str(public.get("redacted") or ""),
    )


__all__ = ["AuthService"]
