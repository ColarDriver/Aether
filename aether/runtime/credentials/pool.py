"""Optional local credential pool with in-process rotation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, cast

from aether.runtime.credentials.sources import CredentialValue, default_credential_lookup


CredentialPoolStrategy = Literal["fill_first", "round_robin"]


@dataclass(slots=True)
class PooledCredential:
    provider: str
    name: str
    credential: CredentialValue
    healthy: bool = True
    last_error: str | None = None

    def public_metadata(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "name": self.name,
            "source": self.credential.source,
            "key_name": self.credential.key_name,
            "configured": bool(self.credential.value),
            "redacted": self.credential.redacted(),
            "healthy": self.healthy,
            "last_error": self.last_error,
        }


@dataclass(slots=True)
class CredentialPoolSelection:
    credential: PooledCredential
    strategy: CredentialPoolStrategy

    def public_metadata(self) -> dict[str, object]:
        return {
            "provider": self.credential.provider,
            "credential_name": self.credential.name,
            "strategy": self.strategy,
            "source": self.credential.credential.source,
            "key_name": self.credential.credential.key_name,
        }


@dataclass(slots=True)
class CredentialPool:
    providers: dict[str, list[PooledCredential]] = field(default_factory=dict)
    strategy: CredentialPoolStrategy = "fill_first"
    _cursors: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "CredentialPool":
        raw_strategy = str(mapping.get("strategy") or "fill_first")
        if raw_strategy not in {"fill_first", "round_robin"}:
            raise ValueError("credential pool strategy must be fill_first or round_robin")
        strategy = cast(CredentialPoolStrategy, raw_strategy)
        raw_providers = mapping.get("providers")
        if not isinstance(raw_providers, Mapping):
            return cls(strategy=strategy)
        lookup = default_credential_lookup(environ=environ)
        providers: dict[str, list[PooledCredential]] = {}
        for provider, entries in raw_providers.items():
            if not isinstance(provider, str) or not isinstance(entries, list):
                continue
            resolved: list[PooledCredential] = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    continue
                env_name = entry.get("api_key_env")
                if not isinstance(env_name, str) or not env_name.strip():
                    continue
                credential = lookup.get_first((env_name.strip(),))
                if credential is None:
                    continue
                name = entry.get("name")
                resolved.append(
                    PooledCredential(
                        provider=provider,
                        name=str(name).strip() if isinstance(name, str) and name.strip() else f"key-{index + 1}",
                        credential=credential,
                    )
                )
            if resolved:
                providers[provider] = resolved
        return cls(providers=providers, strategy=strategy)

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "CredentialPool":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("credential pool file must contain an object")
        return cls.from_mapping(raw, environ=environ)

    def select(self, provider: str) -> CredentialPoolSelection | None:
        entries = self.providers.get(provider) or []
        healthy = [entry for entry in entries if entry.healthy]
        if not healthy:
            return None
        if self.strategy == "fill_first":
            return CredentialPoolSelection(healthy[0], self.strategy)
        cursor = self._cursors.get(provider, 0)
        selected = healthy[cursor % len(healthy)]
        self._cursors[provider] = cursor + 1
        return CredentialPoolSelection(selected, self.strategy)

    def mark_unhealthy(
        self,
        provider: str,
        credential_name: str,
        *,
        reason: str | None = None,
    ) -> bool:
        for entry in self.providers.get(provider) or []:
            if entry.name == credential_name:
                entry.healthy = False
                entry.last_error = reason
                return True
        return False

    def rotate_after_error(
        self,
        current: CredentialPoolSelection,
        *,
        reason: str,
    ) -> CredentialPoolSelection | None:
        self.mark_unhealthy(
            current.credential.provider,
            current.credential.name,
            reason=reason,
        )
        return self.select(current.credential.provider)

    def public_metadata(self) -> dict[str, object]:
        return {
            "enabled": any(self.providers.values()),
            "strategy": self.strategy,
            "providers": {
                provider: [entry.public_metadata() for entry in entries]
                for provider, entries in self.providers.items()
            },
        }


__all__ = [
    "CredentialPool",
    "CredentialPoolSelection",
    "CredentialPoolStrategy",
    "PooledCredential",
]
