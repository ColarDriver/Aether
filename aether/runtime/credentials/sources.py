"""Credential lookup contracts and environment source."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Protocol

from aether.runtime.credentials.redaction import redact_secret


@dataclass(frozen=True, slots=True)
class CredentialValue:
    """A raw credential plus source metadata.

    ``repr`` deliberately excludes the raw value; use ``value`` only at the
    provider construction boundary.
    """

    value: str
    source: str
    key_name: str

    def redacted(self) -> str:
        return redact_secret(self.value)

    def public_metadata(self) -> dict[str, object]:
        return {
            "source": self.source,
            "name": self.key_name,
            "configured": bool(self.value),
            "redacted": self.redacted(),
        }

    def __repr__(self) -> str:
        return (
            "CredentialValue("
            f"source={self.source!r}, key_name={self.key_name!r}, "
            f"value={self.redacted()!r})"
        )


class CredentialSource(Protocol):
    name: str

    def get(self, key_name: str) -> CredentialValue | None: ...


@dataclass(frozen=True, slots=True)
class EnvCredentialSource:
    """Credential source backed by an environment mapping."""

    environ: Mapping[str, str] | None = None
    name: str = "env"

    def get(self, key_name: str) -> CredentialValue | None:
        env = self.environ if self.environ is not None else os.environ
        value = env.get(key_name)
        if not isinstance(value, str) or not value:
            return None
        return CredentialValue(value=value, source=self.name, key_name=key_name)


@dataclass(frozen=True, slots=True)
class CredentialLookup:
    """Try credential sources in order."""

    sources: tuple[CredentialSource, ...]

    def get_first(self, key_names: tuple[str, ...]) -> CredentialValue | None:
        for key_name in key_names:
            for source in self.sources:
                found = source.get(key_name)
                if found is not None:
                    return found
        return None


def default_credential_lookup(*, environ: Mapping[str, str] | None = None) -> CredentialLookup:
    return CredentialLookup((EnvCredentialSource(environ=environ),))


__all__ = [
    "CredentialLookup",
    "CredentialSource",
    "CredentialValue",
    "EnvCredentialSource",
    "default_credential_lookup",
]
