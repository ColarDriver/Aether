"""Environment variable service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EnvValueSource = Literal["file", "process", "missing"]


@dataclass(frozen=True, slots=True)
class EnvVarSummary:
    key: str
    is_set: bool
    source: EnvValueSource
    redacted_value: str | None = None
    description: str = ""
    category: str = "other"
    is_secret: bool = True
    advanced: bool = False
    url: str | None = None


@dataclass(frozen=True, slots=True)
class EnvCatalog:
    env_path: str
    variables: list[EnvVarSummary] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EnvMutationResult:
    ok: bool
    key: str
    env_path: str


@dataclass(frozen=True, slots=True)
class EnvRevealResult:
    key: str
    value: str
    source: EnvValueSource


@dataclass(frozen=True, slots=True)
class EnvRevealAuditEntry:
    key: str
    source: EnvValueSource
    revealed_at: float


__all__ = [
    "EnvCatalog",
    "EnvMutationResult",
    "EnvRevealAuditEntry",
    "EnvRevealResult",
    "EnvValueSource",
    "EnvVarSummary",
]
