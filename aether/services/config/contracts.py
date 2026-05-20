"""Configuration and preference service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EffectiveConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConfigPaths:
    aether_home: str
    sessions_dir: str
    prefs_file: str


@dataclass(frozen=True, slots=True)
class EnvironmentPathStatus:
    name: str
    path: str
    exists: bool


@dataclass(frozen=True, slots=True)
class ConfigDefaults:
    values: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ConfigDefaults",
    "ConfigPaths",
    "EffectiveConfig",
    "EnvironmentPathStatus",
]
