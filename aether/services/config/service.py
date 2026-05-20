"""Configuration service implementation."""

from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
from typing import Any

from aether.cli.prefs import _prefs_file
from aether.cli.sessions import default_session_dir
from aether.config.schema import EngineConfig
from aether.services.config.contracts import (
    ConfigDefaults,
    ConfigPaths,
    EffectiveConfig,
    EnvironmentPathStatus,
)


class ConfigService:
    def __init__(self, *, config: EngineConfig | None = None) -> None:
        self._config = config or EngineConfig()

    def effective(self) -> EffectiveConfig:
        return EffectiveConfig(values=_public_config_values(self._config))

    def defaults(self) -> ConfigDefaults:
        return ConfigDefaults(values=_public_config_values(EngineConfig()))

    def paths(self) -> ConfigPaths:
        return ConfigPaths(
            aether_home=str(_aether_home()),
            sessions_dir=str(default_session_dir()),
            prefs_file=str(_prefs_file()),
        )

    def environment_paths(self) -> list[EnvironmentPathStatus]:
        paths = self.paths()
        return [
            _path_status("AETHER_HOME", paths.aether_home),
            _path_status("AETHER_SESSIONS", paths.sessions_dir),
            _path_status("AETHER_PREFS", paths.prefs_file),
        ]


def _aether_home() -> Path:
    return Path(os.getenv("AETHER_HOME", Path.home() / ".aether")).expanduser()


def _path_status(name: str, path: str) -> EnvironmentPathStatus:
    candidate = Path(path).expanduser()
    return EnvironmentPathStatus(
        name=name,
        path=str(candidate),
        exists=candidate.exists(),
    )


def _public_config_values(config: EngineConfig) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in fields(config):
        value = getattr(config, item.name)
        if "key" in item.name.lower() or "token" in item.name.lower():
            values[item.name] = bool(value)
        elif isinstance(value, Path):
            values[item.name] = str(value)
        elif isinstance(value, tuple):
            values[item.name] = [str(part) if isinstance(part, Path) else part for part in value]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            values[item.name] = value
    return values


__all__ = ["ConfigService"]
