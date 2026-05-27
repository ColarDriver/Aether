"""Configuration loading for external MCP servers.

The runtime accepts explicit ``EngineConfig.mcp_servers`` values and also
supports local config files under ``$AETHER_HOME`` so web/TUI processes share
one integration surface without routing through gateway handlers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Any
from urllib.parse import urlparse


_DEFAULT_TOOL_TIMEOUT_SECONDS = 120.0
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 30.0
_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    transport: str = "stdio"
    timeout: float = _DEFAULT_TOOL_TIMEOUT_SECONDS
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS
    enabled: bool = True

    @property
    def is_remote(self) -> bool:
        return bool(self.url)


def load_mcp_server_configs(
    source: Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    aether_home: Path | None = None,
) -> list[McpServerConfig]:
    """Load enabled MCP server configs from explicit values, env, or files.

    Precedence:
    1. Explicit mapping, usually ``EngineConfig.mcp_servers``.
    2. ``AETHER_MCP_SERVERS`` JSON object.
    3. ``AETHER_MCP_CONFIG`` path.
    4. ``$AETHER_HOME/mcp_servers.json``, ``mcp.json``, ``config.json``,
       ``config.toml``, ``config.yaml``, or ``config.yml``.
    """

    env = environ if environ is not None else os.environ
    raw: Mapping[str, Any] | None = source if source else None
    if raw is None:
        raw = _load_from_env(env)
    if raw is None:
        raw = _load_from_files(env=env, aether_home=aether_home)
    if not raw:
        return []

    servers = raw.get("mcp_servers") if "mcp_servers" in raw else raw
    if not isinstance(servers, Mapping):
        return []

    configs: list[McpServerConfig] = []
    for name, payload in servers.items():
        config = _coerce_server_config(str(name), payload, env)
        if config is not None and config.enabled:
            configs.append(config)
    return configs


def sanitize_mcp_name_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unnamed"


def _load_from_env(env: Mapping[str, str]) -> Mapping[str, Any] | None:
    raw = env.get("AETHER_MCP_SERVERS")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _load_from_files(*, env: Mapping[str, str], aether_home: Path | None) -> Mapping[str, Any] | None:
    explicit = env.get("AETHER_MCP_CONFIG")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    home = aether_home or _aether_home(env)
    candidates.extend(
        [
            home / "mcp_servers.json",
            home / "mcp.json",
            home / "config.json",
            home / "config.toml",
            home / "config.yaml",
            home / "config.yml",
        ]
    )
    for path in candidates:
        payload = _read_config_file(path)
        if payload is not None:
            return payload
    return None


def _read_config_file(path: Path) -> Mapping[str, Any] | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".toml":
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            payload = _load_yaml(path)
        else:
            return None
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except Exception:
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _coerce_server_config(
    name: str,
    payload: Any,
    env: Mapping[str, str],
) -> McpServerConfig | None:
    if not isinstance(payload, Mapping):
        return None
    enabled = bool(payload.get("enabled", True))
    command = _optional_str(payload.get("command"))
    url = _optional_str(payload.get("url"))
    if command and url:
        return None
    if not command and not url:
        return None
    if url is not None:
        try:
            url = _validate_url(name, url)
        except ValueError:
            return None
    args = tuple(str(item) for item in payload.get("args", []) if item is not None) if isinstance(payload.get("args", []), list) else ()
    server_env = _string_mapping(payload.get("env"), env)
    headers = _string_mapping(payload.get("headers"), env)
    transport = _optional_str(payload.get("transport")) or ("http" if url else "stdio")
    return McpServerConfig(
        name=sanitize_mcp_name_component(name),
        command=_expand_env_vars(command, env) if command else None,
        args=tuple(_expand_env_vars(arg, env) for arg in args),
        env=server_env,
        url=url,
        headers=headers,
        transport=transport.strip().lower(),
        timeout=_coerce_float(payload.get("timeout"), _DEFAULT_TOOL_TIMEOUT_SECONDS),
        connect_timeout=_coerce_float(payload.get("connect_timeout"), _DEFAULT_CONNECT_TIMEOUT_SECONDS),
        enabled=enabled,
    )


def _string_mapping(value: Any, env: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        if key is None or item is None:
            continue
        result[str(key)] = _expand_env_vars(str(item), env)
    return result


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _expand_env_vars(value: str, env: Mapping[str, str]) -> str:
    return _ENV_VAR_RE.sub(lambda match: env.get(match.group(1), ""), value)


def _validate_url(server_name: str, value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid MCP URL for {server_name!r}: expected http(s) URL")
    return value


def _aether_home(env: Mapping[str, str]) -> Path:
    raw = env.get("AETHER_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".aether"


__all__ = [
    "McpServerConfig",
    "load_mcp_server_configs",
    "sanitize_mcp_name_component",
]
