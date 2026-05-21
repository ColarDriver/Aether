"""Read and edit Aether local environment variables."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
import os
import time
from pathlib import Path
import re
from typing import Any

from aether.services.common import ServiceConflictError, ServiceNotFoundError, ServiceValidationError
from aether.services.environment.contracts import (
    EnvCatalog,
    EnvRevealAuditEntry,
    EnvMutationResult,
    EnvRevealResult,
    EnvValueSource,
    EnvVarSummary,
)

_KEY_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")

_ENV_DEFINITIONS: dict[str, dict[str, Any]] = {
    "AETHER_PROVIDER": {
        "description": "Global provider family: codex, claude, or openai-compatible.",
        "category": "runtime",
        "is_secret": False,
    },
    "AETHER_MODEL": {
        "description": "Default model used by the active provider.",
        "category": "runtime",
        "is_secret": False,
    },
    "AETHER_BASE_URL": {
        "description": "Optional CLI launcher base URL override.",
        "category": "runtime",
        "is_secret": False,
        "advanced": True,
    },
    "AETHER_ENV": {
        "description": "Local runtime environment label.",
        "category": "runtime",
        "is_secret": False,
        "advanced": True,
    },
    "OPENAI_API_KEY": {
        "description": "OpenAI-compatible API key.",
        "category": "provider",
        "url": "https://platform.openai.com/api-keys",
    },
    "OPENAI_BASE_URL": {
        "description": "OpenAI-compatible base URL.",
        "category": "provider",
        "is_secret": False,
    },
    "ANTHROPIC_API_KEY": {
        "description": "Anthropic API key for Claude provider.",
        "category": "provider",
        "url": "https://console.anthropic.com/settings/keys",
    },
    "ANTHROPIC_AUTH_TOKEN": {
        "description": "Optional Anthropic-compatible auth token used by some gateways.",
        "category": "provider",
        "advanced": True,
    },
    "ANTHROPIC_MODEL": {
        "description": "Optional Anthropic model override.",
        "category": "provider",
        "is_secret": False,
        "advanced": True,
    },
    "ANTHROPIC_BASE_URL": {
        "description": "Optional Anthropic-compatible base URL override.",
        "category": "provider",
        "is_secret": False,
        "advanced": True,
    },
    "CODEX_ACCESS_TOKEN": {
        "description": "Codex access token override; normally loaded from Codex CLI credentials.",
        "category": "provider",
        "advanced": True,
    },
    "CODEX_API_KEY": {
        "description": "Optional Codex API key or compatibility credential.",
        "category": "provider",
    },
    "CODEX_AUTH_PATH": {
        "description": "Path to Codex CLI auth JSON.",
        "category": "provider",
        "is_secret": False,
        "advanced": True,
    },
    "NOUS_API_KEY": {
        "description": "Nous API key for compatible hosted providers.",
        "category": "provider",
    },
    "WEB_SEARCH_PROVIDER": {
        "description": "Local web search backend: brave, tavily, or bocha.",
        "category": "tools",
        "is_secret": False,
    },
    "WEB_SEARCH_API_KEY": {
        "description": "API key for the selected local web search backend.",
        "category": "tools",
    },
}


@dataclass(slots=True)
class _EnvLine:
    raw: str
    key: str | None = None
    value: str | None = None


class EnvironmentService:
    """Manage the project .env file without exposing secrets by default."""

    def __init__(
        self,
        *,
        env_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
        definitions: Mapping[str, Mapping[str, Any]] | None = None,
        reveal_limit: int = 5,
        reveal_window_seconds: float = 60.0,
        monotonic: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
    ) -> None:
        default_path = Path(os.getenv("AETHER_ENV_PATH") or (Path.cwd() / ".env"))
        self._env_path = env_path.expanduser() if env_path is not None else default_path.expanduser()
        self._environ = environ if environ is not None else os.environ
        self._definitions = dict(definitions or _ENV_DEFINITIONS)
        self._reveal_limit = max(0, int(reveal_limit))
        self._reveal_window_seconds = max(1.0, float(reveal_window_seconds))
        self._monotonic = monotonic or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._reveal_attempts: list[float] = []
        self._reveal_audit: list[EnvRevealAuditEntry] = []

    @property
    def env_path(self) -> Path:
        return self._env_path

    def catalog(self) -> EnvCatalog:
        file_values = self._read_values()
        keys = sorted(set(self._definitions) | set(file_values))
        variables = [self._summary(key, file_values) for key in keys]
        return EnvCatalog(env_path=str(self._env_path), variables=variables)

    def set(self, key: str, value: str) -> EnvMutationResult:
        normalized = _validate_key(key)
        if not isinstance(value, str):
            raise ServiceValidationError("environment value must be a string", details={"key": normalized})
        lines = self._read_lines()
        updated = False
        for line in lines:
            if line.key == normalized:
                line.value = value
                line.raw = _format_env_line(normalized, value)
                updated = True
        if not updated:
            if lines and lines[-1].raw.strip():
                lines.append(_EnvLine(raw=""))
            lines.append(_EnvLine(raw=_format_env_line(normalized, value), key=normalized, value=value))
        self._write_lines(lines)
        self._set_process_value(normalized, value)
        return EnvMutationResult(ok=True, key=normalized, env_path=str(self._env_path))

    def delete(self, key: str) -> EnvMutationResult:
        normalized = _validate_key(key)
        lines = self._read_lines()
        kept = [line for line in lines if line.key != normalized]
        if len(kept) == len(lines):
            raise ServiceNotFoundError(f"environment key not found: {normalized}", details={"key": normalized})
        self._write_lines(kept)
        self._delete_process_value(normalized)
        return EnvMutationResult(ok=True, key=normalized, env_path=str(self._env_path))

    def reveal(self, key: str) -> EnvRevealResult:
        normalized = _validate_key(key)
        self._check_reveal_rate(normalized)
        file_values = self._read_values()
        if normalized in file_values:
            return self._record_reveal(normalized, file_values[normalized], "file")
        value = self._environ.get(normalized)
        if value is not None:
            return self._record_reveal(normalized, value, "process")
        raise ServiceNotFoundError(f"environment key not found: {normalized}", details={"key": normalized})

    def reveal_audit(self) -> list[EnvRevealAuditEntry]:
        return list(self._reveal_audit)

    def _check_reveal_rate(self, key: str) -> None:
        if self._reveal_limit <= 0:
            return
        now = self._monotonic()
        window_start = now - self._reveal_window_seconds
        self._reveal_attempts = [item for item in self._reveal_attempts if item > window_start]
        if len(self._reveal_attempts) >= self._reveal_limit:
            retry_after = max(1.0, self._reveal_window_seconds - (now - self._reveal_attempts[0]))
            raise ServiceConflictError(
                "environment reveal rate limit exceeded",
                details={
                    "key": key,
                    "limit": self._reveal_limit,
                    "window_seconds": self._reveal_window_seconds,
                    "retry_after_seconds": round(retry_after, 3),
                },
            )
        self._reveal_attempts.append(now)

    def _record_reveal(self, key: str, value: str, source: EnvValueSource) -> EnvRevealResult:
        self._reveal_audit.append(
            EnvRevealAuditEntry(key=key, source=source, revealed_at=self._wall_clock())
        )
        return EnvRevealResult(key=key, value=value, source=source)

    def _set_process_value(self, key: str, value: str) -> None:
        if isinstance(self._environ, MutableMapping):
            self._environ[key] = value

    def _delete_process_value(self, key: str) -> None:
        if isinstance(self._environ, MutableMapping):
            self._environ.pop(key, None)

    def _summary(self, key: str, file_values: Mapping[str, str]) -> EnvVarSummary:
        definition = self._definitions.get(key, {})
        source: EnvValueSource
        value: str | None
        if key in file_values:
            source = "file"
            value = file_values[key]
        elif key in self._environ:
            source = "process"
            value = self._environ[key]
        else:
            source = "missing"
            value = None
        is_secret = bool(definition.get("is_secret", _looks_secret(key)))
        return EnvVarSummary(
            key=key,
            is_set=value is not None and value != "",
            source=source,
            redacted_value=_redact(value) if value else None,
            description=str(definition.get("description") or ""),
            category=str(definition.get("category") or "other"),
            is_secret=is_secret,
            advanced=bool(definition.get("advanced", False)),
            url=str(definition.get("url")) if definition.get("url") else None,
        )

    def _read_values(self) -> dict[str, str]:
        return {line.key: line.value for line in self._read_lines() if line.key is not None and line.value is not None}

    def _read_lines(self) -> list[_EnvLine]:
        if not self._env_path.exists():
            return []
        return [_parse_env_line(raw) for raw in self._env_path.read_text(encoding="utf-8").splitlines()]

    def _write_lines(self, lines: list[_EnvLine]) -> None:
        self._env_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(line.raw for line in lines).rstrip() + "\n"
        self._env_path.write_text(text, encoding="utf-8")


def _validate_key(key: str) -> str:
    normalized = (key or "").strip().upper()
    if not _KEY_PATTERN.fullmatch(normalized):
        raise ServiceValidationError("invalid environment key", details={"key": key})
    return normalized


def _parse_env_line(raw: str) -> _EnvLine:
    stripped = raw.strip()
    if not stripped or stripped.startswith("#") or "=" not in raw:
        return _EnvLine(raw=raw)
    key, value = raw.split("=", 1)
    key = key.strip()
    if not _KEY_PATTERN.fullmatch(key):
        return _EnvLine(raw=raw)
    return _EnvLine(raw=raw, key=key, value=_unquote(value.strip()))


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _format_env_line(key: str, value: str) -> str:
    if value == "" or re.search(r"\s|#|['\"]", value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'{key}="{escaped}"'
    return f"{key}={value}"


def _looks_secret(key: str) -> bool:
    return any(word in key for word in _SECRET_WORDS)


def _redact(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "..." + value[-4:]


__all__ = ["EnvironmentService"]
