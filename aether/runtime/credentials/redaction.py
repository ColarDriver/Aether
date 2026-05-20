"""Small secret redaction helpers for runtime status and metadata."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


_SECRET_KEY_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
    "credential",
    "credentials",
}

_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
    "credential",
)

_BEARER_RE = re.compile(r"\b(Bearer\s+)([A-Za-z0-9._~+/=-]{10,})", re.IGNORECASE)
_AUTH_HEADER_RE = re.compile(
    r"\b(Authorization\s*:\s*(?:Bearer\s+)?)([^\s,;]+)",
    re.IGNORECASE,
)
_PREFIX_SECRET_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{10,}|sk_[A-Za-z0-9_]{10,}|"
    r"gho_[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,}|"
    r"tvly-[A-Za-z0-9]{10,}|BSAA[A-Za-z0-9]{10,}|xai-[A-Za-z0-9]{10,})"
    r"(?![A-Za-z0-9_-])"
)


def redact_secret(value: str) -> str:
    """Return a display-safe version of a single secret value."""

    if value == "":
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def redact_text(text: str) -> str:
    """Redact common token shapes inside free text."""

    redacted = _BEARER_RE.sub(lambda m: m.group(1) + redact_secret(m.group(2)), text)
    redacted = _AUTH_HEADER_RE.sub(lambda m: m.group(1) + redact_secret(m.group(2)), redacted)
    return _PREFIX_SECRET_RE.sub(lambda m: redact_secret(m.group(0)), redacted)


def redact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively redact secret-looking values in a mapping."""

    return {str(key): _redact_value(str(key), value) for key, value in mapping.items()}


def contains_secret_like_text(text: str) -> bool:
    """Best-effort check for raw secret-like text."""

    if _BEARER_RE.search(text) or _AUTH_HEADER_RE.search(text):
        return True
    return bool(_PREFIX_SECRET_RE.search(text))


def _redact_value(key: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(key, item) for item in value)
    if isinstance(value, str):
        if _is_secret_key(key):
            return redact_secret(value)
        return redact_text(value)
    return value


def _is_secret_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized.endswith("_env") or normalized.endswith("_env_name") or normalized.endswith("_env_names"):
        return False
    if normalized in _SECRET_KEY_NAMES:
        return True
    return any(part in normalized for part in _SECRET_KEY_PARTS)


__all__ = [
    "contains_secret_like_text",
    "redact_mapping",
    "redact_secret",
    "redact_text",
]
