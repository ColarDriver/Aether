"""Cross-session provider rate-limit guard.

The guard is intentionally file-based so separate Aether processes can share a
short-lived "do not call this provider/base_url yet" signal without requiring a
daemon or database.  All filesystem operations are best-effort: a broken or
unwritable guard directory must never break the run loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_LOCK_REASON_RATE_LIMIT = "rate_limit"


@dataclass(slots=True, frozen=True)
class RateGuardKey:
    provider: str
    namespace_hash: str
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name


@dataclass(slots=True, frozen=True)
class RateGuardLock:
    provider: str
    base_url_hash: str
    until_unix: float
    reason: str
    source_session_id: str
    created_unix: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RateGuardLock | None":
        try:
            provider = str(value["provider"])
            base_url_hash = str(value["base_url_hash"])
            until_unix = float(value["until_unix"])
            reason = str(value["reason"])
            source_session_id = str(value["source_session_id"])
            created_unix = float(value["created_unix"])
        except (KeyError, TypeError, ValueError):
            return None
        return cls(
            provider=provider,
            base_url_hash=base_url_hash,
            until_unix=until_unix,
            reason=reason,
            source_session_id=source_session_id,
            created_unix=created_unix,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RateGuardCheck:
    checked: bool
    blocked: bool
    key: RateGuardKey | None = None
    lock: RateGuardLock | None = None
    error: str | None = None

    @property
    def until_unix(self) -> float | None:
        return self.lock.until_unix if self.lock is not None else None


def default_rate_guard_dir() -> Path:
    """Return the shared runtime directory used when config omits one."""

    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        try:
            xdg_path = Path(xdg_runtime_dir)
            if xdg_path.is_dir():
                return xdg_path / "aether" / "rate_guard"
        except OSError:
            pass
    return Path(tempfile.gettempdir()) / "aether" / "rate_guard"


def provider_rate_guard_key(provider: Any, guard_dir: Path | None = None) -> RateGuardKey:
    """Build the stable lock-file key for a provider instance.

    The filename contains the provider identifier and a short hash of the
    endpoint namespace.  It deliberately never embeds the raw base_url.
    """

    root = Path(guard_dir) if guard_dir is not None else default_rate_guard_dir()
    provider_name = _safe_component(_provider_name(provider))
    namespace = _provider_namespace(provider)
    namespace_hash = hashlib.sha256(namespace.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return RateGuardKey(
        provider=provider_name,
        namespace_hash=namespace_hash,
        path=root / f"{provider_name}_{namespace_hash}.json",
    )


class RateGuard:
    """Best-effort file-backed rate-limit guard."""

    def __init__(self, guard_dir: Path | None = None) -> None:
        self.guard_dir = Path(guard_dir) if guard_dir is not None else default_rate_guard_dir()

    def check(self, provider: Any, *, now: float | None = None) -> RateGuardCheck:
        """Return whether ``provider`` is currently locked out."""

        if _stable_provider_namespace(provider) is None:
            return RateGuardCheck(checked=True, blocked=False)
        key = provider_rate_guard_key(provider, self.guard_dir)
        current = time.time() if now is None else float(now)
        try:
            with key.path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except FileNotFoundError:
            return RateGuardCheck(checked=True, blocked=False, key=key)
        except (OSError, json.JSONDecodeError) as exc:
            return RateGuardCheck(checked=True, blocked=False, key=key, error=str(exc))

        if not isinstance(raw, dict):
            return RateGuardCheck(checked=True, blocked=False, key=key, error="invalid lock")

        lock = RateGuardLock.from_mapping(raw)
        if lock is None:
            return RateGuardCheck(checked=True, blocked=False, key=key, error="invalid lock")
        if lock.provider != key.provider or lock.base_url_hash != key.namespace_hash:
            return RateGuardCheck(checked=True, blocked=False, key=key, error="lock key mismatch")
        if lock.until_unix <= current:
            self._unlink_best_effort(key.path)
            return RateGuardCheck(checked=True, blocked=False, key=key, lock=lock)
        return RateGuardCheck(checked=True, blocked=True, key=key, lock=lock)

    def block(
        self,
        provider: Any,
        *,
        until_unix: float,
        reason: str = _LOCK_REASON_RATE_LIMIT,
        source_session_id: str = "",
        now: float | None = None,
    ) -> RateGuardLock | None:
        """Write a lock for ``provider`` atomically.

        Returns the lock on success, ``None`` when the filesystem refused the
        write or the supplied expiry is already stale.
        """

        if _stable_provider_namespace(provider) is None:
            return None
        created = time.time() if now is None else float(now)
        until = float(until_unix)
        if until <= created:
            return None

        key = provider_rate_guard_key(provider, self.guard_dir)
        lock = RateGuardLock(
            provider=key.provider,
            base_url_hash=key.namespace_hash,
            until_unix=until,
            reason=reason,
            source_session_id=str(source_session_id),
            created_unix=created,
        )

        tmp_path = key.path.with_name(
            f".{key.path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            key.path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(lock.to_dict(), handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, key.path)
        except OSError:
            self._unlink_best_effort(tmp_path)
            return None
        return lock

    def clear(self, provider: Any) -> bool:
        """Remove the lock for ``provider`` if present."""

        if _stable_provider_namespace(provider) is None:
            return False
        key = provider_rate_guard_key(provider, self.guard_dir)
        try:
            key.path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            return False
        return True

    @staticmethod
    def _unlink_best_effort(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _provider_name(provider: Any) -> str:
    name = getattr(provider, "provider_name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return type(provider).__name__


def _provider_namespace(provider: Any) -> str:
    stable = _stable_provider_namespace(provider)
    if stable is not None:
        return stable
    return _provider_name(provider)


def _stable_provider_namespace(provider: Any) -> str | None:
    for attr in ("base_url", "model", "model_name"):
        value = getattr(provider, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return cleaned or "provider"


__all__ = [
    "RateGuard",
    "RateGuardCheck",
    "RateGuardKey",
    "RateGuardLock",
    "default_rate_guard_dir",
    "provider_rate_guard_key",
]
