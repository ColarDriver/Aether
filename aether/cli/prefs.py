"""Persistent CLI preferences (last-used model per provider, etc).

Stored at ``$AETHER_HOME/prefs.json`` (default ``~/.aether/prefs.json``).
Writes are atomic — temp file in the same directory, then ``os.replace``
into place — so a crash mid-write can never produce a half-baked file.

Today the only field is ``last_model_by_provider``: when the user picks
a model via ``/model`` or ``--model`` we remember it, and on the next
``aether chat`` startup (without ``--model``, ``AETHER_MODEL``, or
``--resume``) we restore the choice.  The dataclass is forward-compatible
— unknown keys in the file are preserved verbatim through load → save
so older binaries don't strip newer fields.

Why a dedicated file rather than reusing ``sessions/<id>.json``?
``SessionRecord`` already persists a per-session ``model``, but a brand
new session has no record yet — that's exactly the case the user hits
when they re-run ``aether`` after picking a model and finds the choice
forgotten.  Prefs sit at the harness level (cross-session) where
sessions sit at the conversation level.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PREFS_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Prefs:
    """On-disk CLI preferences blob."""

    last_model_by_provider: dict[str, str] = field(default_factory=dict)
    version: int = PREFS_FORMAT_VERSION
    # Anything we don't recognise gets round-tripped untouched.  Keeps
    # forward-compat painless when newer aether builds add fields and an
    # older build is run against the same file.
    unknown: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Storage location
# ---------------------------------------------------------------------------


def _aether_home() -> Path:
    """Return the directory all CLI state lives under."""
    return Path(os.getenv("AETHER_HOME", Path.home() / ".aether"))


def _prefs_file() -> Path:
    return _aether_home() / "prefs.json"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_prefs() -> Prefs:
    """Read prefs from disk; missing or malformed file → empty defaults.

    Never raises — callers treat prefs as best-effort hints, not as a
    correctness-critical store.
    """
    path = _prefs_file()
    if not path.exists():
        return Prefs()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Prefs()
    if not isinstance(raw, dict):
        return Prefs()

    last = raw.pop("last_model_by_provider", {}) or {}
    if not isinstance(last, dict):
        last = {}
    # Keep only string→string entries; defensively coerce the rest.
    sanitized = {str(k): str(v) for k, v in last.items() if v}

    version = raw.pop("version", PREFS_FORMAT_VERSION)
    if not isinstance(version, int):
        version = PREFS_FORMAT_VERSION

    return Prefs(
        last_model_by_provider=sanitized,
        version=version,
        unknown=raw,  # everything else round-trips
    )


def save_prefs(prefs: Prefs) -> None:
    """Atomically write *prefs* to disk.

    Best-effort — failures are swallowed because losing prefs should
    never break the REPL.  We log to stderr for diagnosis.
    """
    base = _aether_home()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    path = _prefs_file()
    payload: dict[str, Any] = {
        "version": prefs.version,
        "last_model_by_provider": dict(prefs.last_model_by_provider),
    }
    payload.update(prefs.unknown)

    try:
        # Atomic write: tempfile in the same directory, fsync optional,
        # rename into place.  os.replace is atomic across the same
        # filesystem on POSIX *and* on Windows.
        fd, tmp_path = tempfile.mkstemp(prefix=".prefs-", suffix=".json", dir=str(base))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_path, path)
        except Exception:
            # Best-effort cleanup of the temp file.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError:
        # Disk full / read-only filesystem / etc.  We can't help here;
        # the user still has env vars + CLI flags as override paths.
        return


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_last_model(provider: str) -> str | None:
    """Return the model the user last picked for *provider*, or ``None``."""
    if not provider:
        return None
    prefs = load_prefs()
    return prefs.last_model_by_provider.get(provider) or None


def set_last_model(provider: str, model: str) -> None:
    """Persist *model* as the last-used choice for *provider*.

    No-op when either argument is empty so a buggy caller can't wipe
    a real entry by accident.
    """
    if not provider or not model:
        return
    prefs = load_prefs()
    prefs.last_model_by_provider[provider] = model
    save_prefs(prefs)
