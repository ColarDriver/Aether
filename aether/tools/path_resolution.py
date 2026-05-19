"""Shared path-resolution helpers for the bundled tools.

All path-aware tools (``read_file``, ``list_dir``, ``grep``, ``glob``,
``write_file``, ``file_edit``) interpret relative paths against a
common base.  This module owns the *priority order* used to pick that
base so the rule is enforced consistently and matches the
session-CWD round-trip implemented in :mod:`aether.tools.builtins.shell`:

1. **Session CWD** — whatever the most recent ``cd`` in this session
   left us in (round-tripped via ``pwd -P`` by :class:`ShellTool`).
   This is the lever that makes ``cd /workspace/foo`` then
   ``read_file('README.md')`` mean ``/workspace/foo/README.md``.

2. **Tool-bound ``default_cwd``** — the value the tool was registered
   with (typically ``EngineRequest.cwd`` / ``EngineConfig.default_cwd``).

3. ``Path.cwd()`` — the process CWD, as a last resort.

Absolute paths bypass the whole chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def session_cwd(session_id: str | None) -> Path | None:
    """Return the tracked session CWD for *session_id*, or ``None``."""
    if not session_id:
        return None
    # Local import to avoid a cycle: ``session_state`` is loaded
    # eagerly at engine init time, but ``tools.path_resolution`` is
    # only loaded by individual tool modules.
    from aether.runtime.session.session_state import get_cwd

    tracked = get_cwd(session_id)
    if not tracked:
        return None
    return Path(tracked)


def resolve_tool_path(
    raw: Any,
    *,
    default_cwd: Path | None,
    session_id: str | None,
) -> Path:
    """Resolve a tool-supplied path against the priority chain above.

    The returned path is resolved (symlinks collapsed, ``..`` removed)
    but the file does not need to exist — ``Path.resolve(strict=False)``
    is intentional so write-class tools can target a new file in a
    real directory.
    """
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    base = session_cwd(session_id) or default_cwd
    if base is not None:
        return (base / candidate).resolve()
    return candidate.resolve()


def resolve_tool_dir(
    raw: Any,
    *,
    default_cwd: Path | None,
    session_id: str | None,
) -> Path:
    """Resolve a directory-shaped tool argument.

    Behaviour matches :func:`resolve_tool_path` for non-empty
    arguments.  When *raw* is empty/blank, returns the priority-chain
    base (session CWD → default_cwd → ``Path.cwd()``) so callers that
    treat "no path supplied" as "current dir" pick up the right CWD.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return session_cwd(session_id) or default_cwd or Path.cwd()
    return resolve_tool_path(raw, default_cwd=default_cwd, session_id=session_id)


__all__ = ["resolve_tool_path", "resolve_tool_dir", "session_cwd"]
