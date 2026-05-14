"""Disk-spill helpers for large tool results.

Each built-in (and future) tool decides its own threshold and IF/WHEN
to spill — this module just provides the shared file-naming, path
resolution, write logic, and notice-text builder so every tool
produces the same machine-readable hint pattern.

Spilled files live under:

    <spill_dir>/<session_id>/<call_id>.<ext>

Default ``spill_dir`` = ``~/.aether/tool_results``; override via
``EngineConfig.tool_result_spill_dir`` (path injected onto
``TurnContext.metadata['_engine_config']`` by the engine).

Why this layout?

* **Session-scoped subdirs** keep `cleanup_session_spills` cheap and
  let humans identify which conversation produced the file.
* **`call_id` filename** guarantees uniqueness even when the same
  tool spills repeatedly in one turn — the ToolCall ids are already
  uuids-or-equivalent.
* **No metadata file** — the receipt only needs to live long enough
  for the model to read the spilled file back via ``read_file``;
  there's no cross-session bookkeeping to persist.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "DEFAULT_SPILL_ROOT",
    "SpillReceipt",
    "build_truncation_notice",
    "cleanup_session_spills",
    "resolve_spill_dir",
    "spill_to_disk",
]


DEFAULT_SPILL_ROOT: Path = Path.home() / ".aether" / "tool_results"


@dataclass(slots=True, frozen=True)
class SpillReceipt:
    """Outcome of a successful ``spill_to_disk`` call.

    Attributes
    ----------
    path:
        Absolute on-disk location of the spilled content.
    full_chars:
        Length of the original content in characters (NOT bytes — keeping
        the same unit as ``MAX_RESULT_CHARS`` thresholds avoids a
        unit-mismatch bug when the model reads back the value).
    preview_chars:
        Number of characters the caller plans to keep inline in the
        ``ToolResult.content`` preview. Useful for notice formatters
        that want to express "shown {preview} of {full}" precisely.
    """

    path: Path
    full_chars: int
    preview_chars: int

    @property
    def relative_hint(self) -> str:
        """Render ``path`` HOME-relative when possible.

        Models tend to copy the hint into a ``read_file`` arg verbatim;
        ``~/`` prefixes survive the trip in any shell that expands
        tildes (which our ReadFileTool does via ``Path.expanduser``).
        """
        try:
            home = Path.home()
            rel = self.path.relative_to(home)
            return f"~/{rel}"
        except ValueError:
            return str(self.path)


def resolve_spill_dir(
    *, session_id: str, config_dir: Optional[Path] = None
) -> Path:
    """Return the per-session directory for spills, creating it if needed.

    The directory is created with ``parents=True, exist_ok=True`` so the
    first spill in a session never has to special-case directory
    creation.  Failures bubble as ``OSError`` (disk full / permission
    denied) — callers should catch and degrade gracefully.
    """
    base = config_dir if config_dir is not None else DEFAULT_SPILL_ROOT
    target = base / session_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def spill_to_disk(
    content: str,
    *,
    session_id: str,
    call_id: str,
    extension: str = "txt",
    config_dir: Optional[Path] = None,
    preview_chars: int = 0,
) -> SpillReceipt:
    """Write ``content`` to a session-scoped file and return the receipt.

    Parameters mirror what every per-tool spill path needs:

    * ``session_id`` keeps tools that ran in different conversations from
      colliding (and lets cleanup nuke a session in one shot).
    * ``call_id`` is the ToolCall id — guaranteed unique per turn by the
      engine, so no extra random suffixing is required.
    * ``extension`` is informational (``.txt`` / ``.md`` / ``.json``).
      We never re-parse based on the extension; it's purely so a human
      browsing ``~/.aether/tool_results`` can guess the format.
    * ``preview_chars`` is just plumbed through to the receipt for
      observability — this function itself doesn't truncate.

    Raises
    ------
    OSError
        On disk-full, read-only filesystem, permission denied, etc.
        Callers should catch and fall back to plain inline truncation
        rather than crashing the tool dispatch.
    """
    spill_dir = resolve_spill_dir(session_id=session_id, config_dir=config_dir)
    target = spill_dir / f"{call_id}.{extension}"
    target.write_text(content, encoding="utf-8")
    return SpillReceipt(
        path=target,
        full_chars=len(content),
        preview_chars=preview_chars,
    )


def build_truncation_notice(
    receipt: SpillReceipt,
    *,
    full_lines: Optional[int] = None,
    full_bytes: Optional[int] = None,
) -> str:
    """Build the standard '... [truncated, see file] ...' notice.

    Format (intentionally machine-readable so models learn the pattern):

        \\n\\n... [output truncated: {N} chars{ / M lines}{ / K bytes}
        saved to {hint} \u2014 use read_file to retrieve the full content] ...

    Notes
    -----
    * ``\u2014`` (em-dash) is intentional — model tokenisers handle it
      fine and it visually separates the metric block from the
      instruction.  Tests assert on the exact substring
      ``"use read_file to retrieve"`` so renaming is a contract change.
    * Newlines are doubled at the start so the notice reliably renders
      as a separate paragraph after whatever preview content precedes
      it (markdown / Anthropic / OpenAI all collapse single newlines).
    """
    parts: list[str] = [f"{receipt.full_chars} chars"]
    if full_lines is not None:
        parts.append(f"{full_lines} lines")
    if full_bytes is not None:
        parts.append(f"{full_bytes} bytes")
    metrics = " / ".join(parts)
    return (
        f"\n\n... [output truncated: {metrics} saved to "
        f"{receipt.relative_hint} \u2014 use read_file to retrieve the full content] ..."
    )


def cleanup_session_spills(
    *,
    session_id: str,
    config_dir: Optional[Path] = None,
    max_age_seconds: int = 7 * 24 * 3600,
) -> int:
    """Remove session-scoped spill files older than ``max_age_seconds``.

    Returns the count of files actually removed.  Per-file failures
    (concurrent unlink, permission flake) are silently swallowed so the
    function degrades to "best-effort" semantics — callers should never
    branch on a specific return value, only treat ``> 0`` as "did some
    cleanup" for logging.

    Default age (7 days) leaves enough headroom for the model to
    realise mid-week it wants to re-read an earlier spill while still
    bounding disk usage on a long-running daemon.
    """
    base = config_dir if config_dir is not None else DEFAULT_SPILL_ROOT
    spill_dir = base / session_id
    if not spill_dir.exists():
        return 0
    now = time.time()
    removed = 0
    for child in spill_dir.iterdir():
        try:
            age = now - child.stat().st_mtime
            if age > max_age_seconds:
                child.unlink()
                removed += 1
        except OSError:
            # Concurrent unlink / permission flake / vanished file —
            # cleanup is best-effort, never fatal.
            continue
    return removed
