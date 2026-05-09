"""Built-in ``read_file`` tool — read a text file with line numbers.

Output format mirrors Cursor / claude-code: ``     12| line content`` so
the model can reference specific line ranges in subsequent edits.

Sprint 3.5 / PR 3.5.1 changes
-----------------------------
* The pre-3.5 hard ``256 KB`` cap (which raised ``is_error=True``) is
  retired.  Files larger than ``MAX_RESULT_CHARS`` (60 KB rendered) now
  flow through the shared spill path: full numbered output goes to
  disk under ``~/.aether/tool_results``, ``ToolResult.content`` keeps
  the inline preview plus the standard ``[output truncated ... saved
  to ...]`` notice.  The model retrieves the rest by reading the
  spilled file back \u2014 which is what the notice tells it to do.
* **Recursion guard**: if the requested ``path`` lives under the spill
  directory itself, we deliberately skip the spill check.  Without this
  guard, the model's "follow the notice" call would re-spill its own
  spill file on every read, exhausting disk and breaking the contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tool_result_storage import DEFAULT_SPILL_ROOT
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool


_DEFAULT_LIMIT = 2000
# Sprint 3.5 / PR 3.5.1 \u2014 see module docstring.  60 KB \u2248 1500 numbered
# lines, intentionally generous because users explicitly asked to read
# the file (vs. shell whose output was incidental).
_MAX_RESULT_CHARS = 60_000


class ReadFileTool(ToolExecutor):
    """Read a file from disk and return numbered lines."""

    def __init__(
        self,
        *,
        default_cwd: Path | None = None,
        default_limit: int = _DEFAULT_LIMIT,
        max_result_chars: int = _MAX_RESULT_CHARS,
    ) -> None:
        self.default_cwd = default_cwd
        self.default_limit = default_limit
        self.max_result_chars = max_result_chars
        self._descriptor = ToolDescriptor(
            name="read_file",
            description=(
                "Read the contents of a text file. Returns lines numbered "
                "starting at 1 in the format ``     12| <content>``. Use "
                "this instead of ``shell`` (cat / head / tail) for any "
                "non-interactive read of a single file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (absolute or relative to cwd).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "1-indexed line to start reading at. Defaults to 1. "
                            "Negative values count from the end of the file "
                            "(``-1`` = last line)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Maximum number of lines to return. Defaults to "
                            f"{self.default_limit}."
                        ),
                        "minimum": 1,
                    },
                },
                "required": ["path"],
            },
            required=["path"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        raw_path = args.get("path")
        if not raw_path:
            return _error(call, "'path' must be a non-empty string")

        path = self._resolve_path(raw_path)

        if not path.exists():
            return _error(call, f"file not found: {path}", metadata={"path": str(path)})
        if path.is_dir():
            return _error(
                call,
                f"path is a directory (use list_dir): {path}",
                metadata={"path": str(path)},
            )

        try:
            stat = path.stat()
        except OSError as exc:
            return _error(call, f"could not stat {path}: {exc}")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _error(call, f"could not read {path}: {exc}")

        all_lines = text.splitlines()
        total_lines = len(all_lines)

        offset = self._resolve_offset(args.get("offset"), total_lines)
        limit = self._resolve_limit(args.get("limit"))

        end = min(total_lines, offset - 1 + limit)
        slice_ = all_lines[offset - 1: end]

        line_truncated = end < total_lines

        rendered_lines: list[str] = []
        for i, line in enumerate(slice_, start=offset):
            rendered_lines.append(f"{i:>6}| {line}")

        header_lines: list[str] = [f"# {path}"]
        if total_lines == 0:
            header_lines.append("(empty file)")
        else:
            header_lines.append(
                f"# lines {offset}-{end} of {total_lines}"
                + (" (truncated)" if line_truncated else "")
            )

        body = "\n".join(rendered_lines) if rendered_lines else "(no lines in range)"
        full_output = "\n".join(header_lines) + "\n" + body

        # Sprint 3.5 / PR 3.5.1 \u2014 recursion guard for the spill path.
        # When the model follows a previous truncation notice and reads
        # back a spilled file, we MUST NOT re-spill it: the file lives
        # under the configured spill root, so detecting that and
        # short-circuiting the maybe_spill_for_tool call breaks the
        # otherwise-infinite "preview \u2192 read \u2192 preview \u2192 read" loop.
        if self._is_under_spill_root(path, context=context):
            content = full_output
            spilled = False
        else:
            original_chars = len(full_output)
            content = maybe_spill_for_tool(
                full_output,
                call=call,
                context=context,
                max_chars=self.max_result_chars,
                extension="txt",
                full_lines=total_lines,
                full_bytes=stat.st_size,
            )
            spilled = len(content) != original_chars

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=False,
            metadata={
                "path": str(path),
                "offset": offset,
                "limit": limit,
                "lines_returned": len(slice_),
                "total_lines": total_lines,
                "truncated": line_truncated or spilled,
                "size_bytes": stat.st_size,
                "spilled": spilled,
            },
        )

    @staticmethod
    def _is_under_spill_root(path: Path, *, context: TurnContext) -> bool:
        """Return True if ``path`` lives under the spill root for this run.

        We resolve symlinks on both sides so a tool that read back a
        spilled file via a symlink-into-tmpdir still hits the guard.
        Configured override (``EngineConfig.tool_result_spill_dir``) is
        consulted first; the canonical default acts as a fallback so
        even ad-hoc TurnContexts (no engine config) still benefit.
        """
        cfg = context.metadata.get("_engine_config") if context.metadata else None
        configured = getattr(cfg, "tool_result_spill_dir", None)
        candidates: list[Path] = []
        if configured is not None:
            candidates.append(Path(configured))
        candidates.append(DEFAULT_SPILL_ROOT)
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            return False
        for root in candidates:
            try:
                root_resolved = root.resolve()
            except (OSError, RuntimeError):
                continue
            try:
                resolved.relative_to(root_resolved)
                return True
            except ValueError:
                continue
        return False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, raw: Any) -> Path:
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute() and self.default_cwd is not None:
            return (self.default_cwd / candidate).resolve()
        return candidate.resolve()

    def _resolve_offset(self, value: Any, total_lines: int) -> int:
        try:
            offset = int(value) if value is not None else 1
        except (TypeError, ValueError):
            offset = 1
        if offset < 0:
            offset = max(1, total_lines + 1 + offset)
        if offset < 1:
            offset = 1
        if offset > max(1, total_lines):
            offset = max(1, total_lines)
        return offset

    def _resolve_limit(self, value: Any) -> int:
        try:
            limit = int(value) if value is not None else self.default_limit
        except (TypeError, ValueError):
            limit = self.default_limit
        return max(1, limit)


def _error(call: ToolCall, message: str, *, metadata: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"error: {message}",
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["ReadFileTool"]
