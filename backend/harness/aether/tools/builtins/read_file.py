"""Built-in ``read_file`` tool — read a text file with line numbers.

Output format mirrors Cursor / claude-code: ``     12| line content`` so
the model can reference specific line ranges in subsequent edits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor


_DEFAULT_LIMIT = 2000
_MAX_BYTES = 256 * 1024


class ReadFileTool(ToolExecutor):
    """Read a file from disk and return numbered lines."""

    def __init__(
        self,
        *,
        default_cwd: Path | None = None,
        default_limit: int = _DEFAULT_LIMIT,
        max_bytes: int = _MAX_BYTES,
    ) -> None:
        self.default_cwd = default_cwd
        self.default_limit = default_limit
        self.max_bytes = max_bytes
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

        if stat.st_size > self.max_bytes:
            return _error(
                call,
                (
                    f"file too large: {stat.st_size} bytes > {self.max_bytes} byte cap. "
                    "Use offset/limit, or run a `shell` command (head/tail/grep) for "
                    "targeted reads."
                ),
                metadata={"path": str(path), "size_bytes": stat.st_size},
            )

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

        truncated = end < total_lines

        rendered_lines: list[str] = []
        for i, line in enumerate(slice_, start=offset):
            rendered_lines.append(f"{i:>6}| {line}")

        header_lines: list[str] = [f"# {path}"]
        if total_lines == 0:
            header_lines.append("(empty file)")
        else:
            header_lines.append(
                f"# lines {offset}-{end} of {total_lines}"
                + (" (truncated)" if truncated else "")
            )

        body = "\n".join(rendered_lines) if rendered_lines else "(no lines in range)"
        content = "\n".join(header_lines) + "\n" + body

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
                "truncated": truncated,
                "size_bytes": stat.st_size,
            },
        )

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
