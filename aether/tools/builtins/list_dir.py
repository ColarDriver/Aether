"""Built-in ``list_dir`` tool — list the contents of a directory.

The output mirrors a flat ``ls -lah`` style: one line per entry with a
``d``/``-`` type marker, size, and name, plus an indication of whether
the listing was truncated. Hidden entries (``.git``, ``node_modules``,
``__pycache__``, …) are skipped by default — agents almost never want
them and they balloon the listing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool


_DEFAULT_MAX_ENTRIES = 256
# ``list_dir`` is naturally bounded by ``max_entries``; spill is a
# last-resort backstop in case a caller bumps that limit and a colossal
# directory still overflows the context. 20 KB is roughly 256 typical
# entries.
_MAX_RESULT_CHARS = 20_000
_DEFAULT_SKIP = (
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".tox",
    ".idea",
    ".vscode",
)


class ListDirTool(ToolExecutor):
    """List a directory's immediate children."""

    def __init__(
        self,
        *,
        default_cwd: Path | None = None,
        default_max_entries: int = _DEFAULT_MAX_ENTRIES,
        default_skip: tuple[str, ...] = _DEFAULT_SKIP,
    ) -> None:
        self.default_cwd = default_cwd
        self.default_max_entries = default_max_entries
        self.default_skip = default_skip
        self._descriptor = ToolDescriptor(
            name="list_dir",
            description=(
                "List the immediate contents of a directory. Returns one "
                "line per entry: ``d  4096  name/`` for directories, "
                "``-  size  name`` for files. By default, common cache / "
                "vendor directories (``.git``, ``node_modules``, …) are "
                "skipped — pass ``include_hidden: true`` to see everything."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (absolute or relative to cwd).",
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": (
                            f"Maximum number of entries to return. Defaults to "
                            f"{self.default_max_entries}."
                        ),
                        "minimum": 1,
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": (
                            "Include dotfiles and the default skip-list entries "
                            "(.git, node_modules, …)."
                        ),
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
            return _error(call, f"directory not found: {path}", metadata={"path": str(path)})
        if not path.is_dir():
            return _error(
                call,
                f"path is not a directory: {path}",
                metadata={"path": str(path)},
            )

        max_entries = self._resolve_max_entries(args.get("max_entries"))
        include_hidden = bool(args.get("include_hidden") or False)

        try:
            children = sorted(
                path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError as exc:
            return _error(call, f"could not list {path}: {exc}", metadata={"path": str(path)})

        skipped = 0
        rows: list[str] = []
        for child in children:
            name = child.name
            if not include_hidden:
                if name.startswith("."):
                    skipped += 1
                    continue
                if name in self.default_skip:
                    skipped += 1
                    continue
            try:
                if child.is_dir():
                    rows.append(f"d  {0:>10}  {name}/")
                else:
                    size = child.stat().st_size
                    rows.append(f"-  {size:>10}  {name}")
            except OSError:
                rows.append(f"?  {0:>10}  {name}")

            if len(rows) >= max_entries:
                break

        truncated = len(rows) >= max_entries and any(True for _ in path.iterdir())
        header = [f"# {path}"]
        header.append(f"# {len(rows)} entries shown" + (f", {skipped} skipped" if skipped else ""))
        if truncated:
            header.append("# (output truncated — pass a larger max_entries to see more)")

        body = "\n".join(rows) if rows else "(empty)"
        full_output = "\n".join(header) + "\n" + body

        original_chars = len(full_output)
        content = maybe_spill_for_tool(
            full_output,
            call=call,
            context=context,
            max_chars=_MAX_RESULT_CHARS,
            extension="txt",
            full_lines=full_output.count("\n") + 1,
        )
        spilled = len(content) != original_chars

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=False,
            metadata={
                "path": str(path),
                "entries_returned": len(rows),
                "entries_skipped": skipped,
                "truncated": truncated or spilled,
                "spilled": spilled,
            },
        )

    def _resolve_path(self, raw: Any) -> Path:
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute() and self.default_cwd is not None:
            return (self.default_cwd / candidate).resolve()
        return candidate.resolve()

    def _resolve_max_entries(self, value: Any) -> int:
        try:
            value = int(value) if value is not None else self.default_max_entries
        except (TypeError, ValueError):
            value = self.default_max_entries
        return max(1, min(value, 4096))


def _error(call: ToolCall, message: str, *, metadata: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"error: {message}",
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["ListDirTool"]
