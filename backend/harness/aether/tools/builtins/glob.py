"""Built-in ``glob`` tool — find files by name pattern.

A thin wrapper around :func:`pathlib.Path.glob` that:

* automatically promotes a non-recursive pattern (``*.py``) to a
  recursive one (``**/*.py``) when no ``/`` separator is present, so
  agents don't have to remember ``**`` every single time;
* skips common cache / vendor directories (``.git``, ``node_modules``,
  ``__pycache__``, …) the same way the other builtins do;
* sorts results by modification time descending so the freshest hits
  surface first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool


_DEFAULT_MAX_RESULTS = 200
# Sprint 3.5 / PR 3.5.1 \u2014 path lists are dense (one path per line);
# 20 KB is roughly 600-800 typical paths.
_MAX_RESULT_CHARS = 20_000
_DEFAULT_SKIP = {
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
}


class GlobTool(ToolExecutor):
    """Find files by glob pattern."""

    def __init__(
        self,
        *,
        default_cwd: Path | None = None,
        default_max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> None:
        self.default_cwd = default_cwd
        self.default_max_results = default_max_results
        self._descriptor = ToolDescriptor(
            name="glob",
            description=(
                "Find files matching a glob pattern. Patterns without a ``/`` "
                "(e.g. ``*.py``) are automatically promoted to recursive "
                "patterns (``**/*.py``). Skips ``.git`` / ``node_modules`` / "
                "cache directories. Returns at most "
                f"{self.default_max_results} paths sorted by modification time."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern. Examples: ``*.py``, ``src/**/*.tsx``, "
                            "``backend/**/test_*.py``."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional directory to search. Defaults to the "
                            "current working directory."
                        ),
                    },
                },
                "required": ["pattern"],
            },
            required=["pattern"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        pattern = args.get("pattern")
        if not pattern or not isinstance(pattern, str):
            return _error(call, "'pattern' must be a non-empty string")

        path = self._resolve_path(args.get("path"))
        if not path.exists():
            return _error(call, f"path not found: {path}", metadata={"path": str(path)})
        if not path.is_dir():
            return _error(call, f"path is not a directory: {path}", metadata={"path": str(path)})

        effective_pattern = pattern
        if "/" not in effective_pattern and not effective_pattern.startswith("**"):
            effective_pattern = f"**/{effective_pattern}"

        try:
            raw_matches = list(path.glob(effective_pattern))
        except (OSError, ValueError) as exc:
            return _error(call, f"glob failed: {exc}")

        filtered: list[Path] = []
        for entry in raw_matches:
            if entry.is_dir():
                continue
            if any(part in _DEFAULT_SKIP for part in entry.parts):
                continue
            filtered.append(entry)

        try:
            filtered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            filtered.sort()

        truncated = len(filtered) > self.default_max_results
        if truncated:
            filtered = filtered[: self.default_max_results]

        header = [
            f"# pattern: {pattern}",
            f"# path:    {path}",
            f"# matches: {len(filtered)}" + (" (truncated)" if truncated else ""),
        ]
        body = "\n".join(str(p) for p in filtered) if filtered else "(no matches)"
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
                "pattern": pattern,
                "path": str(path),
                "match_count": len(filtered),
                "truncated": truncated or spilled,
                "spilled": spilled,
            },
        )

    def _resolve_path(self, raw: Any) -> Path:
        if not raw:
            return self.default_cwd or Path.cwd()
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute() and self.default_cwd is not None:
            return (self.default_cwd / candidate).resolve()
        return candidate.resolve()


def _error(call: ToolCall, message: str, *, metadata: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"error: {message}",
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["GlobTool"]
