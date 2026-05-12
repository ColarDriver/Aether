"""Built-in ``grep`` tool — pattern search across files.

Prefers ``ripgrep`` (``rg``) when it's on PATH because it's an order of
magnitude faster than the Python fallback and respects ``.gitignore``
out of the box.  When ripgrep isn't available we walk the directory
tree in pure Python, matching against ``re.compile(pattern)`` and
applying a tiny built-in skip list (``.git``, ``node_modules``, etc.)
so we don't drown in cache noise.
"""

from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool


_DEFAULT_MAX_MATCHES = 200
# Sprint 3.5 / PR 3.5.1 \u2014 grep results are line-formatted (``path:lineno:
# content``), so spill kicks in late and the preview is line-aligned
# (no half-line tails).  30 KB roughly covers 250-300 typical hits.
_MAX_RESULT_CHARS = 30_000
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


class GrepTool(ToolExecutor):
    """Search for a regex pattern across files."""

    def __init__(
        self,
        *,
        default_cwd: Path | None = None,
        default_max_matches: int = _DEFAULT_MAX_MATCHES,
    ) -> None:
        self.default_cwd = default_cwd
        self.default_max_matches = default_max_matches
        # Sprint 3.5 / PR 3.5.1 \u2014 fixed a pre-existing bug where the
        # initial sentinel was ``object()`` (each call returned a new
        # instance, so ``self._rg_path is object()`` was always False
        # and the resolution branch never fired).  Existing callers
        # masked it via ``try: subprocess.run except OSError`` falling
        # back to the python walker.  Replaced with explicit
        # ``_rg_resolved`` flag for clarity \u2014 None now legitimately
        # means "ripgrep not available", distinct from "not yet
        # resolved".
        self._rg_path: str | None = None
        self._rg_resolved: bool = False
        self._descriptor = ToolDescriptor(
            name="grep",
            description=(
                "Search for a regex pattern across files. Uses ripgrep "
                "(``rg``) when available, otherwise a Python regex walker. "
                "Returns at most "
                f"{self.default_max_matches} matches in the form "
                "``path:line:content``. Respects gitignore via ripgrep; the "
                "fallback walker skips ``.git`` / ``node_modules`` / cache "
                "directories by default."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python-style regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional directory or file to search. Defaults "
                            "to the current working directory."
                        ),
                    },
                    "glob": {
                        "type": "string",
                        "description": (
                            "Optional glob filter (e.g. ``*.py`` or ``**/*.ts``)."
                        ),
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "If true, perform a case-insensitive search.",
                    },
                },
                "required": ["pattern"],
            },
            required=["pattern"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        pattern = args.get("pattern")
        if not pattern or not isinstance(pattern, str):
            return _error(call, "'pattern' must be a non-empty string")

        raw_path = args.get("path")
        path = self._resolve_path(raw_path)
        if not path.exists():
            return _error(call, f"path not found: {path}", metadata={"path": str(path)})

        glob = args.get("glob")
        if glob is not None and not isinstance(glob, str):
            return _error(call, "'glob' must be a string")

        case_insensitive = bool(args.get("case_insensitive") or False)

        rg = self._ripgrep()
        if rg is not None:
            try:
                return self._run_ripgrep(call, context, rg, pattern, path, glob, case_insensitive)
            except OSError:
                pass

        return self._run_python_walker(call, context, pattern, path, glob, case_insensitive)

    # ------------------------------------------------------------------
    # ripgrep
    # ------------------------------------------------------------------

    def _ripgrep(self) -> str | None:
        if not self._rg_resolved:
            self._rg_path = shutil.which("rg")
            self._rg_resolved = True
        return self._rg_path

    def _run_ripgrep(
        self,
        call: ToolCall,
        context: TurnContext,
        rg: str,
        pattern: str,
        path: Path,
        glob: str | None,
        case_insensitive: bool,
    ) -> ToolResult:
        cmd: list[str] = [
            rg,
            "--line-number",
            "--no-heading",
            "--color=never",
            f"--max-count={self.default_max_matches}",
            f"--max-columns=400",
        ]
        if case_insensitive:
            cmd.append("-i")
        if glob:
            cmd.extend(["-g", glob])
        cmd.append(pattern)
        cmd.append(str(path))

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        # rg exit codes: 0 = matches, 1 = no matches, 2 = error
        if completed.returncode == 2:
            stderr = (completed.stderr or "").strip()
            return _error(call, f"ripgrep error: {stderr or 'unknown'}")

        stdout = completed.stdout or ""
        matches = stdout.splitlines()
        truncated = len(matches) >= self.default_max_matches
        if truncated:
            matches = matches[: self.default_max_matches]

        return _format_result(call, context, pattern, path, matches, truncated, engine="ripgrep")

    # ------------------------------------------------------------------
    # python fallback
    # ------------------------------------------------------------------

    def _run_python_walker(
        self,
        call: ToolCall,
        context: TurnContext,
        pattern: str,
        path: Path,
        glob: str | None,
        case_insensitive: bool,
    ) -> ToolResult:
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return _error(call, f"invalid regex: {exc}")

        matches: list[str] = []
        truncated = False

        if path.is_file():
            files = [path]
        else:
            files = self._iter_files(path, glob)

        for file in files:
            if len(matches) >= self.default_max_matches:
                truncated = True
                break
            try:
                with file.open("r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if regex.search(line):
                            stripped = line.rstrip("\n")
                            if len(stripped) > 400:
                                stripped = stripped[:397] + "..."
                            matches.append(f"{file}:{lineno}:{stripped}")
                            if len(matches) >= self.default_max_matches:
                                truncated = True
                                break
            except OSError:
                continue

        return _format_result(call, context, pattern, path, matches, truncated, engine="python")

    def _iter_files(self, path: Path, glob: str | None):
        for entry in path.rglob("*"):
            if entry.is_dir():
                continue
            if any(part in _DEFAULT_SKIP for part in entry.parts):
                continue
            if glob and not fnmatch.fnmatch(str(entry.name), glob) and not fnmatch.fnmatch(str(entry), glob):
                continue
            yield entry

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, raw: Any) -> Path:
        if not raw:
            return self.default_cwd or Path.cwd()
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute() and self.default_cwd is not None:
            return (self.default_cwd / candidate).resolve()
        return candidate.resolve()


def _format_result(
    call: ToolCall,
    context: TurnContext,
    pattern: str,
    path: Path,
    matches: list[str],
    truncated: bool,
    *,
    engine: str,
) -> ToolResult:
    header = [
        f"# pattern: {pattern}",
        f"# path:    {path}",
        f"# engine:  {engine}",
        f"# matches: {len(matches)}" + (" (truncated)" if truncated else ""),
    ]
    if matches:
        body = "\n".join(matches)
    else:
        body = "(no matches)"
    full_output = "\n".join(header) + "\n" + body

    # Line-aligned preview: maybe_spill_for_tool slices at exactly
    # ``max_chars``, which can split the last grep line in half.  We
    # nudge the preview boundary back to the nearest newline so models
    # never see ``"path:42:def foo("`` followed by a truncation notice
    # mid-token.  This is grep-specific tidying \u2014 other tools whose
    # output is naturally line-uniform (glob, list_dir) don't need it.
    spilled = False
    if (
        len(full_output) > _MAX_RESULT_CHARS
        and getattr(
            context.metadata.get("_engine_config") if context.metadata else None,
            "tool_result_spill_enabled",
            True,
        )
    ):
        # Land the cut on (and including) the newline that closes the
        # previous match line, so the preview always ends with ``\n``
        # rather than mid-line.  ``rfind`` returns the newline's
        # position; ``+ 1`` includes it in the preview slice.
        cutoff = full_output.rfind("\n", 0, _MAX_RESULT_CHARS)
        line_aligned_max = (cutoff + 1) if cutoff > 0 else _MAX_RESULT_CHARS
        original_chars = len(full_output)
        content = maybe_spill_for_tool(
            full_output,
            call=call,
            context=context,
            max_chars=line_aligned_max,
            extension="txt",
            full_lines=full_output.count("\n") + 1,
        )
        spilled = len(content) != original_chars
    else:
        content = full_output

    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=content,
        is_error=False,
        metadata={
            "pattern": pattern,
            "path": str(path),
            "engine": engine,
            "match_count": len(matches),
            "truncated": truncated or spilled,
            "spilled": spilled,
        },
    )


def _error(call: ToolCall, message: str, *, metadata: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"error: {message}",
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["GrepTool"]
