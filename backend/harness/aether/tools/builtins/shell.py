"""Built-in ``shell`` tool — run a shell command in a subprocess.

Mirrors the ergonomics of claude-code's ``Bash`` and hermes-agent's
``terminal`` tool: a single ``command`` string is executed via the user's
shell, stdout/stderr are captured, and a structured ``ToolResult`` is
returned with an exit code, duration, and a ``truncated`` flag.

Output handling \u2014 Sprint 3.5 / PR 3.5.1
----------------------------------------
The pre-3.5 head+tail truncation has been replaced with the unified
spill mechanism shared across builtins.  Combined ``$ command`` +
exit_code header + stdout + stderr is built first; if the total exceeds
``max_result_chars`` (40 KB), the full payload is written to disk and
``ToolResult.content`` retains only the inline preview plus a standard
``... [output truncated: ... saved to ...] ...`` notice.  The model
follows the notice with ``read_file <spilled-path>`` to retrieve the
complete output.

Why head+tail was retired: with disk spill we no longer need to
synthesise a "head + tail" view to stay under context budget \u2014 the
model can read the exact same contiguous output it would have seen
without truncation.  Single-form preview also makes the spill notice
cleaner (no mid-content marker).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool


_DEFAULT_TIMEOUT_SEC = 60
# Sprint 3.5 / PR 3.5.1 \u2014 see module docstring.  40 KB is roughly 800
# lines of typical CLI output, ~10 KB tokens; large enough that most
# shell calls render inline, small enough that runaway ``find /`` style
# commands trigger spill rather than blowing the context.
_MAX_RESULT_CHARS = 40_000


class ShellTool(ToolExecutor):
    """Execute shell commands via the platform shell."""

    def __init__(
        self,
        *,
        default_cwd: Path | None = None,
        default_timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
        max_result_chars: int = _MAX_RESULT_CHARS,
    ) -> None:
        self.default_cwd = default_cwd
        self.default_timeout_sec = default_timeout_sec
        self.max_result_chars = max_result_chars
        self._descriptor = ToolDescriptor(
            name="shell",
            description=(
                "Run a shell command and return its stdout / stderr / exit "
                "code. Use this for anything that requires a real subprocess "
                "(git, find, npm, pytest, etc.). When combined output exceeds "
                f"{self.max_result_chars // 1024} KB the full payload spills "
                "to disk and the inline preview ends with a ``[output truncated"
                " ... saved to ...]`` notice; use ``read_file`` on the saved "
                "path to retrieve the complete output."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The shell command to run, exactly as you would "
                            "type it at a prompt. Compound commands using "
                            "``&&`` / ``||`` / pipes are supported."
                        ),
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Optional working directory. Defaults to the "
                            "harness's current working directory."
                        ),
                    },
                    "timeout_sec": {
                        "type": "integer",
                        "description": (
                            "Optional per-command timeout in seconds. Defaults "
                            f"to {self.default_timeout_sec}s; cap is 600s."
                        ),
                        "minimum": 1,
                        "maximum": 600,
                    },
                },
                "required": ["command"],
            },
            required=["command"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        command = str(args.get("command") or "").strip()
        if not command:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content="error: 'command' must be a non-empty string",
                is_error=True,
                metadata={"exit_code": -1},
            )

        cwd = self._resolve_cwd(args.get("cwd"))
        timeout = self._resolve_timeout(args.get("timeout_sec"))

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(cwd) if cwd else None,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            stdout_text = stdout if isinstance(stdout, str) else stdout.decode("utf-8", "replace")
            stderr_text = stderr if isinstance(stderr, str) else stderr.decode("utf-8", "replace")
            full_output = self._format_output(
                command=command,
                cwd=cwd,
                exit_code=None,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_ms=duration_ms,
                timed_out=True,
                timeout=timeout,
            )
            content = maybe_spill_for_tool(
                full_output,
                call=call,
                context=context,
                max_chars=self.max_result_chars,
                extension="txt",
                full_lines=full_output.count("\n") + 1,
            )
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=content,
                is_error=True,
                metadata={
                    "exit_code": None,
                    "timed_out": True,
                    "timeout_sec": timeout,
                    "duration_ms": duration_ms,
                    "cwd": str(cwd) if cwd else None,
                    "command": command,
                },
            )
        except FileNotFoundError as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"error: shell unavailable: {exc}",
                is_error=True,
                metadata={"exit_code": -1, "cwd": str(cwd) if cwd else None},
            )

        duration_ms = int((time.monotonic() - started) * 1000)
        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""

        full_output = self._format_output(
            command=command,
            cwd=cwd,
            exit_code=completed.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_ms=duration_ms,
            timed_out=False,
            timeout=timeout,
        )

        # Pre-spill length is the canonical "real" payload size that
        # downstream observers care about; capture it here so the
        # metadata doesn't lie about whether the model saw the full body.
        original_chars = len(full_output)
        content = maybe_spill_for_tool(
            full_output,
            call=call,
            context=context,
            max_chars=self.max_result_chars,
            extension="txt",
            full_lines=full_output.count("\n") + 1,
        )
        spilled = len(content) != original_chars

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=completed.returncode != 0,
            metadata={
                "exit_code": completed.returncode,
                "duration_ms": duration_ms,
                "truncated": spilled,
                "cwd": str(cwd) if cwd else None,
                "command": command,
            },
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_cwd(self, value: Any) -> Path | None:
        if value:
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute() and self.default_cwd is not None:
                candidate = (self.default_cwd / candidate).resolve()
            else:
                candidate = candidate.resolve()
            return candidate
        return self.default_cwd

    def _resolve_timeout(self, value: Any) -> int:
        try:
            timeout = int(value) if value is not None else self.default_timeout_sec
        except (TypeError, ValueError):
            timeout = self.default_timeout_sec
        if timeout <= 0:
            timeout = self.default_timeout_sec
        return min(max(timeout, 1), 600)

    @staticmethod
    def _format_output(
        *,
        command: str,
        cwd: Path | None,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        duration_ms: int,
        timed_out: bool,
        timeout: int,
    ) -> str:
        # Header order is intentional: the model needs ``exit`` (or
        # timeout) and ``stderr_lines`` to judge success even if the
        # body is later spilled and only the preview survives in
        # context.  Matches the contract described in the module
        # docstring.
        lines: list[str] = []
        lines.append(f"$ {command}")
        if cwd is not None:
            lines.append(f"(cwd: {cwd})")
        if timed_out:
            lines.append(
                f"[timeout after {timeout}s, partial output below \u2014 duration {duration_ms}ms]"
            )
        else:
            stderr_line_count = stderr.count("\n") if stderr else 0
            lines.append(
                f"[exit {exit_code} in {duration_ms}ms, "
                f"stderr_lines={stderr_line_count}]"
            )
        if stdout:
            lines.append("--- stdout ---")
            lines.append(stdout.rstrip("\n"))
        if stderr:
            lines.append("--- stderr ---")
            lines.append(stderr.rstrip("\n"))
        if not stdout and not stderr and not timed_out:
            lines.append("(no output)")
        return "\n".join(lines)


__all__ = ["ShellTool"]
