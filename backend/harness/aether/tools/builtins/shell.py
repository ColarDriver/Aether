"""Built-in ``shell`` tool — run a shell command in a subprocess.

Mirrors the ergonomics of claude-code's ``Bash`` and hermes-agent's
``terminal`` tool: a single ``command`` string is executed via the user's
shell, stdout/stderr are captured (with size caps so a misbehaving
command can't blow the context window), and a structured ``ToolResult``
is returned with an exit code, duration, and a ``truncated`` flag.

Output cap rationale: 16 KB per stream keeps the round-trip token bill
bounded while still showing 100+ lines of typical CLI output.  Past
that we keep the head **and** the tail — the head identifies what
command ran / what its banner looked like, the tail is where errors and
final output usually live.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor


_DEFAULT_TIMEOUT_SEC = 60
_MAX_BYTES_PER_STREAM = 16 * 1024


class ShellTool(ToolExecutor):
    """Execute shell commands via the platform shell."""

    def __init__(
        self,
        *,
        default_cwd: Path | None = None,
        default_timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
        max_bytes_per_stream: int = _MAX_BYTES_PER_STREAM,
    ) -> None:
        self.default_cwd = default_cwd
        self.default_timeout_sec = default_timeout_sec
        self.max_bytes_per_stream = max_bytes_per_stream
        self._descriptor = ToolDescriptor(
            name="shell",
            description=(
                "Run a shell command and return its stdout / stderr / exit "
                "code. Use this for anything that requires a real subprocess "
                "(git, find, npm, pytest, etc.). Output is truncated to "
                f"{self.max_bytes_per_stream // 1024} KB per stream."
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
            content = self._format_output(
                command=command,
                cwd=cwd,
                exit_code=None,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_ms=duration_ms,
                timed_out=True,
                timeout=timeout,
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
        stdout, stdout_truncated = self._cap(completed.stdout or "")
        stderr, stderr_truncated = self._cap(completed.stderr or "")
        truncated = stdout_truncated or stderr_truncated

        content = self._format_output(
            command=command,
            cwd=cwd,
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=False,
            timeout=timeout,
        )

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=completed.returncode != 0,
            metadata={
                "exit_code": completed.returncode,
                "duration_ms": duration_ms,
                "truncated": truncated,
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

    def _cap(self, text: str) -> tuple[str, bool]:
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= self.max_bytes_per_stream:
            return text, False
        head_size = self.max_bytes_per_stream * 2 // 3
        tail_size = self.max_bytes_per_stream - head_size
        head = encoded[:head_size].decode("utf-8", errors="replace")
        tail = encoded[-tail_size:].decode("utf-8", errors="replace")
        omitted = len(encoded) - head_size - tail_size
        marker = f"\n... [{omitted} bytes truncated] ...\n"
        return head + marker + tail, True

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
        lines: list[str] = []
        lines.append(f"$ {command}")
        if cwd is not None:
            lines.append(f"(cwd: {cwd})")
        if timed_out:
            lines.append(
                f"[timeout after {timeout}s, partial output below — duration {duration_ms}ms]"
            )
        else:
            lines.append(f"[exit {exit_code} in {duration_ms}ms]")
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
