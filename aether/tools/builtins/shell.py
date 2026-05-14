"""Built-in ``shell`` tool — run a shell command in a subprocess."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tools.tool_permissions import ToolPermissionPreview
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool

_DEFAULT_TIMEOUT_SEC = 60
_MAX_RESULT_CHARS = 40_000
_INTERRUPT_GRACE_SEC = 2.0


class ShellTool(ToolExecutor):
    """Execute shell commands via the platform shell."""

    interrupt_behavior = "cancel"

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

    def build_permission_preview(
        self,
        call: ToolCall,
        context: TurnContext,
    ) -> ToolPermissionPreview | ToolResult:
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
        return ToolPermissionPreview(
            title="Run command",
            subtitle=str(cwd) if cwd else None,
            command=command,
            body=f"timeout: {timeout}s",
            metadata={
                "cwd": str(cwd) if cwd else None,
                "timeout_sec": timeout,
            },
        )

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
        interrupted = False
        timed_out = False

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd) if cwd else None,
            start_new_session=True,
        )

        def _cancel(_reason: str | None) -> None:
            nonlocal interrupted
            if process.poll() is not None:
                return
            interrupted = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + _INTERRUPT_GRACE_SEC
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        listener = None
        if context.interrupt_signal is not None:
            listener = _cancel
            context.interrupt_signal.add_listener(listener)

        try:
            stdout_text, stderr_text = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _cancel("timeout")
            stdout_text, stderr_text = process.communicate()
        finally:
            if context.interrupt_signal is not None and listener is not None:
                context.interrupt_signal.remove_listener(listener)

        duration_ms = int((time.monotonic() - started) * 1000)
        exit_code = process.returncode
        stderr_value = stderr_text or ""
        stderr_lines = stderr_value.count("\n") + (1 if stderr_value else 0)
        full_output = self._format_output(
            command=command,
            cwd=cwd,
            exit_code=exit_code,
            stdout=stdout_text or "",
            stderr=stderr_value,
            duration_ms=duration_ms,
            timed_out=timed_out,
            timeout=timeout,
            interrupted=interrupted,
        )
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
            is_error=bool(interrupted or timed_out or (exit_code or 0) != 0),
            metadata={
                "exit_code": exit_code,
                "duration_ms": duration_ms,
                "truncated": spilled,
                "timed_out": timed_out,
                "interrupted": interrupted,
                "cwd": str(cwd) if cwd else None,
                "command": command,
                "stderr_lines": stderr_lines,
            },
        )

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
        interrupted: bool,
    ) -> str:
        stderr_lines = stderr.count("\n") + (1 if stderr else 0)
        if interrupted:
            header = f"[interrupted after {duration_ms}ms · stderr_lines={stderr_lines}]"
        elif timed_out:
            header = f"[timeout after {timeout}s · {duration_ms}ms · stderr_lines={stderr_lines}]"
        else:
            header = f"[exit {exit_code} · {duration_ms}ms · stderr_lines={stderr_lines}]"
        lines = [f"$ {command}", header]
        if cwd is not None:
            lines.append(f"cwd: {cwd}")
        if stdout:
            lines.extend(["", stdout])
        if stderr:
            lines.extend(["", "[stderr]", stderr])
        return "\n".join(lines).rstrip() + "\n"
