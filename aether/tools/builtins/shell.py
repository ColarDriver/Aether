"""Built-in ``shell`` tool — run a shell command in a subprocess.

CWD is tracked per-session and persists across calls.  Parity with
open-claude-code's pattern (``utils/Shell.ts`` + ``shell/bashProvider.ts``):
each command is wrapped with ``; pwd -P >| <capture>`` so that after
``cd /workspace/foo``, the next shell call automatically starts in
``/workspace/foo``.  The captured path is stored in
``session_state.set_cwd(session_id, ...)``; the next ``_resolve_cwd``
reads it back.  Without this, every ``cd`` would be silently undone
by the fresh subprocess and the model would have to repeat
``cd /workspace/foo &&`` on every command.

Env vars and activated venvs are NOT round-tripped (would require a
persistent shell process or env-dump capture).  Models should invoke
venv interpreters directly (``./.venv/bin/python …``) rather than
expecting ``source .venv/bin/activate`` to persist.  The
``<shell_tool_contract>`` system-prompt section spells this out.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.session.session_state import get_cwd, set_cwd
from aether.runtime.tools.tool_permissions import ToolPermissionPreview
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool

_DEFAULT_TIMEOUT_SEC = 60
_MAX_RESULT_CHARS = 40_000
_INTERRUPT_GRACE_SEC = 2.0

_logger = logging.getLogger(__name__)


def _shell_quote(value: str) -> str:
    """POSIX single-quote-safe quoting for a single token."""
    return "'" + value.replace("'", "'\\''") + "'"


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
        cwd = self._resolve_cwd(
            args.get("cwd"),
            session_id=context.session_id,
        )
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

        cwd = self._resolve_cwd(
            args.get("cwd"),
            session_id=context.session_id,
        )
        # CWD recovery: if the tracked CWD was deleted by a previous
        # command (or never existed), fall back to default_cwd then to
        # the process CWD.  Matches OCC's behaviour in
        # ``Shell.ts:220-237``.
        cwd = self._recover_cwd_if_missing(
            cwd, session_id=context.session_id
        )
        timeout = self._resolve_timeout(args.get("timeout_sec"))
        started = time.monotonic()
        interrupted = False
        timed_out = False

        # Wrap with ``; pwd -P > <capture>`` so the post-command CWD
        # round-trips back to ``session_state.set_cwd``.  Uses an
        # unconditional ``;`` separator (not ``&&``) so a failing
        # user command still emits the final CWD — we want to track
        # ``cd /workspace/foo && false`` as "now in /workspace/foo"
        # even though the chain exited non-zero.
        cwd_capture_path = self._make_cwd_capture_path()
        wrapped_command = self._wrap_with_cwd_capture(
            command, cwd_capture_path
        )

        process = subprocess.Popen(
            wrapped_command,
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

            def _escalate() -> None:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            threading.Timer(_INTERRUPT_GRACE_SEC, _escalate).start()

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

        # Persist the post-command CWD before any further work so that
        # subsequent tool calls within the same session see it even if
        # the rest of this method raises.  Cleans up the capture file
        # unconditionally.
        new_cwd = self._consume_cwd_capture(
            cwd_capture_path,
            previous=cwd,
            session_id=context.session_id,
        )

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
                "new_cwd": str(new_cwd) if new_cwd else None,
                "command": command,
                "stderr_lines": stderr_lines,
            },
        )

    def _resolve_cwd(
        self,
        value: Any,
        *,
        session_id: str | None = None,
    ) -> Path | None:
        """Decide which CWD to launch the subprocess in.

        Priority (matches OCC's effective order):

        1. **Explicit ``cwd`` argument on this call.**  The model can
           always pin a one-off CWD (e.g. ``cwd: "/tmp"``).  Relative
           paths resolve against the session CWD if any, otherwise
           ``default_cwd``.

        2. **Session CWD** — what the previous ``cd`` in this session
           left us in (round-tripped via ``pwd -P``).  This is the
           lever that makes ``cd /workspace/foo`` persist.

        3. **``default_cwd``** — what the tool was registered with
           (typically from ``EngineRequest.cwd`` / ``EngineConfig``).

        4. ``None`` — let ``subprocess.Popen`` inherit ``os.getcwd()``.
        """
        if value:
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute():
                # Relative ``cwd`` arg: resolve against the session
                # CWD first (so ``cwd: "src"`` means
                # ``<session_cwd>/src``), then ``default_cwd``.
                base = self._session_cwd(session_id) or self.default_cwd
                if base is not None:
                    candidate = (base / candidate).resolve()
                else:
                    candidate = candidate.resolve()
            else:
                candidate = candidate.resolve()
            return candidate
        session_cwd = self._session_cwd(session_id)
        if session_cwd is not None:
            return session_cwd
        return self.default_cwd

    def _session_cwd(self, session_id: str | None) -> Path | None:
        if not session_id:
            return None
        tracked = get_cwd(session_id)
        if not tracked:
            return None
        return Path(tracked)

    def _recover_cwd_if_missing(
        self,
        cwd: Path | None,
        *,
        session_id: str | None,
    ) -> Path | None:
        """Mirror OCC's ``Shell.ts:220-237`` recovery.

        If a previous command deleted the tracked CWD (or it never
        existed), fall back to ``default_cwd`` then to ``os.getcwd``
        so the subprocess can still launch.  Also clears the bad
        session CWD so subsequent calls don't keep tripping over it.
        """
        if cwd is None:
            return None
        try:
            if cwd.is_dir():
                return cwd
        except OSError:
            pass
        # The tracked / requested CWD is gone — log loudly and recover.
        _logger.warning(
            "shell.cwd_recovery: tracked CWD %s no longer exists; "
            "falling back to default_cwd / process CWD",
            cwd,
        )
        if session_id:
            # Drop the stale session CWD so the next call doesn't pick
            # it up again.  Re-import locally to avoid a circular when
            # session_state is loaded before this module.
            from aether.runtime.session.session_state import clear_cwd

            clear_cwd(session_id)
        if self.default_cwd is not None:
            try:
                if self.default_cwd.is_dir():
                    return self.default_cwd
            except OSError:
                pass
        try:
            return Path(os.getcwd())
        except (OSError, FileNotFoundError):
            return None

    def _make_cwd_capture_path(self) -> Path:
        """Return a unique tmp path the shell can ``pwd -P >|`` into.

        Uses ``mkstemp`` to allocate + ensure the parent dir is
        writable, then immediately closes the fd — the shell will
        re-open the path for write.  The file gets unlinked in
        ``_consume_cwd_capture`` (success path) or the post-process
        ``finally`` (failure path).

        Instance method (not static) so tests can override on a single
        :class:`ShellTool` without touching the class descriptor —
        rebinding a static method on the class breaks ``self.foo()``
        callsites on subsequent unrelated tests.
        """
        del self  # parameter is unused; kept so subclasses/tests can override
        fd, path = tempfile.mkstemp(prefix="aether-shell-cwd-", suffix=".txt")
        os.close(fd)
        return Path(path)

    @staticmethod
    def _wrap_with_cwd_capture(command: str, capture_path: Path) -> str:
        """Return a shell-safe wrapped form of ``command``.

        Pattern mirrors OCC ``bashProvider.ts:184``: run the user
        command via ``eval '<quoted>'`` so it executes in the *same*
        shell (a subshell ``( … )`` would isolate ``cd``, defeating
        the whole CWD round-trip), then append ``pwd -P > <capture>``.
        We use ``;`` separators (not ``&&``) so a failing user
        command still emits ``pwd`` — ``cd foo && false`` landed us
        in ``foo`` even though the chain exited non-zero, and we want
        to track that.

        ``eval`` also solves the heredoc-terminator problem: by
        quoting the entire command into a single POSIX-escaped
        string, multi-line bodies (``python - <<'PY'\n...\nPY``) and
        compound shells survive without the suffix being swallowed by
        a heredoc that never terminates.
        """
        quoted_path = _shell_quote(str(capture_path))
        quoted_command = _shell_quote(command)
        return (
            f"eval {quoted_command}; __aether_rc=$?; "
            f"pwd -P > {quoted_path}; exit $__aether_rc"
        )

    @staticmethod
    def _consume_cwd_capture(
        capture_path: Path,
        *,
        previous: Path | None,
        session_id: str | None,
    ) -> Path | None:
        """Read the ``pwd -P`` capture, persist to session, clean up.

        Returns the captured path (or ``None`` if reading failed).
        Mutates :mod:`aether.runtime.session.session_state` only when
        the model actually moved (i.e. ``cd`` ran and the post-command
        CWD differs from the pre-command effective CWD).  This
        matters: a plain ``printf hello`` must NOT pin the session
        CWD to the process CWD, because that would silently override
        the tool's own ``default_cwd`` on subsequent ``read_file`` /
        ``grep`` / ``glob`` calls.

        Comparison is done with :py:meth:`Path.samefile` so symlink
        equivalence works (``/var/foo`` vs ``/private/var/foo`` on
        macOS, etc.) and falls back to a string compare if either
        path isn't statable.
        """
        captured: Path | None = None
        try:
            raw = capture_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw = ""
        try:
            capture_path.unlink(missing_ok=True)
        except OSError:
            pass
        stripped = raw.strip()
        if not stripped:
            return None
        try:
            captured = Path(stripped).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        if not captured.is_dir():
            # Shouldn't happen — ``pwd -P`` returns the kernel's
            # current dir which by definition existed at the time the
            # final builtin ran.  Treat as a no-op so we don't poison
            # the session state.
            return captured
        # Effective pre-command CWD: what Popen actually used.  If
        # the caller didn't pass a cwd, Popen inherited ``os.getcwd``.
        # We compare against that so a no-op command in the inherited
        # CWD doesn't trip persistence.
        if previous is not None:
            effective_previous: Path | None = previous
        else:
            try:
                effective_previous = Path(os.getcwd())
            except (OSError, FileNotFoundError):
                effective_previous = None
        if session_id:
            same_as_previous = False
            if effective_previous is not None:
                try:
                    same_as_previous = effective_previous.samefile(captured)
                except (OSError, FileNotFoundError):
                    same_as_previous = (
                        str(effective_previous.resolve(strict=False))
                        == str(captured)
                    )
            if not same_as_previous:
                set_cwd(session_id, captured)
        return captured

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
