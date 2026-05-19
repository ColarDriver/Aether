"""Coverage for the per-session CWD round-trip in ``ShellTool``.

Parity with the OCC ``pwd -P`` capture pattern (``Shell.ts:385`` +
``shell/bashProvider.ts:184``).  These tests pin:

* ``cd`` persists across shell calls within the same session.
* Two sessions don't share CWD state.
* CWD recovery kicks in when the tracked directory was deleted.
* Explicit ``cwd`` argument on a single call doesn't permanently
  change the session CWD if the command doesn't ``cd`` itself.
* Path-aware tools (``read_file``) honour the session CWD set by a
  prior ``cd``.
* The ``pwd -P`` capture file is always cleaned up.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.runtime.session.session_state import (
    clear_cwd,
    get_cwd,
    set_cwd,
)
from aether.tools.builtins.read_file import ReadFileTool
from aether.tools.builtins.shell import ShellTool


def _ctx(session_id: str) -> TurnContext:
    return TurnContext(session_id=session_id, iteration=1)


def _call(command: str, *, cwd: str | None = None) -> ToolCall:
    args: dict = {"command": command}
    if cwd is not None:
        args["cwd"] = cwd
    return ToolCall(id=f"call-{abs(hash(command))%10000}", name="shell", arguments=args)


class CwdPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = "shell-cwd-persistence"
        clear_cwd(self.session_id)
        self.addCleanup(clear_cwd, self.session_id)

    def test_cd_persists_to_next_call(self) -> None:
        # Two consecutive shell calls: the first ``cd``s into /tmp;
        # the second runs ``pwd`` with no explicit cwd arg and must
        # report /tmp.  This is the OCC pattern's headline behaviour.
        tool = ShellTool()
        first = tool.execute(_call("cd /tmp"), _ctx(self.session_id))
        self.assertFalse(first.is_error, first.content)

        tracked = get_cwd(self.session_id)
        self.assertIsNotNone(tracked)
        assert tracked is not None
        self.assertEqual(Path(tracked).resolve(), Path("/tmp").resolve())

        second = tool.execute(_call("pwd"), _ctx(self.session_id))
        self.assertFalse(second.is_error, second.content)
        # The pwd output should equal the new CWD (modulo /private
        # macOS realpath quirks, hence the resolve()/samefile dance).
        self.assertIn(str(Path("/tmp").resolve()), second.content)
        self.assertEqual(
            Path(second.metadata["cwd"]).resolve(),
            Path("/tmp").resolve(),
        )

    def test_no_op_command_does_not_change_session_cwd(self) -> None:
        # ``printf hello`` doesn't ``cd``.  The session must NOT
        # acquire a pinned CWD just because shell ran — otherwise
        # subsequent path-aware tools (read_file, grep, ...) would
        # override their own ``default_cwd`` with the process CWD.
        tool = ShellTool()
        result = tool.execute(_call("printf hello"), _ctx(self.session_id))
        self.assertFalse(result.is_error, result.content)
        self.assertIsNone(get_cwd(self.session_id))

    def test_explicit_cwd_arg_is_one_shot_without_cd(self) -> None:
        # Passing an explicit ``cwd`` arg is one-shot: the command
        # launches there, but if it doesn't ``cd`` itself the session
        # CWD stays untouched.  This matches the no-op-doesn't-pin
        # rule — otherwise every ``shell(cwd=...)`` call would
        # silently rebind the session, defeating ``default_cwd``.
        # If the model wants a permanent switch it must ``cd``.
        tool = ShellTool()
        with tempfile.TemporaryDirectory() as workdir:
            workdir_resolved = Path(workdir).resolve()
            result = tool.execute(
                _call("pwd", cwd=str(workdir_resolved)),
                _ctx(self.session_id),
            )
            self.assertFalse(result.is_error, result.content)
            # Output reports the explicit cwd ...
            self.assertEqual(
                Path(result.metadata["cwd"]).resolve(), workdir_resolved
            )
            # ... but session CWD is NOT pinned to it.
            self.assertIsNone(get_cwd(self.session_id))

    def test_explicit_cwd_then_cd_inside_does_pin(self) -> None:
        # When the command itself ``cd``s after the explicit cwd
        # launch, the post-command CWD differs from pre-command — so
        # the round-trip DOES persist.
        tool = ShellTool()
        with tempfile.TemporaryDirectory() as workdir:
            workdir_resolved = Path(workdir).resolve()
            (workdir_resolved / "sub").mkdir()
            result = tool.execute(
                _call("cd sub", cwd=str(workdir_resolved)),
                _ctx(self.session_id),
            )
            self.assertFalse(result.is_error, result.content)
            tracked = get_cwd(self.session_id)
            self.assertIsNotNone(tracked)
            assert tracked is not None
            self.assertEqual(
                Path(tracked).resolve(),
                (workdir_resolved / "sub").resolve(),
            )


class CwdRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = "shell-cwd-recovery"
        clear_cwd(self.session_id)
        self.addCleanup(clear_cwd, self.session_id)

    def test_deleted_session_cwd_recovers_to_process_cwd(self) -> None:
        # Pin a session CWD to a tempdir, delete the tempdir, then
        # issue a shell call.  The tool must self-heal (fall back to
        # default_cwd / process CWD) and clear the dead session CWD
        # so subsequent calls don't keep tripping over it.
        with tempfile.TemporaryDirectory() as workdir:
            set_cwd(self.session_id, workdir)
        # Tempdir is now gone.
        self.assertIsNotNone(get_cwd(self.session_id))

        tool = ShellTool()
        result = tool.execute(_call("pwd"), _ctx(self.session_id))
        self.assertFalse(result.is_error, result.content)
        # Tool must have cleared the stale session CWD AND
        # re-captured the recovered one (process CWD).
        tracked = get_cwd(self.session_id)
        # Either cleared, or pointed at a real directory.  The
        # important thing is the call succeeded.
        if tracked is not None:
            self.assertTrue(Path(tracked).is_dir())


class CwdIsolationTests(unittest.TestCase):
    def test_two_sessions_do_not_share_cwd(self) -> None:
        sess_a = "shell-cwd-iso-a"
        sess_b = "shell-cwd-iso-b"
        self.addCleanup(clear_cwd, sess_a)
        self.addCleanup(clear_cwd, sess_b)
        tool = ShellTool()

        tool.execute(_call("cd /tmp"), _ctx(sess_a))
        self.assertEqual(
            Path(get_cwd(sess_a) or "").resolve(), Path("/tmp").resolve()
        )
        # Session B never touched CWD.
        self.assertIsNone(get_cwd(sess_b))


class FileToolHonoursSessionCwdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = "shell-cwd-readfile"
        clear_cwd(self.session_id)
        self.addCleanup(clear_cwd, self.session_id)

    def test_read_file_relative_resolves_against_session_cwd(self) -> None:
        # Set up a project tree, ``cd`` into it via shell, then read
        # a relative path with ``read_file``.  The session CWD set
        # by the cd must win over read_file's own ``default_cwd``
        # (which is unset here).
        with tempfile.TemporaryDirectory() as project:
            project_resolved = Path(project).resolve()
            (project_resolved / "README.md").write_text("hello\n")
            shell = ShellTool()
            shell.execute(
                _call(f"cd {project_resolved}"), _ctx(self.session_id)
            )
            self.assertEqual(
                Path(get_cwd(self.session_id) or "").resolve(),
                project_resolved,
            )

            reader = ReadFileTool()
            call = ToolCall(
                id="rf-1",
                name="read_file",
                arguments={"path": "README.md"},
            )
            result = reader.execute(call, _ctx(self.session_id))
            self.assertFalse(result.is_error, result.content)
            self.assertIn("hello", result.content)
            self.assertEqual(
                Path(result.metadata["path"]).resolve(),
                (project_resolved / "README.md").resolve(),
            )


class CwdCaptureCleanupTests(unittest.TestCase):
    def test_pwd_capture_file_is_unlinked_after_run(self) -> None:
        # Spy on the per-instance capture-path factory by subclassing
        # — avoids mutating the class itself, which would leak into
        # every subsequent ShellTool in the same test process.
        from aether.tools.builtins.shell import ShellTool as _ShellTool

        captured_paths: list[Path] = []

        class _SpyTool(_ShellTool):
            def _make_cwd_capture_path(self) -> Path:
                path = super()._make_cwd_capture_path()
                captured_paths.append(path)
                return path

        tool = _SpyTool()
        session_id = "shell-cwd-cleanup"
        self.addCleanup(clear_cwd, session_id)
        result = tool.execute(_call("true"), _ctx(session_id))
        self.assertFalse(result.is_error, result.content)

        self.assertEqual(len(captured_paths), 1)
        self.assertFalse(captured_paths[0].exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
