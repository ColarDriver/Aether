"""Boot and lifecycle tests for the gateway process entrypoint (PR 1)."""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import IO
from unittest import mock

from aether.gateway import entry
from aether.gateway.transport import StdioTransport


# ---------------------------------------------------------------------
# In-process unit tests for helpers that are awkward to verify via
# subprocess.  These do not start a real gateway.
# ---------------------------------------------------------------------


class PanicHookWritesCrashLog(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env = mock.patch.dict(os.environ, {"AETHER_HOME": self._tmp.name})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _crash_log(self) -> Path:
        return Path(self._tmp.name) / "logs" / "gateway_crash.log"

    def test_panic_hook_appends_traceback_to_crash_log(self) -> None:
        try:
            raise RuntimeError("boom-one")
        except RuntimeError:
            exc_type, exc_value, exc_tb = sys.exc_info()
        assert exc_type is not None and exc_value is not None

        # Suppress the default-hook chain so the test doesn't print to
        # the real stderr or fail noisily.
        with mock.patch.object(sys, "__excepthook__", lambda *a, **kw: None):
            with mock.patch.object(sys, "stderr"):  # silence one-line summary
                entry._panic_hook(exc_type, exc_value, exc_tb)

        text = self._crash_log().read_text(encoding="utf-8")
        self.assertIn("RuntimeError", text)
        self.assertIn("boom-one", text)
        self.assertIn("unhandled exception", text)

    def test_panic_hook_emits_one_line_to_stderr(self) -> None:
        captured: list[str] = []

        class _Cap:
            def write(self, s: str) -> int:
                captured.append(s)
                return len(s)

            def flush(self) -> None:
                pass

        try:
            raise ValueError("boom-two")
        except ValueError:
            exc_type, exc_value, exc_tb = sys.exc_info()
        assert exc_type is not None and exc_value is not None

        with mock.patch.object(sys, "__excepthook__", lambda *a, **kw: None):
            with mock.patch.object(sys, "stderr", _Cap()):
                entry._panic_hook(exc_type, exc_value, exc_tb)

        joined = "".join(captured)
        self.assertIn("[gateway-crash]", joined)
        self.assertIn("ValueError", joined)
        self.assertIn("boom-two", joined)

    def test_thread_panic_hook_writes_thread_tag(self) -> None:
        try:
            raise KeyError("missing-key")
        except KeyError:
            exc_type, exc_value, exc_tb = sys.exc_info()
        assert exc_type is not None and exc_value is not None

        # threading.excepthook is called with a SimpleNamespace-like
        # object exposing exc_type / exc_value / exc_traceback / thread.
        # Use SimpleNamespace so pyright sees the attributes as set.
        args = SimpleNamespace(
            exc_type=exc_type,
            exc_value=exc_value,
            exc_traceback=exc_tb,
            thread=threading.Thread(name="worker-7"),
        )

        with mock.patch.object(sys, "stderr"):
            entry._thread_panic_hook(args)

        text = self._crash_log().read_text(encoding="utf-8")
        self.assertIn("[thread=worker-7]", text)
        self.assertIn("KeyError", text)

    def test_crash_log_dir_created_on_demand(self) -> None:
        logs_dir = Path(self._tmp.name) / "logs"
        self.assertFalse(logs_dir.exists())

        entry._append_crash_log("hello")
        self.assertTrue(logs_dir.exists())
        self.assertTrue((logs_dir / "gateway_crash.log").exists())


class ShutdownGraceEnv(unittest.TestCase):
    def test_default_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AETHER_GATEWAY_SHUTDOWN_GRACE_S", None)
            self.assertEqual(
                entry._shutdown_grace_seconds(),
                entry._DEFAULT_SHUTDOWN_GRACE_S,
            )

    def test_override_via_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"AETHER_GATEWAY_SHUTDOWN_GRACE_S": "2.5"},
        ):
            self.assertEqual(entry._shutdown_grace_seconds(), 2.5)

    def test_invalid_value_falls_back_to_default(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"AETHER_GATEWAY_SHUTDOWN_GRACE_S": "not-a-float"},
        ):
            self.assertEqual(
                entry._shutdown_grace_seconds(),
                entry._DEFAULT_SHUTDOWN_GRACE_S,
            )

    def test_non_positive_value_falls_back_to_default(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"AETHER_GATEWAY_SHUTDOWN_GRACE_S": "0"},
        ):
            self.assertEqual(
                entry._shutdown_grace_seconds(),
                entry._DEFAULT_SHUTDOWN_GRACE_S,
            )


class RequestLoopExitsOnEof(unittest.TestCase):
    """Verify the PR 2 request loop exits cleanly when stdin closes."""

    def test_request_loop_returns_on_eof(self) -> None:
        sink = StdioTransport(lambda: io.StringIO())
        with mock.patch.object(sys, "stdin", io.StringIO("")):
            # Reset shared shutdown event in case a prior test set it.
            entry._shutdown_event.clear()
            entry._request_loop(sink)  # should return promptly

    def test_request_loop_discards_blank_lines_then_exits_on_eof(self) -> None:
        out = io.StringIO()
        sink = StdioTransport(lambda: out)
        with mock.patch.object(sys, "stdin", io.StringIO("\n\n\n")):
            entry._shutdown_event.clear()
            entry._request_loop(sink)
        # Blank lines never produce any output.
        self.assertEqual(out.getvalue(), "")


# ---------------------------------------------------------------------
# Subprocess tests for end-to-end boot/signal/shutdown behaviour.
# ---------------------------------------------------------------------


def _repo_root() -> Path:
    # aether/tests/gateway/test_entry_boot.py → repo root is 4 levels up.
    return Path(__file__).resolve().parents[3]


@dataclass
class _GatewayProc:
    """A spawned gateway subprocess with non-Optional pipe handles.

    ``subprocess.Popen.stdin`` / ``stdout`` / ``stderr`` are typed as
    ``IO[bytes] | None`` even when ``PIPE`` is requested.  We assert
    they exist once at spawn time and re-expose them as plain fields
    so each test can use them without re-narrowing.
    """

    proc: subprocess.Popen[bytes]
    stdin: IO[bytes]
    stdout: IO[bytes]
    stderr: IO[bytes]

    def poll(self) -> int | None:
        return self.proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.proc.wait(timeout=timeout)

    def send_signal(self, sig: int) -> None:
        self.proc.send_signal(sig)

    def kill(self) -> None:
        self.proc.kill()


def _spawn_gateway(env_overrides: dict[str, str] | None = None) -> _GatewayProc:
    """Spawn a gateway subprocess with stdin/stdout/stderr piped.

    Returns a small wrapper exposing each pipe as a non-Optional
    attribute so test bodies don't have to assert their presence.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if env_overrides:
        env.update(env_overrides)
    # Ensure the repo's src dir is importable.
    pythonpath = env.get("PYTHONPATH", "")
    repo = str(_repo_root())
    env["PYTHONPATH"] = f"{repo}{os.pathsep}{pythonpath}" if pythonpath else repo

    # Invoke the entry function directly via -c so tests don't require
    # `pip install -e .` to have produced the `aether-gateway` console
    # script.  Production callers use the installed script.
    bootstrap = "from aether.gateway.entry import main; raise SystemExit(main())"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", bootstrap],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    return _GatewayProc(
        proc=proc,
        stdin=proc.stdin,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


@unittest.skipIf(os.name == "nt", "POSIX signal semantics; Windows test deferred")
class GatewayBootSubprocess(unittest.TestCase):
    """Spawn the real `python -m aether.gateway` and verify lifecycle."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._aether_home = self._tmp.name

    def test_boots_and_idles_then_exits_on_eof(self) -> None:
        gw = _spawn_gateway({"AETHER_HOME": self._aether_home})
        try:
            # Process should NOT exit immediately.
            time.sleep(0.3)
            self.assertIsNone(
                gw.poll(),
                msg="gateway exited before we closed stdin",
            )

            # PR 2: dispatcher emits a `gateway.ready` notification at
            # boot and nothing else without inbound requests.  Close
            # stdin → EOF → request loop returns → process exits.
            gw.stdin.close()
            rc = gw.wait(timeout=5.0)
            self.assertEqual(rc, 0)

            stdout = gw.stdout.read().decode("utf-8", errors="replace")
            lines = [line for line in stdout.split("\n") if line.strip()]
            self.assertEqual(
                len(lines),
                1,
                msg=f"expected exactly one stdout frame (gateway.ready); got {stdout!r}",
            )

            frame = json.loads(lines[0])
            self.assertEqual(frame["jsonrpc"], "2.0")
            self.assertEqual(frame["method"], "gateway.ready")
            self.assertNotIn(
                "id",
                frame,
                msg="gateway.ready must be a notification (no id)",
            )
            self.assertIn("version", frame["params"])
            self.assertIn("capabilities", frame["params"])
            self.assertIn("ping", frame["params"]["capabilities"])
        finally:
            if gw.poll() is None:
                gw.kill()
                gw.wait(timeout=2.0)

    def test_parse_error_response_has_explicit_null_id(self) -> None:
        """JSON-RPC 2.0: parse-error responses MUST carry an id member set to null."""
        gw = _spawn_gateway({"AETHER_HOME": self._aether_home})
        try:
            time.sleep(0.2)
            gw.stdin.write(b"not json\n")
            gw.stdin.flush()
            gw.stdin.close()
            rc = gw.wait(timeout=5.0)
            self.assertEqual(rc, 0)

            stdout = gw.stdout.read().decode("utf-8", errors="replace")
            lines = [line for line in stdout.split("\n") if line.strip()]
            # gateway.ready + parse error.
            error_frame = json.loads(lines[1])
            self.assertEqual(error_frame["error"]["code"], -32700)
            # id MUST be present (and null), not absent.
            self.assertIn("id", error_frame)
            self.assertIsNone(error_frame["id"])
        finally:
            if gw.poll() is None:
                gw.kill()
                gw.wait(timeout=2.0)

    def test_ping_round_trip(self) -> None:
        """End-to-end smoke: send a ping request, expect a pong response."""
        gw = _spawn_gateway({"AETHER_HOME": self._aether_home})
        try:
            time.sleep(0.3)
            gw.stdin.write(
                b'{"jsonrpc":"2.0","id":"ping-1","method":"gateway.ping"}\n'
            )
            gw.stdin.flush()
            gw.stdin.close()
            rc = gw.wait(timeout=5.0)
            self.assertEqual(rc, 0)

            stdout = gw.stdout.read().decode("utf-8", errors="replace")
            lines = [line for line in stdout.split("\n") if line.strip()]
            # gateway.ready + ping response.
            self.assertEqual(len(lines), 2, msg=stdout)

            ready = json.loads(lines[0])
            response = json.loads(lines[1])
            self.assertEqual(ready["method"], "gateway.ready")
            self.assertEqual(response["id"], "ping-1")
            self.assertTrue(response["result"]["pong"])
        finally:
            if gw.poll() is None:
                gw.kill()
                gw.wait(timeout=2.0)

    def test_sigterm_triggers_clean_exit_within_grace(self) -> None:
        gw = _spawn_gateway({"AETHER_HOME": self._aether_home})
        try:
            time.sleep(0.3)
            self.assertIsNone(gw.poll())

            gw.send_signal(signal.SIGTERM)
            rc = gw.wait(timeout=5.0)
            self.assertEqual(rc, 0)
        finally:
            if gw.poll() is None:
                gw.kill()
                gw.wait(timeout=2.0)

    def test_sigint_triggers_clean_exit(self) -> None:
        gw = _spawn_gateway({"AETHER_HOME": self._aether_home})
        try:
            time.sleep(0.3)
            self.assertIsNone(gw.poll())

            gw.send_signal(signal.SIGINT)
            rc = gw.wait(timeout=5.0)
            self.assertEqual(rc, 0)
        finally:
            if gw.poll() is None:
                gw.kill()
                gw.wait(timeout=2.0)

    def test_sigpipe_does_not_kill_process(self) -> None:
        """Closing the gateway's stdout under it must not crash the process.

        The default SIGPIPE action would terminate the process the
        instant any background write hits a closed pipe.  This test
        proves our signal handler keeps the gateway alive: we close
        stdout, give the gateway time to potentially write something
        (it won't in PR 1, but the handler is what matters), then
        verify it's still running and shuts down cleanly on SIGTERM.
        """
        gw = _spawn_gateway({"AETHER_HOME": self._aether_home})
        try:
            time.sleep(0.2)
            gw.stdout.close()  # closes our end; gateway writes would EPIPE
            time.sleep(0.3)
            self.assertIsNone(
                gw.poll(),
                msg="gateway died after stdout was closed under it",
            )

            gw.send_signal(signal.SIGTERM)
            rc = gw.wait(timeout=5.0)
            self.assertEqual(rc, 0)
        finally:
            if gw.poll() is None:
                gw.kill()
                gw.wait(timeout=2.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
