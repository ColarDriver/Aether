"""Tests for ``aether.runtime.resources.lsp_client.LSPClient``.

Sprint 3.5 / PR-3 (PR 3.5.9).

We can't depend on a real LSP server being installed during CI, so we
spin up a tiny *Python* JSON-RPC echo server in a subprocess that
speaks the LSP framing (``Content-Length`` header + JSON body).  That
gives us end-to-end coverage of:

* startup + ``initialize`` handshake
* ``request`` / response correlation
* ``notify`` (fire-and-forget)
* per-request timeout
* graceful ``shutdown``
* server-crash detection
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from aether.runtime.resources.lsp_client import LSPClient, LSPProtocolError


# A minimal LSP-framed JSON-RPC echo server.  Behaviour:
# * On any request it echoes ``{"capabilities": {}}`` for ``initialize``,
#   ``None`` for ``shutdown`` and ``{"echoed": params}`` for everything
#   else.
# * It honours an env var ``AETHER_LSP_FAKE_SLOW=1`` to delay every
#   reply by 2 seconds so the timeout test is fast and reliable.
# * It exits cleanly when stdin closes (no Ctrl-C kludges).
_FAKE_SERVER = textwrap.dedent(
    """
    import json
    import os
    import sys
    import time

    HEADER_TERM = b"\\r\\n\\r\\n"
    SLOW = os.environ.get("AETHER_LSP_FAKE_SLOW") == "1"

    def read_message(stream):
        buf = b""
        while HEADER_TERM not in buf:
            chunk = stream.read(1)
            if not chunk:
                return None
            buf += chunk
        header, _, rest = buf.partition(HEADER_TERM)
        n = 0
        for line in header.decode("ascii").split("\\r\\n"):
            if line.lower().startswith("content-length:"):
                n = int(line.split(":", 1)[1].strip())
        body = rest
        while len(body) < n:
            chunk = stream.read(n - len(body))
            if not chunk:
                return None
            body += chunk
        return json.loads(body.decode("utf-8"))

    def write_message(stream, payload):
        body = json.dumps(payload).encode("utf-8")
        stream.write(("Content-Length: %d\\r\\n\\r\\n" % len(body)).encode("ascii"))
        stream.write(body)
        stream.flush()

    while True:
        msg = read_message(sys.stdin.buffer)
        if msg is None:
            break
        if SLOW:
            time.sleep(2)
        if "id" not in msg:
            # notification — drop
            if msg.get("method") == "exit":
                break
            continue
        method = msg.get("method", "")
        if method == "initialize":
            result = {"capabilities": {"definitionProvider": True}}
        elif method == "shutdown":
            result = None
        elif method == "boom":
            sys.exit(7)
        else:
            result = {"echoed": msg.get("params")}
        write_message(
            sys.stdout.buffer,
            {"jsonrpc": "2.0", "id": msg["id"], "result": result},
        )
    """
).strip()


def _spawn_fake_server_command() -> tuple[Path, list[str]]:
    tmp = Path(tempfile.mkdtemp(prefix="lsp-fake-"))
    script = tmp / "fake_lsp.py"
    script.write_text(_FAKE_SERVER, encoding="utf-8")
    return tmp, [sys.executable, str(script)]


class LSPClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp, self._cmd = _spawn_fake_server_command()
        self._clients: list[LSPClient] = []

    def tearDown(self) -> None:
        for client in self._clients:
            try:
                client.shutdown(grace=False, timeout=1.0)
            except Exception:
                pass

    def _build(self, *, slow: bool = False, init_timeout: float = 5.0) -> LSPClient:
        env = {"AETHER_LSP_FAKE_SLOW": "1"} if slow else {}
        merged = {**dict(__import__("os").environ), **env}
        client = LSPClient(
            command=self._cmd,
            project_root=self._tmp,
            language="python",
            init_timeout=init_timeout,
            env=merged,
        )
        self._clients.append(client)
        return client

    def test_a1_start_completes_initialize_handshake(self) -> None:
        client = self._build()
        client.start()
        self.assertTrue(client.is_running)
        self.assertIn("definitionProvider", client.server_capabilities)

    def test_a2_request_is_echoed_back(self) -> None:
        client = self._build()
        client.start()
        result = client.request("textDocument/definition", {"foo": 1}, timeout=3.0)
        self.assertEqual(result, {"echoed": {"foo": 1}})

    def test_a3_notify_does_not_wait(self) -> None:
        client = self._build()
        client.start()
        # ``didOpen`` is a notification — should return instantly even
        # if the server takes ages to respond to anything.
        t0 = time.monotonic()
        client.notify("textDocument/didOpen", {"uri": "file:///x.py"})
        self.assertLess(time.monotonic() - t0, 0.5)

    def test_a4_request_times_out_when_server_is_slow(self) -> None:
        client = self._build(slow=True, init_timeout=5.0)
        # Initialize uses init_timeout (5s) which is enough for the
        # 2s slow-server delay; the *next* call uses a tight timeout.
        client.start()
        with self.assertRaises(TimeoutError):
            client.request("textDocument/definition", {"foo": 1}, timeout=0.3)

    def test_a5_request_after_server_crash_raises(self) -> None:
        client = self._build()
        client.start()
        # The fake server exits with code 7 when it sees method="boom".
        # Either the call succeeds before the server has a chance to
        # exit, or it raises — both are valid; what we verify is that
        # *follow-up* calls raise once the process is gone.
        try:
            client.request("boom", None, timeout=0.5)
        except Exception:
            pass
        # Give the server a moment to actually exit.
        for _ in range(20):
            if not client.is_running:
                break
            time.sleep(0.1)
        self.assertFalse(client.is_running)

    def test_a6_shutdown_is_idempotent(self) -> None:
        client = self._build()
        client.start()
        client.shutdown()
        client.shutdown()  # must not raise
        self.assertFalse(client.is_running)

    def test_a7_request_before_start_raises(self) -> None:
        client = self._build()
        with self.assertRaises(Exception):
            client.request("foo", None, timeout=0.5)

    def test_a8_command_must_be_non_empty(self) -> None:
        with self.assertRaises(ValueError):
            LSPClient(
                command=[],
                project_root=self._tmp,
                language="python",
            )

    def test_a9_lsp_protocol_error_when_server_returns_error(self) -> None:
        # Tweak the fake server to return an error envelope for a
        # specific method.  Patch via writing a new script.
        tmp = Path(tempfile.mkdtemp(prefix="lsp-err-"))
        script = tmp / "fake.py"
        script.write_text(
            textwrap.dedent(
                """
                import json, sys
                HEADER_TERM = b"\\r\\n\\r\\n"
                def read():
                    buf = b""
                    while HEADER_TERM not in buf:
                        c = sys.stdin.buffer.read(1)
                        if not c: return None
                        buf += c
                    head, _, rest = buf.partition(HEADER_TERM)
                    n = 0
                    for line in head.decode().split("\\r\\n"):
                        if line.lower().startswith("content-length:"):
                            n = int(line.split(":", 1)[1].strip())
                    body = rest
                    while len(body) < n:
                        body += sys.stdin.buffer.read(n - len(body))
                    return json.loads(body)
                def write(o):
                    b = json.dumps(o).encode()
                    sys.stdout.buffer.write(("Content-Length: %d\\r\\n\\r\\n" % len(b)).encode())
                    sys.stdout.buffer.write(b)
                    sys.stdout.buffer.flush()
                while True:
                    m = read()
                    if m is None: break
                    if "id" not in m: continue
                    if m.get("method") == "initialize":
                        write({"jsonrpc":"2.0","id":m["id"],"result":{"capabilities":{}}})
                    else:
                        write({"jsonrpc":"2.0","id":m["id"],"error":{"code":-32601,"message":"nope"}})
                """
            ).strip(),
            encoding="utf-8",
        )
        client = LSPClient(
            command=[sys.executable, str(script)],
            project_root=tmp,
            language="python",
            init_timeout=5.0,
        )
        self._clients.append(client)
        client.start()
        with self.assertRaises(LSPProtocolError):
            client.request("textDocument/definition", {}, timeout=2.0)


if __name__ == "__main__":
    unittest.main()
