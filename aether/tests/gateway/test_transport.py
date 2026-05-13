"""Unit tests for the gateway transport layer (PR 1)."""

from __future__ import annotations

import concurrent.futures
import contextvars
import errno
import io
import json
import threading
import time
import unittest

from aether.gateway.transport import (
    StdioTransport,
    Transport,
    _PEER_GONE_ERRNOS,
    bind_transport,
    current_transport,
    reset_transport,
    reset_transport_for_tests,
)


class StdioTransportFraming(unittest.TestCase):
    def test_write_appends_single_newline(self) -> None:
        buf = io.StringIO()
        t = StdioTransport(lambda: buf)
        t.write({"hello": "world"})

        self.assertEqual(buf.getvalue(), '{"hello":"world"}\n')

    def test_write_serialises_unicode_without_ascii_escape(self) -> None:
        buf = io.StringIO()
        t = StdioTransport(lambda: buf)
        t.write({"text": "你好"})

        line = buf.getvalue()
        self.assertIn("你好", line)
        self.assertTrue(line.endswith("\n"))
        # Reparseable — i.e. one JSON object per line, no embedded raw newline.
        self.assertEqual(json.loads(line), {"text": "你好"})

    def test_control_chars_in_strings_are_escaped(self) -> None:
        buf = io.StringIO()
        t = StdioTransport(lambda: buf)
        t.write({"payload": "line1\nline2\rline3\tline4"})

        line = buf.getvalue()
        # Exactly one trailing newline; embedded controls have been escaped.
        self.assertEqual(line.count("\n"), 1)
        self.assertEqual(line.count("\r"), 0)
        self.assertEqual(
            json.loads(line),
            {"payload": "line1\nline2\rline3\tline4"},
        )

    def test_compact_separators(self) -> None:
        buf = io.StringIO()
        t = StdioTransport(lambda: buf)
        t.write({"a": 1, "b": 2})

        # No whitespace between fields.
        self.assertIn('"a":1,"b":2', buf.getvalue())


class StdioTransportConcurrency(unittest.TestCase):
    def test_concurrent_writes_do_not_interleave(self) -> None:
        buf = io.StringIO()
        # Wrap with a slow-write stream so any unprotected write would
        # show interleaving.
        slow = _SlowStream(buf, delay_per_char=0.0005)
        t = StdioTransport(lambda: slow)

        frames = [{"i": i, "tag": "X" * 64} for i in range(40)]

        def _writer(f: dict) -> None:
            t.write(f)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_writer, frames))

        lines = buf.getvalue().rstrip("\n").split("\n")
        self.assertEqual(len(lines), len(frames))
        # Each line round-trips cleanly — proves no interleaving.
        parsed = [json.loads(line) for line in lines]
        self.assertEqual(sorted(p["i"] for p in parsed), list(range(len(frames))))


class StdioTransportPeerGone(unittest.TestCase):
    def test_broken_pipe_silently_swallowed_and_marks_closed(self) -> None:
        broken = _BrokenStream(error=BrokenPipeError(errno.EPIPE, "broken"))
        t = StdioTransport(broken)  # _BrokenStream is itself the getter

        t.write({"x": 1})  # must not raise
        self.assertTrue(t.closed)
        self.assertEqual(broken.call_count, 1)

        # Subsequent writes are no-ops; the stream getter is not called again.
        before = broken.call_count
        t.write({"x": 2})
        self.assertEqual(broken.call_count, before)

    def test_connection_reset_silently_swallowed(self) -> None:
        stream = _BrokenStream(error=ConnectionResetError(errno.ECONNRESET, "reset"))
        t = StdioTransport(stream)
        t.write({"x": 1})

        self.assertTrue(t.closed)

    def test_ebadf_treated_as_peer_gone(self) -> None:
        stream = _BrokenStream(error=OSError(errno.EBADF, "bad fd"))
        t = StdioTransport(stream)
        t.write({"x": 1})

        self.assertTrue(t.closed)

    def test_non_peer_gone_oserror_reraises(self) -> None:
        stream = _BrokenStream(error=OSError(errno.ENOSPC, "disk full"))
        t = StdioTransport(stream)

        with self.assertRaises(OSError) as cm:
            t.write({"x": 1})
        self.assertEqual(cm.exception.errno, errno.ENOSPC)
        self.assertFalse(t.closed)

    def test_peer_gone_errnos_includes_expected(self) -> None:
        self.assertIn(errno.EPIPE, _PEER_GONE_ERRNOS)
        self.assertIn(errno.ECONNRESET, _PEER_GONE_ERRNOS)
        self.assertIn(errno.EBADF, _PEER_GONE_ERRNOS)
        self.assertIn(errno.ESHUTDOWN, _PEER_GONE_ERRNOS)


class StdioTransportClose(unittest.TestCase):
    def test_close_is_idempotent(self) -> None:
        t = StdioTransport(lambda: io.StringIO())
        t.close()
        t.close()  # no exception

        self.assertTrue(t.closed)

    def test_write_after_close_is_noop(self) -> None:
        buf = io.StringIO()
        t = StdioTransport(lambda: buf)
        t.close()
        t.write({"x": 1})

        self.assertEqual(buf.getvalue(), "")


class ContextvarTransport(unittest.TestCase):
    def setUp(self) -> None:
        reset_transport_for_tests()

    def test_fallback_when_unbound(self) -> None:
        t = current_transport()
        self.assertIsInstance(t, StdioTransport)
        # Fallback is sticky within the process.
        self.assertIs(current_transport(), t)

    def test_bind_and_reset_restores_prior(self) -> None:
        custom = StdioTransport(lambda: io.StringIO())
        token = bind_transport(custom)
        try:
            self.assertIs(current_transport(), custom)
        finally:
            reset_transport(token)

        # After reset, back to fallback.
        self.assertNotEqual(current_transport(), custom)

    def test_binding_visible_in_child_thread_via_copy_context(self) -> None:
        custom = StdioTransport(lambda: io.StringIO())
        bind_transport(custom)

        seen: dict[str, Transport] = {}

        def _worker() -> None:
            seen["t"] = current_transport()

        ctx = contextvars.copy_context()
        thread = threading.Thread(target=ctx.run, args=(_worker,))
        thread.start()
        thread.join()

        self.assertIs(seen["t"], custom)

    def test_binding_isolated_per_context(self) -> None:
        """Two threads with their own copy_context see independent bindings."""
        a = StdioTransport(lambda: io.StringIO())
        b = StdioTransport(lambda: io.StringIO())

        results: dict[str, Transport] = {}

        def _bind_and_read(name: str, t: Transport) -> None:
            bind_transport(t)
            results[name] = current_transport()

        ctx_a = contextvars.copy_context()
        ctx_b = contextvars.copy_context()

        ta = threading.Thread(target=ctx_a.run, args=(_bind_and_read, "a", a))
        tb = threading.Thread(target=ctx_b.run, args=(_bind_and_read, "b", b))
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        self.assertIs(results["a"], a)
        self.assertIs(results["b"], b)


class TransportProtocolConformance(unittest.TestCase):
    def test_stdio_transport_is_a_transport(self) -> None:
        self.assertIsInstance(StdioTransport(), Transport)


# --- helpers ---------------------------------------------------------


class _SlowStream:
    """Wraps a StringIO and slows writes character-by-character.

    Designed so that unprotected concurrent writes produce visible
    interleaving in the underlying buffer.
    """

    def __init__(self, sink: io.StringIO, delay_per_char: float = 0.0) -> None:
        self._sink = sink
        self._delay = delay_per_char

    def write(self, s: str) -> int:
        # Write one char at a time, sleeping between, to maximise the
        # window during which an unsynchronised second writer could
        # cut in.
        for ch in s:
            self._sink.write(ch)
            if self._delay:
                time.sleep(self._delay)
        return len(s)

    def flush(self) -> None:
        pass


class _BrokenStream:
    """Stream that always raises a given error on write.

    Tracks how many times the lazy stream getter ran (via ``call_count``)
    so tests can verify ``closed=True`` short-circuits subsequent writes.
    """

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.call_count = 0

    def __call__(self) -> "_BrokenStream":
        self.call_count += 1
        return self

    def write(self, s: str) -> int:
        raise self._error

    def flush(self) -> None:
        pass


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
