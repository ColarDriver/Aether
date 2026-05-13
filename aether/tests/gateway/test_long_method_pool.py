"""Long-handler pool tests (PR 2).

Verify that ``long=True`` handlers do not block the dispatcher loop:
a short handler invoked while a long handler is in flight returns
without waiting on the long one.
"""

from __future__ import annotations

import io
import json
import threading
import time
import unittest

from aether.gateway.dispatcher import (
    _LONG_METHODS,
    dispatch_request,
    method,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.protocol import RpcRequest, RpcResponse
from aether.gateway.transport import (
    StdioTransport,
    bind_transport,
    reset_transport,
    reset_transport_for_tests,
)


class _LongHandlerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        reset_dispatcher_for_tests()
        reset_transport_for_tests()
        register_builtins()

    def tearDown(self) -> None:
        reset_dispatcher_for_tests()


class LongHandlerRoutesToPool(_LongHandlerTestCase):
    def test_long_request_returns_none_response_synchronously(self) -> None:
        """``dispatch_request`` returns ``None`` for long handlers; the worker writes."""
        gate = threading.Event()

        @method("test.long", long=True)
        def _h(params):
            gate.wait(timeout=2.0)
            return {"done": True}

        buf = io.StringIO()
        sink = StdioTransport(lambda: buf)
        token = bind_transport(sink)
        try:
            inline_response = dispatch_request(
                RpcRequest(id="1", method="test.long"),
                transport=sink,
            )
            self.assertIsNone(inline_response)
            # Worker is still blocked; buffer is empty.
            time.sleep(0.05)
            self.assertEqual(buf.getvalue(), "")

            gate.set()
            # Worker should write within a reasonable window.
            _wait_until(lambda: buf.getvalue() != "", timeout=2.0)
        finally:
            reset_transport(token)

        line = buf.getvalue().strip()
        parsed = json.loads(line)
        self.assertEqual(parsed["id"], "1")
        self.assertEqual(parsed["result"], {"done": True})

    def test_short_request_does_not_wait_on_long(self) -> None:
        """Long handler in flight does not block short handler dispatch."""
        gate = threading.Event()

        @method("test.long", long=True)
        def _long(params):
            gate.wait(timeout=2.0)
            return {"done": True}

        @method("test.short", long=False)
        def _short(params):
            return {"fast": True}

        buf = io.StringIO()
        sink = StdioTransport(lambda: buf)
        token = bind_transport(sink)
        try:
            # Fire long; it queues on the pool but the call returns immediately.
            dispatch_request(RpcRequest(id="L", method="test.long"), transport=sink)

            start = time.monotonic()
            short_response = dispatch_request(RpcRequest(id="S", method="test.short"))
            elapsed = time.monotonic() - start
            assert isinstance(short_response, RpcResponse)
            self.assertEqual(short_response.result, {"fast": True})
            # Generous threshold (200 ms) — main thread should not be
            # blocked at all by the gated long handler.
            self.assertLess(elapsed, 0.2)

            gate.set()
        finally:
            reset_transport(token)


class LongHandlerErrorMapping(_LongHandlerTestCase):
    def test_long_handler_exception_writes_internal_error(self) -> None:
        @method("test.long.boom", long=True)
        def _h(params):
            raise RuntimeError("workers can blow up too")

        buf = io.StringIO()
        sink = StdioTransport(lambda: buf)
        token = bind_transport(sink)
        try:
            dispatch_request(
                RpcRequest(id="L", method="test.long.boom"),
                transport=sink,
            )
            _wait_until(lambda: buf.getvalue() != "", timeout=2.0)
        finally:
            reset_transport(token)

        line = buf.getvalue().strip()
        parsed = json.loads(line)
        self.assertEqual(parsed["id"], "L")
        self.assertEqual(parsed["error"]["code"], -32603)
        self.assertEqual(parsed["error"]["data"]["exception"], "RuntimeError")


class PoolWorkerScales(_LongHandlerTestCase):
    """A burst of long requests runs concurrently up to the pool size."""

    def test_multiple_long_requests_overlap(self) -> None:
        in_flight = 0
        peak = 0
        lock = threading.Lock()
        gate = threading.Event()

        @method("test.long.many", long=True)
        def _h(params):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            gate.wait(timeout=2.0)
            with lock:
                in_flight -= 1
            return {"ok": True}

        buf = io.StringIO()
        sink = StdioTransport(lambda: buf)
        token = bind_transport(sink)
        try:
            for i in range(4):
                dispatch_request(
                    RpcRequest(id=f"L{i}", method="test.long.many"),
                    transport=sink,
                )
            # Wait until at least two workers are concurrently in flight.
            _wait_until(lambda: peak >= 2, timeout=2.0)
            gate.set()
            _wait_until(lambda: buf.getvalue().count("\n") >= 4, timeout=2.0)
        finally:
            reset_transport(token)

        self.assertGreaterEqual(peak, 2)


def _wait_until(predicate, *, timeout: float, step: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(step)
    raise AssertionError(
        f"predicate did not become true within {timeout:.2f}s"
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
