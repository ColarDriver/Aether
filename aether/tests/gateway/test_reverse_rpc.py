"""Tests for server-initiated reverse RPC (PR 5)."""

from __future__ import annotations

import io
import json
import threading
import time
import unittest

from aether.gateway import reverse_rpc
from aether.gateway.dispatcher import (
    dispatch_request,
    parse_frame,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.handlers import register_handler_methods
from aether.gateway.protocol import ERROR_INVALID_PARAMS, RpcRequest, RpcResponse
from aether.gateway.transport import (
    StdioTransport,
    bind_transport,
    reset_transport,
    reset_transport_for_tests,
)


class ReverseRpcTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_dispatcher_for_tests()
        reset_transport_for_tests()
        reverse_rpc.reset_for_tests()
        register_builtins()
        register_handler_methods()
        self._buf = io.StringIO()
        self._sink = StdioTransport(lambda: self._buf)
        self._token = bind_transport(self._sink)
        self.addCleanup(reset_transport, self._token)
        self.addCleanup(reset_dispatcher_for_tests)
        self.addCleanup(reverse_rpc.reset_for_tests)

    def test_call_writes_request_and_response_envelope_completes_future(self) -> None:
        result_holder: list[dict] = []
        errors: list[BaseException] = []

        def _worker() -> None:
            token = bind_transport(self._sink)
            try:
                result_holder.append(
                    reverse_rpc.call(
                        "approval.request",
                        {"kind": "plan"},
                        timeout=2.0,
                    )
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                reset_transport(token)

        thread = threading.Thread(target=_worker)
        thread.start()
        request = self._wait_for_method("approval.request")
        self.assertTrue(request["id"].startswith("srv_app_"))

        envelope, err = parse_frame(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"kind": "plan", "confirmed": True},
                }
            )
        )
        self.assertIsNone(err)
        assert envelope is not None
        response = dispatch_request(envelope, transport=self._sink)
        assert isinstance(response, RpcResponse)
        self.assertEqual(response.result, {"ok": True})

        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result_holder, [{"kind": "plan", "confirmed": True}])

    def test_unknown_response_id_returns_invalid_params(self) -> None:
        response = dispatch_request(
            RpcRequest(
                id="srv_app_missing",
                method="approval.response",
                params={"confirmed": True},
            ),
            transport=self._sink,
        )
        assert isinstance(response, RpcResponse)
        assert response.error is not None
        self.assertEqual(response.error.code, ERROR_INVALID_PARAMS)
        self.assertEqual(response.error.message, "unknown_pending")

    def test_duplicate_response_id_is_rejected_second_time(self) -> None:
        result_holder: list[dict] = []

        def _worker() -> None:
            token = bind_transport(self._sink)
            try:
                result_holder.append(
                    reverse_rpc.call("permission.request", {"x": 1}, timeout=2.0)
                )
            finally:
                reset_transport(token)

        thread = threading.Thread(target=_worker)
        thread.start()
        request = self._wait_for_method("permission.request")
        first = dispatch_request(
            RpcRequest(
                id=request["id"],
                method="permission.response",
                params={"type": "deny"},
            ),
            transport=self._sink,
        )
        second = dispatch_request(
            RpcRequest(
                id=request["id"],
                method="permission.response",
                params={"type": "deny"},
            ),
            transport=self._sink,
        )
        assert isinstance(first, RpcResponse)
        assert isinstance(second, RpcResponse)
        self.assertEqual(first.result, {"ok": True})
        assert second.error is not None
        self.assertEqual(second.error.message, "unknown_pending")
        thread.join(timeout=2.0)
        self.assertEqual(result_holder, [{"type": "deny"}])

    def test_timeout_drops_pending_id(self) -> None:
        with self.assertRaises(TimeoutError):
            reverse_rpc.call("approval.request", {"kind": "plan"}, timeout=0.01)
        self.assertEqual(reverse_rpc.pending_ids_for_tests(), [])

    def test_transport_close_rejects_pending_call(self) -> None:
        errors: list[BaseException] = []

        def _worker() -> None:
            token = bind_transport(self._sink)
            try:
                reverse_rpc.call("approval.request", {"kind": "plan"}, timeout=2.0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                reset_transport(token)

        thread = threading.Thread(target=_worker)
        thread.start()
        self._wait_for_method("approval.request")
        self._sink.close()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("peer disconnected", str(errors[0]))

    def _frames(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self._buf.getvalue().splitlines()
            if line.strip()
        ]

    def _wait_for_method(self, method: str, *, timeout: float = 2.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for frame in self._frames():
                if frame.get("method") == method:
                    return frame
            time.sleep(0.01)
        self.fail(f"{method} frame did not arrive")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
