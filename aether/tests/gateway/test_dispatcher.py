"""Dispatcher tests: registry, routing, error mapping, frame parsing (PR 2)."""

from __future__ import annotations

import io
import json
import threading
import unittest

from aether.gateway.dispatcher import (
    dispatch_request,
    list_registered_methods,
    method,
    notify,
    parse_frame,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INTERNAL,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_PARSE,
    GatewayError,
    RpcNotification,
    RpcRequest,
    RpcResponse,
)
from aether.gateway.transport import (
    StdioTransport,
    bind_transport,
    reset_transport,
    reset_transport_for_tests,
)


class _DispatcherTestCase(unittest.TestCase):
    """Base case that wipes the registry + builtin ping each test."""

    def setUp(self) -> None:
        reset_dispatcher_for_tests()
        reset_transport_for_tests()
        # Re-register builtins so each test starts with a known baseline.
        register_builtins()

    def tearDown(self) -> None:
        reset_dispatcher_for_tests()


class RegistrationAndListing(_DispatcherTestCase):
    def test_registering_a_handler_makes_it_visible(self) -> None:
        @method("test.echo")
        def _h(params):
            return {"got": params}

        self.assertIn("test.echo", list_registered_methods())

    def test_long_flag_routes_to_pool(self) -> None:
        @method("test.long", long=True)
        def _h(params):  # pragma: no cover - just registration
            return {"ok": True}

        from aether.gateway.dispatcher import _LONG_METHODS

        self.assertIn("test.long", _LONG_METHODS)

    def test_reregistering_overwrites(self) -> None:
        @method("test.dup")
        def _v1(params):
            return {"v": 1}

        @method("test.dup")
        def _v2(params):
            return {"v": 2}

        req = RpcRequest(id="x", method="test.dup")
        resp = dispatch_request(req)
        assert isinstance(resp, RpcResponse)
        self.assertEqual(resp.result, {"v": 2})


class BuiltinPing(_DispatcherTestCase):
    def test_ping_responds_with_pong(self) -> None:
        req = RpcRequest(id="1", method="gateway.ping")
        resp = dispatch_request(req)
        assert isinstance(resp, RpcResponse)
        self.assertTrue(resp.result["pong"])

    def test_ping_echoes_optional_payload(self) -> None:
        req = RpcRequest(id="1", method="gateway.ping", params={"echo": "hi"})
        resp = dispatch_request(req)
        assert isinstance(resp, RpcResponse)
        self.assertEqual(resp.result["echo"], "hi")


class UnknownMethod(_DispatcherTestCase):
    def test_unknown_request_returns_method_not_found(self) -> None:
        req = RpcRequest(id="1", method="not.a.thing")
        resp = dispatch_request(req)
        assert isinstance(resp, RpcResponse)
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_METHOD_NOT_FOUND)
        self.assertIn("not.a.thing", resp.error.message)

    def test_unknown_notification_is_silently_dropped(self) -> None:
        n = RpcNotification(method="not.a.thing")
        self.assertIsNone(dispatch_request(n))


class HandlerErrorMapping(_DispatcherTestCase):
    def test_gateway_error_maps_to_application_code(self) -> None:
        @method("test.fail.app")
        def _h(params):
            raise GatewayError("nope", data={"hint": "policy"})

        resp = dispatch_request(RpcRequest(id="1", method="test.fail.app"))
        assert isinstance(resp, RpcResponse)
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_APPLICATION)
        self.assertEqual(resp.error.message, "nope")
        self.assertEqual(resp.error.data, {"hint": "policy"})

    def test_gateway_error_custom_code_passes_through(self) -> None:
        @method("test.fail.custom")
        def _h(params):
            raise GatewayError("rate", code=-32099, data={"retry": 5})

        resp = dispatch_request(RpcRequest(id="1", method="test.fail.custom"))
        assert isinstance(resp, RpcResponse)
        assert resp.error is not None
        self.assertEqual(resp.error.code, -32099)

    def test_unhandled_exception_maps_to_internal_error(self) -> None:
        @method("test.fail.boom")
        def _h(params):
            raise RuntimeError("oops")

        resp = dispatch_request(RpcRequest(id="1", method="test.fail.boom"))
        assert isinstance(resp, RpcResponse)
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_INTERNAL)
        self.assertEqual(resp.error.data, {"exception": "RuntimeError"})

    def test_exception_in_notification_yields_no_response(self) -> None:
        @method("test.fail.boom2")
        def _h(params):
            raise RuntimeError("oops")

        self.assertIsNone(dispatch_request(RpcNotification(method="test.fail.boom2")))


class HandlerReturnContract(_DispatcherTestCase):
    def test_handler_returning_none_yields_no_response(self) -> None:
        @method("test.deferred")
        def _h(params):
            return None  # "I'll respond later via respond()"

        self.assertIsNone(dispatch_request(RpcRequest(id="1", method="test.deferred")))

    def test_handler_params_passthrough(self) -> None:
        seen: list = []

        @method("test.passthrough")
        def _h(params):
            seen.append(params)
            return {}

        dispatch_request(RpcRequest(id="1", method="test.passthrough", params={"x": 1}))
        dispatch_request(RpcRequest(id="2", method="test.passthrough"))
        self.assertEqual(seen, [{"x": 1}, None])


class FrameParser(_DispatcherTestCase):
    def test_parses_request(self) -> None:
        env, err = parse_frame('{"jsonrpc":"2.0","id":"1","method":"gateway.ping"}')
        self.assertIsNone(err)
        self.assertIsInstance(env, RpcRequest)

    def test_parses_notification(self) -> None:
        env, err = parse_frame('{"jsonrpc":"2.0","method":"event","params":{"k":1}}')
        self.assertIsNone(err)
        self.assertIsInstance(env, RpcNotification)

    def test_invalid_json_returns_parse_error(self) -> None:
        env, err = parse_frame("{not json")
        self.assertIsNone(env)
        assert err is not None and err.error is not None
        self.assertEqual(err.error.code, ERROR_PARSE)
        self.assertIsNone(err.id)

    def test_batch_array_rejected(self) -> None:
        env, err = parse_frame('[{"jsonrpc":"2.0","id":"1","method":"x"}]')
        self.assertIsNone(env)
        assert err is not None and err.error is not None
        self.assertEqual(err.error.code, ERROR_INVALID_REQUEST)
        self.assertIn("batch", err.error.message.lower())

    def test_non_object_rejected(self) -> None:
        env, err = parse_frame("123")
        self.assertIsNone(env)
        assert err is not None and err.error is not None
        self.assertEqual(err.error.code, ERROR_INVALID_REQUEST)

    def test_extra_envelope_field_rejected(self) -> None:
        env, err = parse_frame(
            '{"jsonrpc":"2.0","id":"1","method":"x","junk":true}'
        )
        self.assertIsNone(env)
        assert err is not None and err.error is not None
        self.assertEqual(err.error.code, ERROR_INVALID_REQUEST)
        # The id we *could* read survives onto the error frame.
        self.assertEqual(err.id, "1")

    def test_bad_jsonrpc_version_rejected(self) -> None:
        env, err = parse_frame('{"jsonrpc":"1.0","id":"1","method":"x"}')
        self.assertIsNone(env)
        assert err is not None and err.error is not None
        self.assertEqual(err.error.code, ERROR_INVALID_REQUEST)


class NotifyEmitsOnCurrentTransport(_DispatcherTestCase):
    """``notify`` should write to whichever transport is bound right now."""

    def test_emits_on_explicit_binding(self) -> None:
        buf = io.StringIO()
        t = StdioTransport(lambda: buf)
        token = bind_transport(t)
        try:
            notify("gateway.ready", {"version": "test", "capabilities": []})
        finally:
            reset_transport(token)

        line = buf.getvalue().strip()
        parsed = json.loads(line)
        self.assertEqual(parsed["jsonrpc"], "2.0")
        self.assertEqual(parsed["method"], "gateway.ready")
        self.assertEqual(parsed["params"]["version"], "test")
        self.assertNotIn("id", parsed)


class DispatchUsesPerRequestContext(_DispatcherTestCase):
    """Long handlers run on the pool with a copied request context."""

    def test_short_handler_runs_synchronously(self) -> None:
        called_on = []

        @method("test.short")
        def _h(params):
            called_on.append(threading.current_thread().name)
            return {"ok": True}

        dispatch_request(RpcRequest(id="1", method="test.short"))
        self.assertEqual(called_on, [threading.current_thread().name])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
