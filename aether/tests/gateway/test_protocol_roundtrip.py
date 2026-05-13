"""Round-trip + invariant tests for gateway protocol schemas (PR 2)."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from aether.gateway.protocol import (
    ApprovalQuestion,
    ApprovalRequest,
    ERROR_APPLICATION,
    ERROR_CANCELLED,
    ERROR_INTERNAL,
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_PARSE,
    Cancelled,
    Done,
    ERROR_TIMED_OUT,
    Error,
    GatewayError,
    GatewayErrorEvent,
    GatewayReady,
    IterationEnd,
    IterationStart,
    JSONRPC_VERSION,
    LoopStateChanged,
    PermissionPreview,
    PermissionRequest,
    PermissionToolRequest,
    Status,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolResult,
    RpcError,
    RpcNotification,
    RpcRequest,
    RpcResponse,
    make_error_response,
    make_success_response,
)


class ErrorCodeConstants(unittest.TestCase):
    """Pin the public error code constants so the wire contract is stable."""

    def test_standard_jsonrpc_codes(self) -> None:
        self.assertEqual(ERROR_PARSE, -32700)
        self.assertEqual(ERROR_INVALID_REQUEST, -32600)
        self.assertEqual(ERROR_METHOD_NOT_FOUND, -32601)
        self.assertEqual(ERROR_INVALID_PARAMS, -32602)
        self.assertEqual(ERROR_INTERNAL, -32603)

    def test_aether_application_codes(self) -> None:
        self.assertEqual(ERROR_APPLICATION, -32000)
        self.assertEqual(ERROR_TIMED_OUT, -32001)
        self.assertEqual(ERROR_CANCELLED, -32002)


class RpcRequestRoundTrip(unittest.TestCase):
    def test_minimal_request(self) -> None:
        req = RpcRequest(id="1", method="gateway.ping")
        dumped = req.model_dump(mode="json", exclude_none=True)
        self.assertEqual(
            dumped,
            {"jsonrpc": JSONRPC_VERSION, "id": "1", "method": "gateway.ping"},
        )

    def test_request_with_params(self) -> None:
        req = RpcRequest(id=42, method="session.list", params={"limit": 10})
        dumped = req.model_dump(mode="json", exclude_none=True)
        self.assertEqual(dumped["params"], {"limit": 10})
        self.assertEqual(dumped["id"], 42)

    def test_string_or_int_id_accepted(self) -> None:
        RpcRequest(id="abc", method="m")
        RpcRequest(id=99, method="m")

    def test_extra_field_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RpcRequest.model_validate(
                {"jsonrpc": "2.0", "id": "1", "method": "m", "junk": True}
            )

    def test_jsonrpc_version_pinned(self) -> None:
        with self.assertRaises(ValidationError):
            RpcRequest.model_validate({"jsonrpc": "1.0", "id": "1", "method": "m"})

    def test_method_required_and_nonempty(self) -> None:
        with self.assertRaises(ValidationError):
            RpcRequest.model_validate({"jsonrpc": "2.0", "id": "1", "method": ""})
        with self.assertRaises(ValidationError):
            RpcRequest.model_validate({"jsonrpc": "2.0", "id": "1"})

    def test_id_required_for_request(self) -> None:
        with self.assertRaises(ValidationError):
            RpcRequest.model_validate({"jsonrpc": "2.0", "method": "m"})


class RpcNotificationRoundTrip(unittest.TestCase):
    def test_no_id_allowed(self) -> None:
        n = RpcNotification(method="event", params={"x": 1})
        dumped = n.model_dump(mode="json", exclude_none=True)
        self.assertNotIn("id", dumped)
        self.assertEqual(dumped["method"], "event")

    def test_id_field_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RpcNotification.model_validate(
                {"jsonrpc": "2.0", "method": "event", "id": "1"}
            )


class RpcResponseRoundTrip(unittest.TestCase):
    def test_success_response(self) -> None:
        resp = RpcResponse(id="1", result={"ok": True})
        dumped = resp.model_dump(mode="json", exclude_none=True)
        self.assertEqual(
            dumped, {"jsonrpc": "2.0", "id": "1", "result": {"ok": True}}
        )

    def test_error_response(self) -> None:
        resp = RpcResponse(id="1", error=RpcError(code=-32601, message="nope"))
        dumped = resp.model_dump(mode="json", exclude_none=True)
        self.assertEqual(dumped["error"]["code"], -32601)
        self.assertEqual(dumped["error"]["message"], "nope")

    def test_response_rejects_both_result_and_error(self) -> None:
        with self.assertRaises(ValidationError):
            RpcResponse(id="1", result={"x": 1}, error=RpcError(code=-1, message="x"))

    def test_response_requires_at_least_one_of_result_or_error(self) -> None:
        with self.assertRaises(ValidationError):
            RpcResponse(id="1")

    def test_id_can_be_null_for_parse_error_responses(self) -> None:
        # JSON-RPC spec: when the server cannot parse the request id
        # (parse / invalid-request errors), id must be null in the
        # response.  Our schema accepts this.
        resp = make_error_response(None, ERROR_PARSE, "bad json")
        dumped = resp.model_dump(mode="json", exclude_none=False)
        self.assertIsNone(dumped["id"])


class HelperConstructors(unittest.TestCase):
    def test_make_error_response_shape(self) -> None:
        resp = make_error_response("rid", -32601, "method missing", data={"hint": 1})
        self.assertIsNone(resp.result)
        assert resp.error is not None
        self.assertEqual(resp.error.code, -32601)
        self.assertEqual(resp.error.message, "method missing")
        self.assertEqual(resp.error.data, {"hint": 1})

    def test_make_success_response_shape(self) -> None:
        resp = make_success_response(7, {"ok": True})
        self.assertIsNone(resp.error)
        self.assertEqual(resp.result, {"ok": True})
        self.assertEqual(resp.id, 7)


class GatewayErrorExceptionShape(unittest.TestCase):
    def test_defaults(self) -> None:
        err = GatewayError("boom")
        self.assertEqual(err.code, ERROR_APPLICATION)
        self.assertEqual(err.message, "boom")
        self.assertIsNone(err.data)

    def test_custom_code_and_data(self) -> None:
        err = GatewayError("rate limited", code=-32099, data={"retry_after": 5})
        self.assertEqual(err.code, -32099)
        self.assertEqual(err.data, {"retry_after": 5})


class EventModels(unittest.TestCase):
    def test_gateway_ready_default_type(self) -> None:
        evt = GatewayReady(version="0.2.0", capabilities=["ping"])
        dumped = evt.model_dump(mode="json")
        self.assertEqual(dumped["type"], "gateway.ready")
        self.assertEqual(dumped["version"], "0.2.0")
        self.assertEqual(dumped["capabilities"], ["ping"])

    def test_gateway_error_event(self) -> None:
        evt = GatewayErrorEvent(message="something went wrong", where="dispatcher")
        dumped = evt.model_dump(mode="json", exclude_none=True)
        self.assertEqual(dumped["type"], "gateway.error")
        self.assertEqual(dumped["where"], "dispatcher")

    def test_event_extra_field_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GatewayReady.model_validate(
                {"type": "gateway.ready", "version": "1", "capabilities": [], "extra": True}
            )

    def test_agent_event_models_carry_routing_fields(self) -> None:
        events = [
            TextDelta(session_id="s", run_id="r", text="hi", sequence=0),
            ToolCall(
                session_id="s",
                run_id="r",
                tool_call_id="tc",
                tool_name="echo",
                arguments={"x": 1},
                iteration=0,
            ),
            ToolResult(
                session_id="s",
                run_id="r",
                tool_call_id="tc",
                tool_name="echo",
                content="ok",
                iteration=0,
            ),
            IterationStart(session_id="s", run_id="r", iteration=0),
            IterationEnd(session_id="s", run_id="r", iteration=0),
            LoopStateChanged(session_id="s", run_id="r", state="LLM_CALL"),
            Status(session_id="s", run_id="r", kind="thinking"),
            TokenUsage(session_id="s", run_id="r", input_tokens=1, output_tokens=2),
            Done(session_id="s", run_id="r", final_text="ok"),
            Cancelled(session_id="s", run_id="r", reason="rpc-cancel"),
            Error(session_id="s", run_id="r", message="boom"),
        ]
        for event in events:
            dumped = event.model_dump(mode="json", exclude_none=True)
            self.assertEqual(dumped["session_id"], "s")
            self.assertEqual(dumped["run_id"], "r")
            self.assertIn("type", dumped)

    def test_approval_and_permission_request_models(self) -> None:
        approval = ApprovalRequest(
            kind="questions",
            session_id="s",
            run_id="r",
            questions=[
                ApprovalQuestion(
                    id="q1",
                    text="Choose",
                    kind="select",
                    options=["a", "b"],
                )
            ],
            deadline_ms=60_000,
        )
        dumped_approval = approval.model_dump(mode="json", exclude_none=False)
        self.assertEqual(dumped_approval["tool_call_id"], None)
        self.assertEqual(dumped_approval["questions"][0]["options"], ["a", "b"])

        permission = PermissionRequest(
            session_id="s",
            run_id="r",
            request=PermissionToolRequest(
                tool_call_id="tc",
                tool_name="file_edit",
                arguments={"path": "a.py"},
                category="write",
                risk="medium",
                preview=PermissionPreview(title="Edit file", path="a.py"),
            ),
            deadline_ms=120_000,
        )
        dumped_permission = permission.model_dump(mode="json", exclude_none=False)
        self.assertEqual(dumped_permission["request"]["tool_name"], "file_edit")
        self.assertEqual(dumped_permission["request"]["preview"]["path"], "a.py")


class JsonRoundTrip(unittest.TestCase):
    """Serialise → text → parse — ensures encoders agree with validators."""

    def test_request_text_roundtrip(self) -> None:
        req = RpcRequest(id="x", method="m", params={"a": 1})
        text = json.dumps(req.model_dump(mode="json", exclude_none=True))
        parsed = RpcRequest.model_validate_json(text)
        self.assertEqual(parsed, req)

    def test_response_text_roundtrip(self) -> None:
        resp = RpcResponse(id=1, result={"ok": True})
        text = json.dumps(resp.model_dump(mode="json", exclude_none=True))
        parsed = RpcResponse.model_validate_json(text)
        self.assertEqual(parsed.id, 1)
        self.assertEqual(parsed.result, {"ok": True})

    def test_notification_text_roundtrip(self) -> None:
        n = RpcNotification(method="event", params={"k": "v"})
        text = json.dumps(n.model_dump(mode="json", exclude_none=True))
        parsed = RpcNotification.model_validate_json(text)
        self.assertEqual(parsed, n)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
