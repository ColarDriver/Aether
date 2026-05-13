"""Tests for gateway tool-permission bridge (PR 5)."""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from aether.cli.sessions import SessionRecord, save_session
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.gateway.dispatcher import (
    dispatch_request,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.handlers import register_handler_methods
from aether.gateway.handlers.agent_methods import reset_agent_runs_for_tests
from aether.gateway.handlers.prompter_bridge import GatewayToolPermissionPrompter
from aether.gateway.handlers.state import reset_state_for_tests
from aether.gateway.protocol import RpcRequest, RpcResponse
from aether.gateway.reverse_rpc import (
    reject_all,
    reset_for_tests as reset_reverse_rpc_for_tests,
)
from aether.gateway.transport import (
    StdioTransport,
    bind_transport,
    reset_transport,
    reset_transport_for_tests,
)
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.tools.tool_permissions import (
    ToolPermissionDecisionType,
    ToolPermissionPreview,
    ToolPermissionRequest,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class GatewayToolPermissionPrompterTests(unittest.TestCase):
    def test_response_maps_to_tool_permission_decision(self) -> None:
        request = ToolPermissionRequest(
            session_id="ses_1",
            tool_call_id="tc_1",
            tool_name="file_edit",
            arguments={"path": "a.py"},
            category="write",
            risk="medium",
            preview=ToolPermissionPreview(title="Edit file", path="a.py"),
        )
        prompter = GatewayToolPermissionPrompter(run_id="run_1", request_timeout=9.0)
        with mock.patch(
            "aether.gateway.handlers.prompter_bridge.reverse_rpc.call",
            return_value={
                "type": "allow_session",
                "updated_arguments": {"path": "b.py"},
                "rule": {
                    "tool_name": "file_edit",
                    "behavior": "allow",
                    "scope": "session",
                    "path_prefix": "src/",
                },
            },
        ) as call:
            decision = prompter.request_tool_permission(request)

        self.assertEqual(decision.type, ToolPermissionDecisionType.ALLOW_SESSION)
        self.assertEqual(decision.updated_arguments, {"path": "b.py"})
        self.assertIsNotNone(decision.rule)
        assert decision.rule is not None
        self.assertEqual(decision.rule.path_prefix, "src/")
        method, params = call.call_args.args
        self.assertEqual(method, "permission.request")
        self.assertEqual(params["session_id"], "ses_1")
        self.assertEqual(params["run_id"], "run_1")
        self.assertEqual(params["deadline_ms"], 9_000)
        self.assertEqual(params["request"]["tool_call_id"], "tc_1")
        self.assertEqual(params["request"]["preview"]["title"], "Edit file")

    def test_timeout_returns_deny_decision(self) -> None:
        request = ToolPermissionRequest(
            session_id="ses_1",
            tool_call_id="tc_1",
            tool_name="write_file",
            arguments={},
            category="write",
            risk="medium",
        )
        prompter = GatewayToolPermissionPrompter(run_id="run_1")
        with mock.patch(
            "aether.gateway.handlers.prompter_bridge.reverse_rpc.call",
            side_effect=TimeoutError("too slow"),
        ):
            decision = prompter.request_tool_permission(request)
        self.assertEqual(decision.type, ToolPermissionDecisionType.DENY)
        self.assertEqual(decision.source, "timeout")


class AgentRunPermissionBridgeIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env = mock.patch.dict(os.environ, {"AETHER_HOME": self._tmp.name})
        self._env.start()
        self.addCleanup(self._env.stop)

        reset_dispatcher_for_tests()
        reset_transport_for_tests()
        reset_state_for_tests()
        reset_agent_runs_for_tests()
        reset_reverse_rpc_for_tests()
        register_builtins()
        register_handler_methods()

        self._buf = io.StringIO()
        self._sink = StdioTransport(lambda: self._buf)
        self._token = bind_transport(self._sink)
        self.addCleanup(reset_transport, self._token)
        self.addCleanup(reset_dispatcher_for_tests)
        self.addCleanup(reset_agent_runs_for_tests)
        self.addCleanup(reset_reverse_rpc_for_tests)

        record = SessionRecord.new(
            session_id="ses_perm",
            provider="mock",
            model="mock-model",
        )
        save_session(record)

    def test_permission_deny_skips_tool_execution(self) -> None:
        provider = _ToolThenTextProvider()
        tool = _WriteSpyTool()
        registry = ToolRegistry()
        registry.register(tool)

        with mock.patch(
            "aether.gateway.handlers.agent_methods._build_provider_for_record",
            return_value=provider,
        ):
            with mock.patch(
                "aether.gateway.handlers.agent_methods._build_tool_registry",
                return_value=registry,
            ):
                with mock.patch(
                    "aether.gateway.handlers.agent_methods._build_engine_config",
                    _test_config,
                ):
                    dispatch_request(
                        RpcRequest(
                            id="run-1",
                            method="agent.run",
                            params={
                                "session_id": "ses_perm",
                                "user_message": "write",
                            },
                        ),
                        transport=self._sink,
                    )
                    permission = self._wait_for_method("permission.request")
                    response = dispatch_request(
                        RpcRequest(
                            id=permission["id"],
                            method="permission.response",
                            params={"type": "deny", "feedback": "not now"},
                        ),
                        transport=self._sink,
                    )
                    assert isinstance(response, RpcResponse)
                    self.assertEqual(response.result, {"ok": True})
                    final = self._wait_for_response("run-1")

        self.assertEqual(final["result"]["exit_reason"], "done")
        self.assertEqual(tool.execute_calls, [])
        events = [
            frame["params"]
            for frame in self._frames()
            if frame.get("method") == "event"
        ]
        tool_results = [event for event in events if event["type"] == "tool.result"]
        self.assertEqual(len(tool_results), 1)
        self.assertIn("permission denied", tool_results[0]["content"])

    def test_peer_disconnect_mid_permission_returns_run_error(self) -> None:
        provider = _ToolThenTextProvider()
        tool = _WriteSpyTool()
        registry = ToolRegistry()
        registry.register(tool)

        with mock.patch(
            "aether.gateway.handlers.agent_methods._build_provider_for_record",
            return_value=provider,
        ):
            with mock.patch(
                "aether.gateway.handlers.agent_methods._build_tool_registry",
                return_value=registry,
            ):
                with mock.patch(
                    "aether.gateway.handlers.agent_methods._build_engine_config",
                    _test_config,
                ):
                    dispatch_request(
                        RpcRequest(
                            id="run-disconnect",
                            method="agent.run",
                            params={
                                "session_id": "ses_perm",
                                "user_message": "write",
                            },
                        ),
                        transport=self._sink,
                    )
                    self._wait_for_method("permission.request")
                    reject_all(OSError("peer disconnected"))
                    final = self._wait_for_response("run-disconnect")

        self.assertEqual(final["result"]["exit_reason"], "error")
        self.assertIn("peer disconnected", repr(final["result"]["metadata"]))
        self.assertEqual(tool.execute_calls, [])

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

    def _wait_for_response(self, request_id: str, *, timeout: float = 2.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for frame in self._frames():
                if frame.get("id") == request_id and "result" in frame:
                    return frame
            time.sleep(0.01)
        self.fail(f"{request_id} response did not arrive")


class _ToolThenTextProvider(ModelProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        self.calls += 1
        if self.calls == 1:
            return NormalizedResponse(
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="write_file",
                        arguments={"path": "/tmp/a.txt", "content": "x"},
                    )
                ]
            )
        return NormalizedResponse(content="noted")


class _WriteSpyTool(ToolExecutor):
    def __init__(self) -> None:
        self.execute_calls: list[ToolCall] = []

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="write_file")

    def build_permission_preview(
        self,
        call: ToolCall,
        context: TurnContext,
    ) -> ToolPermissionPreview:
        return ToolPermissionPreview(
            title="Write file",
            path=str(call.arguments.get("path") or ""),
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        self.execute_calls.append(call)
        return ToolResult(tool_call_id=call.id, name=call.name, content="wrote")


def _test_config(max_iterations: object = None) -> EngineConfig:
    config = EngineConfig()
    config.max_iterations = int(max_iterations or 4)
    config.use_builtin_tools = False
    config.tool_use_contract_enabled = False
    config.memory_enabled = False
    config.summary_on_budget_exhausted = False
    config.tool_permissions_enabled = True
    return config


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
