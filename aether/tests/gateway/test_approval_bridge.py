"""Integration tests for gateway approval bridge (PR 5)."""

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
from aether.gateway.handlers.state import reset_state_for_tests
from aether.gateway.reverse_rpc import reset_for_tests as reset_reverse_rpc_for_tests
from aether.gateway.transport import (
    StdioTransport,
    bind_transport,
    reset_transport,
    reset_transport_for_tests,
)
from aether.gateway.protocol import RpcRequest, RpcResponse
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    TurnContext,
)
from aether.runtime.session.session_state import SessionMode, clear_mode, set_mode
from aether.tools.base import ToolDescriptor
from aether.tools.builtins.exit_plan_mode import ExitPlanModeTool
from aether.tools.registry import ToolRegistry


class AgentRunApprovalBridgeIntegration(unittest.TestCase):
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
        clear_mode("ses_plan")
        register_builtins()
        register_handler_methods()

        self._buf = io.StringIO()
        self._sink = StdioTransport(lambda: self._buf)
        self._token = bind_transport(self._sink)
        self.addCleanup(reset_transport, self._token)
        self.addCleanup(reset_dispatcher_for_tests)
        self.addCleanup(reset_agent_runs_for_tests)
        self.addCleanup(reset_reverse_rpc_for_tests)
        self.addCleanup(clear_mode, "ses_plan")

        record = SessionRecord.new(
            session_id="ses_plan",
            provider="mock",
            model="mock-model",
        )
        save_session(record)

    def test_plan_approval_yes_allows_exit_plan_mode(self) -> None:
        set_mode("ses_plan", SessionMode.PLAN)
        provider = _PlanProvider(final_text="implemented")
        registry = ToolRegistry()
        registry.register(ExitPlanModeTool())

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
                            id="run-plan",
                            method="agent.run",
                            params={
                                "session_id": "ses_plan",
                                "user_message": "apply plan",
                            },
                        ),
                        transport=self._sink,
                    )
                    approval = self._wait_for_method("approval.request")
                    self.assertEqual(approval["params"]["kind"], "plan")
                    self.assertEqual(approval["params"]["plan_text"], "1. edit")
                    response = dispatch_request(
                        RpcRequest(
                            id=approval["id"],
                            method="approval.response",
                            params={"kind": "plan", "confirmed": True},
                        ),
                        transport=self._sink,
                    )
                    assert isinstance(response, RpcResponse)
                    self.assertEqual(response.result, {"ok": True})
                    final = self._wait_for_response("run-plan")

        self.assertEqual(final["result"]["final_text"], "implemented")
        self.assertEqual(final["result"]["exit_reason"], "done")

    def test_plan_approval_no_keeps_tool_result_rejected(self) -> None:
        set_mode("ses_plan", SessionMode.PLAN)
        provider = _PlanProvider(final_text="will revise")
        registry = ToolRegistry()
        registry.register(ExitPlanModeTool())

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
                            id="run-reject",
                            method="agent.run",
                            params={
                                "session_id": "ses_plan",
                                "user_message": "apply plan",
                            },
                        ),
                        transport=self._sink,
                    )
                    approval = self._wait_for_method("approval.request")
                    dispatch_request(
                        RpcRequest(
                            id=approval["id"],
                            method="approval.response",
                            params={"kind": "plan", "confirmed": False},
                        ),
                        transport=self._sink,
                    )
                    final = self._wait_for_response("run-reject")

        self.assertEqual(final["result"]["final_text"], "will revise")
        events = [
            frame["params"]
            for frame in self._frames()
            if frame.get("method") == "event"
        ]
        tool_results = [event for event in events if event["type"] == "tool.result"]
        self.assertEqual(len(tool_results), 1)
        self.assertIn("did not approve", tool_results[0]["content"])

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


class _PlanProvider(ModelProvider):
    def __init__(self, *, final_text: str) -> None:
        self.calls = 0
        self.final_text = final_text

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
                        id="plan_1",
                        name="exit_plan_mode",
                        arguments={"plan": "1. edit"},
                    )
                ]
            )
        return NormalizedResponse(content=self.final_text)


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
