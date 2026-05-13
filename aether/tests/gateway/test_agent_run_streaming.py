"""Tests for ``agent.run`` streaming events (PR 4)."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from aether.cli.sessions import SessionRecord, load_session, save_session
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.gateway.dispatcher import (
    _LONG_METHODS,
    dispatch_request,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.handlers import register_handler_methods
from aether.gateway.handlers.agent_methods import reset_agent_runs_for_tests
from aether.gateway.handlers.state import reset_state_for_tests
from aether.gateway.protocol import ERROR_APPLICATION, RpcRequest
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
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _GatewayAgentCase(unittest.TestCase):
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
        register_builtins()
        register_handler_methods()

        self._buf = io.StringIO()
        self._sink = StdioTransport(lambda: self._buf)
        self._token = bind_transport(self._sink)
        self.addCleanup(reset_transport, self._token)
        self.addCleanup(reset_dispatcher_for_tests)
        self.addCleanup(reset_agent_runs_for_tests)

        self._config_patch = mock.patch(
            "aether.gateway.handlers.agent_methods._build_engine_config",
            _test_config,
        )
        self._config_patch.start()
        self.addCleanup(self._config_patch.stop)

    def _save_session(self, session_id: str = "ses_agent") -> SessionRecord:
        record = SessionRecord.new(
            session_id=session_id,
            provider="mock",
            model="mock-model",
        )
        save_session(record)
        return record

    def _frames(self) -> list[dict]:
        text = self._buf.getvalue()
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def _wait_for_response(self, request_id: str, *, timeout: float = 2.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for frame in self._frames():
                if frame.get("id") == request_id:
                    return frame
            time.sleep(0.01)
        self.fail(f"response {request_id!r} did not arrive")

    def _run_agent(self, request_id: str, params: dict) -> dict:
        self._buf.seek(0)
        self._buf.truncate(0)
        response = dispatch_request(
            RpcRequest(id=request_id, method="agent.run", params=params),
            transport=self._sink,
        )
        self.assertIsNone(response)
        self.assertIn("agent.run", _LONG_METHODS)
        return self._wait_for_response(request_id)

    def _events(self) -> list[dict]:
        return [
            frame["params"]
            for frame in self._frames()
            if frame.get("method") == "event"
        ]


class _StreamingProvider(ModelProvider):
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        for chunk in self._chunks:
            if stream_callback is not None:
                stream_callback(chunk)
        return NormalizedResponse(
            content="".join(self._chunks),
            metadata={
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                }
            },
        )


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
                    ToolCall(id="tc_1", name="echo", arguments={"text": "hello"})
                ]
            )
        if stream_callback is not None:
            stream_callback("done")
        return NormalizedResponse(content="done")


class _BlockingProvider(ModelProvider):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        self.started.set()
        self.release.wait(timeout=2.0)
        return NormalizedResponse(content="done")


class _EchoTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="echo",
            description="echo text",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
            required=["text"],
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=str(call.arguments.get("text", "")),
        )


class AgentRunStreaming(_GatewayAgentCase):
    def test_streams_text_events_and_final_response(self) -> None:
        self._save_session()
        provider = _StreamingProvider(["hel", "lo"])
        with mock.patch(
            "aether.gateway.handlers.agent_methods._build_provider_for_record",
            return_value=provider,
        ):
            frame = self._run_agent(
                "r1",
                {"session_id": "ses_agent", "user_message": "say hello"},
            )

        self.assertEqual(frame["result"]["final_text"], "hello")
        self.assertEqual(frame["result"]["exit_reason"], "done")

        events = self._events()
        for event in events:
            self.assertEqual(event["session_id"], "ses_agent")
            self.assertEqual(event["run_id"], "r1")

        tags = [
            f"{event['type']}:{event.get('kind')}"
            if event["type"] == "status"
            else event["type"]
            for event in events
        ]
        _assert_subsequence(
            self,
            tags,
            [
                "status:thinking",
                "iteration.start",
                "text.delta",
                "text.delta",
                "iteration.end",
                "status:idle",
            ],
        )
        self.assertIn("usage", tags)
        self.assertIn("done", tags)

        deltas = [event for event in events if event["type"] == "text.delta"]
        self.assertEqual([event["text"] for event in deltas], ["hel", "lo"])
        self.assertEqual([event["sequence"] for event in deltas], [0, 1])

        saved = load_session("ses_agent")
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertGreaterEqual(len(saved.messages), 2)

    def test_tool_call_and_result_events_share_call_id(self) -> None:
        self._save_session()
        provider = _ToolThenTextProvider()
        registry = ToolRegistry()
        registry.register(_EchoTool())

        with mock.patch(
            "aether.gateway.handlers.agent_methods._build_provider_for_record",
            return_value=provider,
        ):
            with mock.patch(
                "aether.gateway.handlers.agent_methods._build_tool_registry",
                return_value=registry,
            ):
                frame = self._run_agent(
                    "tool-run",
                    {"session_id": "ses_agent", "user_message": "use echo"},
                )

        self.assertEqual(frame["result"]["exit_reason"], "done")
        events = self._events()
        calls = [event for event in events if event["type"] == "tool.call"]
        results = [event for event in events if event["type"] == "tool.result"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(calls[0]["tool_call_id"], "tc_1")
        self.assertEqual(results[0]["tool_call_id"], "tc_1")
        self.assertEqual(results[0]["content"], "hello")


class AgentRunActiveGuard(_GatewayAgentCase):
    def test_second_run_same_session_gets_application_error(self) -> None:
        self._save_session()
        provider = _BlockingProvider()
        with mock.patch(
            "aether.gateway.handlers.agent_methods._build_provider_for_record",
            return_value=provider,
        ):
            first = dispatch_request(
                RpcRequest(
                    id="r1",
                    method="agent.run",
                    params={"session_id": "ses_agent", "user_message": "first"},
                ),
                transport=self._sink,
            )
            self.assertIsNone(first)
            self.assertTrue(provider.started.wait(timeout=2.0))

            second = dispatch_request(
                RpcRequest(
                    id="r2",
                    method="agent.run",
                    params={"session_id": "ses_agent", "user_message": "second"},
                ),
                transport=self._sink,
            )
            self.assertIsNone(second)
            second_frame = self._wait_for_response("r2")

            provider.release.set()
            self._wait_for_response("r1")

        self.assertEqual(second_frame["error"]["code"], ERROR_APPLICATION)
        self.assertEqual(second_frame["error"]["message"], "RUN_ALREADY_ACTIVE")


def _test_config(max_iterations: object = None) -> EngineConfig:
    config = EngineConfig()
    config.max_iterations = int(max_iterations or 4)
    config.use_builtin_tools = False
    config.tool_use_contract_enabled = False
    config.memory_enabled = False
    config.summary_on_budget_exhausted = False
    return config


def _assert_subsequence(
    case: unittest.TestCase,
    values: list[str],
    expected: list[str],
) -> None:
    cursor = 0
    for item in expected:
        try:
            cursor = values.index(item, cursor) + 1
        except ValueError:
            case.fail(f"{item!r} was not found after index {cursor} in {values!r}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
