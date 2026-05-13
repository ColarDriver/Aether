"""Tests for ``agent.cancel`` (PR 4)."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
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
from aether.gateway.protocol import RpcRequest, RpcResponse
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
    TurnContext,
)
from aether.tools.base import ToolDescriptor


class AgentCancel(unittest.TestCase):
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

    def test_cancel_is_idempotent_for_unknown_session(self) -> None:
        response = dispatch_request(
            RpcRequest(
                id="c1",
                method="agent.cancel",
                params={"session_id": "missing"},
            ),
            transport=self._sink,
        )
        assert isinstance(response, RpcResponse)
        self.assertEqual(response.result, {"ok": True})

    def test_cancel_interrupts_streaming_run(self) -> None:
        record = SessionRecord.new(
            session_id="ses_cancel",
            provider="mock",
            model="mock-model",
        )
        save_session(record)
        provider = _CancellableProvider()

        with mock.patch(
            "aether.gateway.handlers.agent_methods._build_provider_for_record",
            return_value=provider,
        ):
            run_response = dispatch_request(
                RpcRequest(
                    id="run-1",
                    method="agent.run",
                    params={"session_id": "ses_cancel", "user_message": "stream"},
                ),
                transport=self._sink,
            )
            self.assertIsNone(run_response)
            self.assertTrue(provider.first_delta.wait(timeout=2.0))

            cancel_response = dispatch_request(
                RpcRequest(
                    id="cancel-1",
                    method="agent.cancel",
                    params={"session_id": "ses_cancel"},
                ),
                transport=self._sink,
            )
            assert isinstance(cancel_response, RpcResponse)
            self.assertEqual(cancel_response.result, {"ok": True})

            provider.allow_next_delta.set()
            final = self._wait_for_response("run-1")

        self.assertEqual(final["result"]["exit_reason"], "cancelled")
        self.assertEqual(final["result"]["final_text"], "a")

        events = [
            frame["params"]
            for frame in self._frames()
            if frame.get("method") == "event"
        ]
        self.assertIn("cancelled", [event["type"] for event in events])
        for event in events:
            self.assertEqual(event["session_id"], "ses_cancel")
            self.assertEqual(event["run_id"], "run-1")

    def _frames(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self._buf.getvalue().splitlines()
            if line.strip()
        ]

    def _wait_for_response(self, request_id: str, *, timeout: float = 2.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for frame in self._frames():
                if frame.get("id") == request_id:
                    return frame
            time.sleep(0.01)
        self.fail(f"response {request_id!r} did not arrive")


class _CancellableProvider(ModelProvider):
    def __init__(self) -> None:
        self.first_delta = threading.Event()
        self.allow_next_delta = threading.Event()

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        if stream_callback is not None:
            stream_callback("a")
        self.first_delta.set()
        self.allow_next_delta.wait(timeout=2.0)
        if stream_callback is not None:
            stream_callback("b")
        return NormalizedResponse(content="ab")


def _test_config(max_iterations: object = None) -> EngineConfig:
    config = EngineConfig()
    config.max_iterations = int(max_iterations or 4)
    config.use_builtin_tools = False
    config.tool_use_contract_enabled = False
    config.memory_enabled = False
    config.summary_on_budget_exhausted = False
    return config


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
