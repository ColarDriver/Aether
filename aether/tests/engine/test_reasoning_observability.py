from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineStatus,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.tools.base import ToolDescriptor


class DumpFailingProvider(ModelProvider):
    provider_name = "dump-provider"
    api_mode = "chat"

    def __init__(self, error: Exception) -> None:
        self.model = "dump-model"
        self.base_url = "https://dump.test/v1"
        self.error = error

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del messages, tools, config, context, stream_callback, stream_silent_callback
        raise self.error


class ReasoningProvider(ModelProvider):
    provider_name = "reasoning-provider"
    api_mode = "chat"

    def __init__(self, response: NormalizedResponse) -> None:
        self.model = "reasoning-model"
        self.response = response

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del messages, tools, config, context, stream_callback, stream_silent_callback
        return self.response


class ReasoningObservabilityTests(unittest.TestCase):
    def test_engine_result_populates_current_turn_reasoning(self) -> None:
        provider = ReasoningProvider(
            NormalizedResponse(
                content="answer",
                metadata={"reasoning_content": "current reasoning"},
            )
        )
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="reasoning-s",
                user_message="hello",
                messages=[
                    {
                        "role": "assistant",
                        "content": "old",
                        "metadata": {"reasoning_content": "old reasoning"},
                    }
                ],
            )
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(
            result.metadata["reasoning"]["last_reasoning"],
            "current reasoning",
        )

    def test_failed_request_dump_is_written_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = DumpFailingProvider(RuntimeError("provider boom"))
            engine = AgentEngine(
                provider,
                config=EngineConfig(
                    use_builtin_tools=False,
                    dump_failed_requests=True,
                    request_dump_dir=Path(tmp),
                ),
            )

            result = engine.run_turn(
                EngineRequest(session_id="dump-s", user_message="hello")
            )

            self.assertEqual(result.status, EngineStatus.FAILED)
            dump_meta = result.metadata["request_dump"]
            self.assertEqual(dump_meta["reason"], "non_retryable_client_error")
            dump_path = Path(dump_meta["path"])
            self.assertTrue(dump_path.exists())
            payload = json.loads(dump_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "dump-model")
            self.assertEqual(payload["provider"], "dump-provider")
            self.assertEqual(payload["base_url"], "https://dump.test/v1")
            self.assertIn("provider boom", payload["error"])
            self.assertEqual(payload["kwargs"]["messages"][0]["content"], "hello")

    def test_failed_request_dump_default_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = DumpFailingProvider(RuntimeError("provider boom"))
            engine = AgentEngine(
                provider,
                config=EngineConfig(
                    use_builtin_tools=False,
                    request_dump_dir=Path(tmp),
                ),
            )

            result = engine.run_turn(
                EngineRequest(session_id="dump-disabled", user_message="hello")
            )

            self.assertEqual(result.status, EngineStatus.FAILED)
            self.assertIsNone(result.metadata["request_dump"]["path"])
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
