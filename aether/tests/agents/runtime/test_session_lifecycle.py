from __future__ import annotations

import copy
import unittest
from typing import Any

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse, TurnContext
from aether.runtime.core.hooks import EngineHooks
from aether.runtime.session import clear_mode, set_mode
from aether.runtime.session.session_state import SessionMode
from aether.runtime.session.session_store import InMemorySessionStore
from aether.tools.base import ToolDescriptor


class _RecordingProvider(ModelProvider):
    provider_name = "test"
    api_mode = "chat"

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: Any = None,
        stream_silent_callback: Any = None,
    ) -> NormalizedResponse:
        del tools, config, context, stream_callback, stream_silent_callback
        self.calls.append(copy.deepcopy(messages))
        return self.responses.pop(0)


class _Hooks(EngineHooks):
    def __init__(self) -> None:
        self.starts: list[str] = []
        self.ends: list[tuple[str, bool, bool]] = []

    def on_session_start(self, *, session_id: str, context_metadata: dict[str, Any]) -> None:
        del context_metadata
        self.starts.append(session_id)

    def on_session_end(
        self,
        *,
        session_id: str,
        completed: bool,
        interrupted: bool,
        context_metadata: dict[str, Any],
    ) -> None:
        del context_metadata
        self.ends.append((session_id, completed, interrupted))


def _engine(provider: ModelProvider, *, hooks: EngineHooks | None = None, store=None) -> AgentEngine:
    return AgentEngine(
        provider,
        hooks=hooks,
        session_store=store,
        config=EngineConfig(
            use_builtin_tools=False,
            verification_directive_enabled=False,
            faithful_reporting_enabled=False,
            verifier_gate_enabled=False,
        ),
    )


class SessionLifecycleTests(unittest.TestCase):
    def test_new_session_start_and_end_hooks_fire(self) -> None:
        hooks = _Hooks()
        provider = _RecordingProvider([NormalizedResponse(content="ok")])
        engine = _engine(provider, hooks=hooks)

        result = engine.run_turn(EngineRequest(session_id="life-1", user_message="hi"))

        self.assertEqual(result.final_response, "ok")
        self.assertEqual(hooks.starts, ["life-1"])
        self.assertEqual(hooks.ends, [("life-1", True, False)])

    def test_stored_system_prompt_is_reused_and_request_prompt_persists(self) -> None:
        store = InMemorySessionStore()
        provider = _RecordingProvider(
            [NormalizedResponse(content="first"), NormalizedResponse(content="second")]
        )
        engine = _engine(provider, store=store)

        first = engine.run_turn(
            EngineRequest(
                session_id="life-prompt",
                user_message="hi",
                system_message="Persist me",
            )
        )
        second = engine.run_turn(
            EngineRequest(
                session_id="life-prompt",
                user_message="again",
                messages=first.messages,
            )
        )

        self.assertEqual(second.system_prompt, "Persist me")
        self.assertEqual(provider.calls[1][0]["role"], "system")
        self.assertEqual(provider.calls[1][0]["content"], "Persist me")

    def test_plan_mode_marks_turn_metadata(self) -> None:
        session_id = "life-plan"
        self.addCleanup(clear_mode, session_id)
        set_mode(session_id, SessionMode.PLAN)
        provider = _RecordingProvider([NormalizedResponse(content="ok")])
        engine = _engine(provider)

        result = engine.run_turn(EngineRequest(session_id=session_id, user_message="hi"))

        self.assertTrue(result.metadata["turn"]["plan_mode_active"])


if __name__ == "__main__":
    unittest.main()

