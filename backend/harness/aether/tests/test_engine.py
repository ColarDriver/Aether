from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.contracts import EngineRequest, EngineStatus, ExitReason, NormalizedResponse, ToolCall, ToolResult
from aether.runtime.hooks import EngineHooks
from aether.runtime.interrupts import InterruptController
from aether.runtime.session_store import InMemorySessionStore
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class SumTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="sum")

    def execute(self, call: ToolCall, context) -> ToolResult:
        total = int(call.arguments.get("a", 0)) + int(call.arguments.get("b", 0))
        return ToolResult(tool_call_id=call.id, name=call.name, content=str(total))


class TodoTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="todo")

    def execute(self, call: ToolCall, context) -> ToolResult:
        todos = call.arguments.get("todos", [])
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="ok",
            metadata={"todos": todos},
        )


class SpyHooks(EngineHooks):
    def __init__(self) -> None:
        self.session_starts: list[str] = []
        self.pre_llm_calls: list[tuple[str, int]] = []
        self.post_llm_calls: list[tuple[str, int]] = []
        self.session_ends: list[tuple[str, bool, bool]] = []

    def on_session_start(self, *, session_id: str, context_metadata: dict) -> None:
        self.session_starts.append(session_id)

    def pre_llm_call(self, *, session_id: str, iteration: int, messages: list[dict], context_metadata: dict) -> None:
        self.pre_llm_calls.append((session_id, iteration))

    def post_llm_call(self, *, session_id: str, iteration: int, response_text: str, context_metadata: dict) -> None:
        self.post_llm_calls.append((session_id, iteration))

    def on_session_end(self, *, session_id: str, completed: bool, interrupted: bool, context_metadata: dict) -> None:
        self.session_ends.append((session_id, completed, interrupted))


class EngineTests(unittest.TestCase):
    def test_completes_with_text_response(self) -> None:
        provider = ScriptedProvider([NormalizedResponse(content="hello world")])
        engine = AgentEngine(provider, config=EngineConfig(max_iterations=4))

        result = engine.run_turn(EngineRequest(session_id="s1", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(result.final_response, "hello world")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.messages[0]["role"], "user")
        self.assertEqual(result.messages[-1]["role"], "assistant")
        self.assertIsNotNone(result.task_id)
        self.assertIsNotNone(result.turn_id)

    def test_system_message_injected_as_first_message(self) -> None:
        provider = ScriptedProvider([NormalizedResponse(content="ok")])
        engine = AgentEngine(provider)

        result = engine.run_turn(
            EngineRequest(
                session_id="sys-1",
                user_message="hi",
                system_message="You are test system",
            )
        )

        self.assertEqual(result.messages[0]["role"], "system")
        self.assertEqual(result.messages[0]["content"], "You are test system")
        self.assertEqual(result.system_prompt, "You are test system")

    def test_session_store_reuses_system_prompt_on_resume(self) -> None:
        store = InMemorySessionStore()
        provider = ScriptedProvider([NormalizedResponse(content="first"), NormalizedResponse(content="second")])
        engine = AgentEngine(provider, session_store=store)

        first = engine.run_turn(
            EngineRequest(
                session_id="sess-1",
                user_message="hello",
                system_message="Persist me",
            )
        )
        second = engine.run_turn(
            EngineRequest(
                session_id="sess-1",
                user_message="again",
                messages=first.messages,
            )
        )

        self.assertEqual(first.system_prompt, "Persist me")
        self.assertEqual(second.system_prompt, "Persist me")
        self.assertEqual(second.messages[0]["role"], "system")

    def test_stream_callback_receives_delta(self) -> None:
        provider = ScriptedProvider([NormalizedResponse(content="streamed text")])
        engine = AgentEngine(provider)
        deltas: list[str] = []

        result = engine.run_turn(
            EngineRequest(
                session_id="stream-1",
                user_message="say",
                stream_callback=deltas.append,
            )
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertTrue(deltas)
        self.assertEqual("".join(deltas), "streamed text")
        self.assertTrue(result.streamed)
        self.assertGreater(result.metadata["turn"].get("stream_callback_calls", 0), 0)

    def test_tool_call_round_trip_then_text(self) -> None:
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="sum", arguments={"a": 2, "b": 3})],
                ),
                NormalizedResponse(content="done"),
            ]
        )
        registry = ToolRegistry()
        registry.register(SumTool())

        engine = AgentEngine(provider, tool_registry=registry, config=EngineConfig(max_iterations=5))
        result = engine.run_turn(EngineRequest(session_id="s2", user_message="calculate"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "done")
        self.assertEqual(result.iterations, 2)
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["content"], "5")

    def test_todo_hydration_restores_snapshot(self) -> None:
        provider = ScriptedProvider([NormalizedResponse(content="done")])
        engine = AgentEngine(provider, config=EngineConfig(enable_todo_hydration=True))
        history = [
            {
                "role": "tool",
                "tool_call_id": "todo-1",
                "name": "todo",
                "content": "[]",
                "metadata": {
                    "todos": [
                        {"id": "t1", "content": "A", "status": "pending"},
                        {"id": "t2", "content": "B", "status": "completed"},
                    ]
                },
            }
        ]

        result = engine.run_turn(
            EngineRequest(
                session_id="todo-1",
                user_message="continue",
                messages=history,
            )
        )

        self.assertEqual(result.metadata["turn"].get("todo_restored_count"), 2)
        self.assertEqual(len(result.metadata["turn"].get("todo_snapshot", [])), 2)

    def test_nudge_counters_trigger_metadata_flags(self) -> None:
        provider = ScriptedProvider(
            [
                NormalizedResponse(content="", tool_calls=[ToolCall(id="c1", name="sum", arguments={"a": 1, "b": 1})]),
                NormalizedResponse(content="done"),
            ]
        )
        registry = ToolRegistry()
        registry.register(SumTool())

        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(memory_nudge_interval=1, skill_nudge_interval=1, max_iterations=4),
        )
        result = engine.run_turn(EngineRequest(session_id="nudge-1", user_message="go"))

        self.assertTrue(result.metadata["turn"].get("should_review_memory"))
        self.assertTrue(result.metadata["turn"].get("should_review_skills"))

    def test_hooks_receive_lifecycle_events(self) -> None:
        hooks = SpyHooks()
        provider = ScriptedProvider([NormalizedResponse(content="ok")])
        engine = AgentEngine(provider, hooks=hooks)

        result = engine.run_turn(EngineRequest(session_id="hook-1", user_message="hello"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(hooks.session_starts, ["hook-1"])
        self.assertEqual(hooks.pre_llm_calls, [("hook-1", 1)])
        self.assertEqual(hooks.post_llm_calls, [("hook-1", 1)])
        self.assertEqual(hooks.session_ends, [("hook-1", True, False)])

    def test_interrupt_before_turn(self) -> None:
        provider = ScriptedProvider([NormalizedResponse(content="should not run")])
        interrupts = InterruptController()
        engine = AgentEngine(provider, interrupt_controller=interrupts)
        engine.interrupt("s3", "stop")

        result = engine.run_turn(EngineRequest(session_id="s3", user_message="hello"))

        self.assertEqual(result.status, EngineStatus.INTERRUPTED)
        self.assertEqual(result.exit_reason, ExitReason.INTERRUPTED)
        self.assertEqual(result.iterations, 0)

    def test_provider_error_fails(self) -> None:
        provider = ScriptedProvider([])
        engine = AgentEngine(provider)

        result = engine.run_turn(EngineRequest(session_id="s4", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.PROVIDER_ERROR)
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
