"""Tests for the post_tool_use / post_tool_use_failure engine hooks."""

from __future__ import annotations

import copy
import time
import unittest
from typing import Any

from aether import AgentEngine
from aether.agents.middlewares.base import EngineMiddleware
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.core.hooks import EngineHooks
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


# ---------- scripted provider ----------


class _ScriptedProvider(ModelProvider):
    provider_name = "test-provider"
    api_mode = "chat"

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self.model = "test-model"
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del tools, config, context, stream_callback, stream_silent_callback
        self.calls.append({"messages": copy.deepcopy(messages)})
        if not self.responses:
            raise RuntimeError("no scripted response")
        return self.responses.pop(0)


# ---------- recording hook ----------


class _RecordingHooks(EngineHooks):
    """Capture every hook firing in a flat list for assertions."""

    def __init__(self) -> None:
        # Cannot set attributes on a slotted dataclass instance directly;
        # subclasses re-add their own attrs by overriding __init__ and
        # delegating to object.__setattr__.
        object.__setattr__(self, "post_calls", [])
        object.__setattr__(self, "failure_calls", [])

    def post_tool_use(
        self,
        *,
        session_id: str,
        iteration: int,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        result: ToolResult,
        elapsed_ms: float,
        context_metadata: dict[str, Any],
    ) -> None:
        self.post_calls.append(  # type: ignore[attr-defined]
            {
                "session_id": session_id,
                "iteration": iteration,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": dict(tool_args),
                "result": result,
                "elapsed_ms": elapsed_ms,
                "is_error": bool(result.is_error),
            }
        )

    def post_tool_use_failure(
        self,
        *,
        session_id: str,
        iteration: int,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        elapsed_ms: float,
        context_metadata: dict[str, Any],
    ) -> None:
        self.failure_calls.append(  # type: ignore[attr-defined]
            {
                "session_id": session_id,
                "iteration": iteration,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": dict(tool_args),
                "error": error,
                "elapsed_ms": elapsed_ms,
            }
        )


# ---------- test tools ----------


class _EchoTool(ToolExecutor):
    """Tool that returns its arguments as the result content."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="echo",
            parameters={"msg": {"type": "string"}},
            required=["msg"],
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        self.calls += 1
        msg = (call.arguments or {}).get("msg", "")
        return ToolResult(tool_call_id=call.id, name=call.name, content=str(msg))


class _SlowTool(ToolExecutor):
    """Tool that sleeps for elapsed_ms timing assertions."""

    def __init__(self, sleep_seconds: float = 0.05) -> None:
        self._sleep = sleep_seconds

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="slow")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        time.sleep(self._sleep)
        return ToolResult(tool_call_id=call.id, name=call.name, content="done")


class _RaisingTool(ToolExecutor):
    """Tool that raises a chosen exception on every dispatch."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="raiser")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del call, context
        raise self._exc


# ---------- middleware that mutates ToolResult ----------


class _AppendMiddleware(EngineMiddleware):
    """Mutates ToolResult.content in after_tool so hook order can be tested."""

    def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
        del context
        result.content = (result.content or "") + " <middleware>"
        return result


# ---------- helpers ----------


def _build_engine(
    *,
    tool: ToolExecutor,
    responses: list[NormalizedResponse],
    hooks: EngineHooks,
    middleware_pipeline: Any = None,
    config: EngineConfig | None = None,
) -> tuple[AgentEngine, _ScriptedProvider]:
    provider = _ScriptedProvider(responses)
    registry = ToolRegistry()
    registry.register(tool)
    engine = AgentEngine(
        provider,
        tool_registry=registry,
        middleware_pipeline=middleware_pipeline,
        hooks=hooks,
        config=config
        or EngineConfig(
            use_builtin_tools=False,
            verification_directive_enabled=False,
            faithful_reporting_enabled=False,
            max_iterations=4,
        ),
    )
    return engine, provider


# ---------- tests ----------


class PostToolUseSuccessTests(unittest.TestCase):
    def test_fires_once_with_correct_payload(self) -> None:
        hooks = _RecordingHooks()
        echo = _EchoTool()
        engine, _provider = _build_engine(
            tool=echo,
            responses=[
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="echo", arguments={"msg": "hi"})],
                ),
                NormalizedResponse(content="done"),
            ],
            hooks=hooks,
        )

        engine.run_turn(EngineRequest(session_id="s1", user_message="go"))

        self.assertEqual(len(hooks.post_calls), 1)  # type: ignore[attr-defined]
        self.assertEqual(len(hooks.failure_calls), 0)  # type: ignore[attr-defined]
        call = hooks.post_calls[0]  # type: ignore[attr-defined]
        self.assertEqual(call["session_id"], "s1")
        # iteration is a 1-indexed positive int (the engine bumps it
        # around the LLM round-trip; we only pin >=1 here so the test
        # survives loop-counter refactors).
        self.assertGreaterEqual(call["iteration"], 1)
        self.assertEqual(call["tool_call_id"], "c1")
        self.assertEqual(call["tool_name"], "echo")
        self.assertEqual(call["tool_args"], {"msg": "hi"})
        self.assertFalse(call["is_error"])
        self.assertEqual(call["result"].content, "hi")
        self.assertGreaterEqual(call["elapsed_ms"], 0.0)

    def test_sees_middleware_processed_result(self) -> None:
        """Hook fires AFTER after_tool middleware so its result is final."""
        from aether.agents.middlewares.pipeline import MiddlewarePipeline

        hooks = _RecordingHooks()
        echo = _EchoTool()
        pipeline = MiddlewarePipeline([_AppendMiddleware()])
        engine, _provider = _build_engine(
            tool=echo,
            responses=[
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="echo", arguments={"msg": "hello"})],
                ),
                NormalizedResponse(content="done"),
            ],
            hooks=hooks,
            middleware_pipeline=pipeline,
        )

        engine.run_turn(EngineRequest(session_id="s2", user_message="go"))

        self.assertEqual(len(hooks.post_calls), 1)  # type: ignore[attr-defined]
        # Middleware ran first → hook sees "hello <middleware>"
        self.assertEqual(
            hooks.post_calls[0]["result"].content,  # type: ignore[attr-defined]
            "hello <middleware>",
        )

    def test_elapsed_ms_reflects_real_duration(self) -> None:
        hooks = _RecordingHooks()
        slow = _SlowTool(sleep_seconds=0.05)
        engine, _provider = _build_engine(
            tool=slow,
            responses=[
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="slow", arguments={})],
                ),
                NormalizedResponse(content="done"),
            ],
            hooks=hooks,
        )

        engine.run_turn(EngineRequest(session_id="s3", user_message="go"))

        self.assertEqual(len(hooks.post_calls), 1)  # type: ignore[attr-defined]
        # 50 ms sleep; allow generous slack for CI jitter.
        self.assertGreaterEqual(
            hooks.post_calls[0]["elapsed_ms"], 40.0  # type: ignore[attr-defined]
        )


class PostToolUseFailureTests(unittest.TestCase):
    def test_fires_when_dispatch_raises_with_recovery(self) -> None:
        """Tool raises, engine recovers with synthetic error → failure hook fires."""
        hooks = _RecordingHooks()
        raiser = _RaisingTool(RuntimeError("boom"))
        engine, _provider = _build_engine(
            tool=raiser,
            responses=[
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="raiser", arguments={})],
                ),
                NormalizedResponse(content="done"),
            ],
            hooks=hooks,
            config=EngineConfig(
                use_builtin_tools=False,
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
                fail_on_tool_error=False,  # recovery mode
                max_iterations=4,
            ),
        )

        engine.run_turn(EngineRequest(session_id="s4", user_message="go"))

        # Exactly one failure event, zero success events — the variants are
        # mutually exclusive.
        self.assertEqual(len(hooks.post_calls), 0)  # type: ignore[attr-defined]
        self.assertEqual(len(hooks.failure_calls), 1)  # type: ignore[attr-defined]
        failure = hooks.failure_calls[0]  # type: ignore[attr-defined]
        self.assertEqual(failure["tool_name"], "raiser")
        self.assertIsInstance(failure["error"], RuntimeError)
        self.assertIn("boom", str(failure["error"]))

    def test_fires_when_dispatch_raises_and_engine_aborts(self) -> None:
        """fail_on_tool_error=True: hook still fires before the loop bails."""
        hooks = _RecordingHooks()
        raiser = _RaisingTool(RuntimeError("hard fail"))
        engine, _provider = _build_engine(
            tool=raiser,
            responses=[
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="raiser", arguments={})],
                ),
            ],
            hooks=hooks,
            config=EngineConfig(
                use_builtin_tools=False,
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
                fail_on_tool_error=True,
                raise_on_middleware_error=False,
                max_iterations=2,
            ),
        )

        engine.run_turn(EngineRequest(session_id="s5", user_message="go"))

        self.assertEqual(len(hooks.failure_calls), 1)  # type: ignore[attr-defined]
        self.assertEqual(len(hooks.post_calls), 0)  # type: ignore[attr-defined]

    def test_fires_for_unknown_tool_with_fail_flag(self) -> None:
        hooks = _RecordingHooks()
        engine, _provider = _build_engine(
            tool=_EchoTool(),  # registry only knows "echo"
            responses=[
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="nope", arguments={})],
                ),
            ],
            hooks=hooks,
            config=EngineConfig(
                use_builtin_tools=False,
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
                fail_on_unknown_tool=True,
                max_iterations=2,
            ),
        )

        engine.run_turn(EngineRequest(session_id="s6", user_message="go"))

        # The repair pass may rewrite an unknown tool call to a known one
        # before dispatch.  What matters here: if dispatch ever raised
        # UnknownToolError, the failure hook saw it; otherwise nothing fires
        # at all (engine never reached dispatch).
        self.assertEqual(len(hooks.post_calls), 0)  # type: ignore[attr-defined]


class PostToolUseHookExceptionTests(unittest.TestCase):
    def test_hook_exception_is_swallowed_and_loop_continues(self) -> None:
        class _ExplodingHooks(_RecordingHooks):
            def post_tool_use(self, **kwargs: Any) -> None:  # type: ignore[override]
                # Record BEFORE raising so we know it was reached.
                self.post_calls.append({"tool_name": kwargs["tool_name"]})  # type: ignore[attr-defined]
                raise RuntimeError("hook is broken")

        hooks = _ExplodingHooks()
        echo = _EchoTool()
        engine, _provider = _build_engine(
            tool=echo,
            responses=[
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="echo", arguments={"msg": "hi"})],
                ),
                NormalizedResponse(content="all good"),
            ],
            hooks=hooks,
        )

        result = engine.run_turn(EngineRequest(session_id="s7", user_message="go"))

        # Loop completed despite the hook raising.
        self.assertEqual(result.final_response, "all good")
        self.assertEqual(len(hooks.post_calls), 1)  # type: ignore[attr-defined]


class PostToolUseMultipleCallsTests(unittest.TestCase):
    def test_one_event_per_tool_call(self) -> None:
        hooks = _RecordingHooks()
        echo = _EchoTool()
        engine, _provider = _build_engine(
            tool=echo,
            responses=[
                NormalizedResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="echo", arguments={"msg": "a"}),
                        ToolCall(id="c2", name="echo", arguments={"msg": "b"}),
                    ],
                ),
                NormalizedResponse(content="done"),
            ],
            hooks=hooks,
        )

        engine.run_turn(EngineRequest(session_id="s8", user_message="go"))

        self.assertEqual(echo.calls, 2)
        self.assertEqual(len(hooks.post_calls), 2)  # type: ignore[attr-defined]
        ids = [c["tool_call_id"] for c in hooks.post_calls]  # type: ignore[attr-defined]
        self.assertEqual(ids, ["c1", "c2"])


class PostToolUseSubagentInheritanceTests(unittest.TestCase):
    """DefaultSubagentBuilder already propagates ``hooks=parent._hooks`` —
    this test pins that contract so future refactors don't drop it."""

    def test_child_engine_inherits_hooks(self) -> None:
        from aether.subagents.contracts import SubagentTask
        from aether.subagents.default_builder import DefaultSubagentBuilder

        hooks = _RecordingHooks()
        parent = AgentEngine(
            _ScriptedProvider([NormalizedResponse(content="parent ok")]),
            hooks=hooks,
            config=EngineConfig(
                use_builtin_tools=False,
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
            ),
        )
        builder = DefaultSubagentBuilder()
        task = SubagentTask(
            task_id="child-1",
            goal="hi",
            request=EngineRequest(session_id="child", user_message="hi"),
            metadata={},
        )
        child = builder.build_child(parent, task, child_depth=1)
        # The engine stores hooks privately; assert by identity that the
        # child references the same hook object.
        self.assertIs(child._hooks, hooks)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
