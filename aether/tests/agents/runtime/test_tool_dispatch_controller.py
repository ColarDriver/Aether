from __future__ import annotations

import logging
import threading
import time
import unittest
from typing import Any

from aether.agents.middlewares.base import EngineMiddleware
from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.agents.runtime.tool_dispatch import (
    ToolDispatchController,
    ToolDispatchRequest,
    ToolDispatchResult,
)
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.control.interrupts import InterruptController
from aether.runtime.core.contracts import (
    EngineRequest,
    ExitReason,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.core.hooks import EngineHooks
from aether.runtime.core.services import EngineServices
from aether.runtime.recovery.strategies import GenericBackoffStrategy
from aether.runtime.session.session_state import SessionMode, clear_mode, set_mode
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _EchoTool(ToolExecutor):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="echo")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        self.calls += 1
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=f"ok:{call.arguments.get('value', '')}",
            metadata={"edited_paths": call.arguments.get("edited_paths", [])},
        )


class _FailTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="fail")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del call, context
        raise RuntimeError("tool exploded")


class _ReadFileTool(ToolExecutor):
    def __init__(self, *, delay: float = 0.0, fail_path: str | None = None) -> None:
        self.delay = delay
        self.fail_path = fail_path
        self.calls: list[str] = []
        self._lock = threading.Lock()

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="read_file")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        path = str(call.arguments.get("path", ""))
        with self._lock:
            self.calls.append(path)
        if self.delay:
            time.sleep(self.delay)
        if path == self.fail_path:
            raise RuntimeError("read failed")
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=f"read:{path}",
        )


class _WriteFileTool(ToolExecutor):
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="write_file")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        path = str(call.arguments.get("path", ""))
        self.calls.append(path)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=f"wrote:{path}",
        )


class _AfterToolMiddleware(EngineMiddleware):
    def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
        del context
        result.metadata["after_tool"] = True
        result.content = f"{result.content}:after"
        return result


class _Adapter:
    def __init__(self) -> None:
        self.permission_result: ToolResult | None = None
        self.post_hooks: list[dict[str, Any]] = []
        self.failure_hooks: list[dict[str, Any]] = []
        self.edited_paths_seen: list[list[str]] = []
        self.diagnostic_updates = 0
        self.pending_steer_calls = 0
        self.cheap_names: set[str] = set()

    def maybe_inject_schema_errors(self, **kwargs: Any) -> None:
        del kwargs
        return None

    def apply_pending_steer_to_tool_results(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.pending_steer_calls += 1

    def is_interrupted(self, session_id: str, context: TurnContext) -> bool:
        del session_id, context
        return False

    def record_interrupt_metadata(self, context: TurnContext, *, was_in_tool_call: bool) -> None:
        context.metadata["interrupt"] = {"was_in_tool_call": was_in_tool_call}

    def apply_tool_permission_gate(
        self,
        call: ToolCall,
        *,
        request: EngineRequest,
        context: TurnContext,
    ) -> ToolCall | ToolResult:
        del request, context
        return self.permission_result or call

    def handle_pipeline_error(self, error: Exception, state: Any, context: TurnContext) -> None:
        del state
        context.metadata["handled_pipeline_error"] = str(error)

    def format_unknown_tool_content(self, tool_name: str, *, context: TurnContext) -> str:
        del context
        return f"Unknown tool: {tool_name}"

    def fire_post_tool_hook(
        self,
        *,
        tool_call: ToolCall,
        result: ToolResult | None,
        dispatch_error: BaseException | None,
        elapsed_ms: float,
        session_id: str,
        iteration: int,
        context: TurnContext,
    ) -> None:
        del elapsed_ms, context
        payload = {
            "tool": tool_call.name,
            "result": result,
            "error": dispatch_error,
            "session_id": session_id,
            "iteration": iteration,
        }
        if dispatch_error is None:
            self.post_hooks.append(payload)
        else:
            self.failure_hooks.append(payload)

    def accumulate_edited_paths(self, result: ToolResult | None, context: TurnContext) -> None:
        del context
        edited = []
        if result is not None:
            edited = list(result.metadata.get("edited_paths") or [])
        self.edited_paths_seen.append(edited)

    def maybe_mark_verifier_invoked(self, tool_call: ToolCall, context: TurnContext) -> None:
        del tool_call, context

    def dispatch_internal_diagnostic_update(
        self,
        *,
        tool_call: ToolCall,
        result: ToolResult,
        context: TurnContext,
    ) -> None:
        del tool_call, result, context
        self.diagnostic_updates += 1

    def record_tool_result_error(self, context: TurnContext, result: ToolResult) -> None:
        if result.is_error:
            context.metadata["tool_result_errors"] = (
                int(context.metadata.get("tool_result_errors", 0)) + 1
            )

    def append_tool_result_message(
        self,
        messages: list[dict[str, Any]],
        result: ToolResult,
    ) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "name": result.name,
                "content": result.content,
                "is_error": result.is_error,
                "metadata": dict(result.metadata),
            }
        )

    def is_permission_abort_result(self, result: ToolResult) -> bool:
        return bool(result.metadata.get("permission_abort"))

    def is_cheap_tool(self, tool_name: str) -> bool:
        return tool_name in self.cheap_names


def _controller(
    registry: ToolRegistry,
    adapter: _Adapter,
    *,
    config: EngineConfig | None = None,
    middleware: MiddlewarePipeline | None = None,
) -> ToolDispatchController:
    services = EngineServices(
        provider=ScriptedProvider([NormalizedResponse(content="unused")]),
        tool_registry=registry,
        middleware_pipeline=middleware or MiddlewarePipeline(),
        interrupt_controller=InterruptController(),
        logger=logging.getLogger(__name__),
        recovery_strategy=GenericBackoffStrategy(),
    )
    return ToolDispatchController(
        services=services,
        hooks=EngineHooks(),
        config=config or EngineConfig(use_builtin_tools=False),
        adapter=adapter,
    )


def _dispatch(
    controller: ToolDispatchController,
    call: ToolCall,
) -> tuple[ToolDispatchResult, TurnContext, list[dict[str, Any]]]:
    return _dispatch_many(controller, [call])


def _dispatch_many(
    controller: ToolDispatchController,
    calls: list[ToolCall],
) -> tuple[ToolDispatchResult, TurnContext, list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "", "tool_calls": []}
    ]
    context = TurnContext(session_id="tools", iteration=1, metadata={})
    result = controller.dispatch(
        ToolDispatchRequest(
            tool_calls=calls,
            messages=messages,
            context=context,
            request=EngineRequest(session_id="tools"),
            iteration=1,
            tool_result_start_idx=1,
        )
    )
    return result, context, messages


class ToolDispatchControllerTests(unittest.TestCase):
    def test_read_only_tool_dispatch_success(self) -> None:
        registry = ToolRegistry()
        tool = _EchoTool()
        registry.register(tool)
        adapter = _Adapter()
        controller = _controller(registry, adapter)

        result, _context, messages = _dispatch(
            controller,
            ToolCall(id="call-1", name="echo", arguments={"value": "x"}),
        )

        self.assertIsNone(result.exit_reason)
        self.assertEqual(result.dispatched_count, 1)
        self.assertEqual(tool.calls, 1)
        self.assertEqual(messages[-1]["content"], "ok:x")
        self.assertEqual(len(adapter.post_hooks), 1)
        self.assertEqual(adapter.pending_steer_calls, 1)

    def test_after_tool_runs_before_post_hook(self) -> None:
        registry = ToolRegistry()
        registry.register(_EchoTool())
        adapter = _Adapter()
        controller = _controller(
            registry,
            adapter,
            middleware=MiddlewarePipeline([_AfterToolMiddleware()]),
        )

        result, _context, messages = _dispatch(
            controller,
            ToolCall(id="call-1", name="echo", arguments={"value": "x"}),
        )

        self.assertIsNone(result.exit_reason)
        self.assertEqual(messages[-1]["content"], "ok:x:after")
        self.assertTrue(adapter.post_hooks[0]["result"].metadata["after_tool"])

    def test_permission_result_skips_registry_dispatch(self) -> None:
        registry = ToolRegistry()
        tool = _EchoTool()
        registry.register(tool)
        adapter = _Adapter()
        adapter.permission_result = ToolResult(
            tool_call_id="call-1",
            name="echo",
            content="denied",
            is_error=True,
            metadata={"permission_denied": True},
        )
        controller = _controller(registry, adapter)

        result, context, messages = _dispatch(
            controller,
            ToolCall(id="call-1", name="echo", arguments={"value": "x"}),
        )

        self.assertIsNone(result.exit_reason)
        self.assertEqual(tool.calls, 0)
        self.assertEqual(messages[-1]["content"], "denied")
        self.assertEqual(context.metadata["tool_result_errors"], 1)
        self.assertEqual(adapter.post_hooks, [])

    def test_strict_tool_error_triggers_failure_hook(self) -> None:
        registry = ToolRegistry()
        registry.register(_FailTool())
        adapter = _Adapter()
        controller = _controller(
            registry,
            adapter,
            config=EngineConfig(use_builtin_tools=False, fail_on_tool_error=True),
        )

        result, context, _messages = _dispatch(
            controller,
            ToolCall(id="call-1", name="fail", arguments={}),
        )

        self.assertEqual(result.exit_reason, ExitReason.TOOL_ERROR)
        self.assertEqual(result.error_text, "tool exploded")
        self.assertEqual(context.metadata["handled_pipeline_error"], "tool exploded")
        self.assertEqual(len(adapter.failure_hooks), 1)
        self.assertIsInstance(adapter.failure_hooks[0]["error"], RuntimeError)

    def test_duplicate_calls_are_deduped_with_synthetic_result(self) -> None:
        registry = ToolRegistry()
        tool = _EchoTool()
        registry.register(tool)
        adapter = _Adapter()
        controller = _controller(registry, adapter)
        messages = [{"role": "assistant", "content": "", "tool_calls": []}]
        context = TurnContext(session_id="tools", iteration=1, metadata={})

        result = controller.dispatch(
            ToolDispatchRequest(
                tool_calls=[
                    ToolCall(id="call-1", name="echo", arguments={"value": "x"}),
                    ToolCall(id="call-2", name="echo", arguments={"value": "x"}),
                ],
                messages=messages,
                context=context,
                request=EngineRequest(session_id="tools"),
                iteration=1,
                tool_result_start_idx=1,
            )
        )

        self.assertIsNone(result.exit_reason)
        self.assertEqual(tool.calls, 1)
        self.assertEqual(context.metadata["tool_calls_deduped"], 1)
        self.assertEqual(len([m for m in messages if m.get("role") == "tool"]), 2)

    def test_cheap_tool_batch_surfaces_refund_signal(self) -> None:
        registry = ToolRegistry()
        registry.register(_EchoTool())
        adapter = _Adapter()
        adapter.cheap_names.add("echo")
        controller = _controller(registry, adapter)

        result, _context, _messages = _dispatch(
            controller,
            ToolCall(id="call-1", name="echo", arguments={"value": "x"}),
        )

        self.assertTrue(result.all_tools_cheap)

    def test_safe_read_only_batch_parallelizes(self) -> None:
        registry = ToolRegistry()
        tool = _ReadFileTool(delay=0.08)
        registry.register(tool)
        adapter = _Adapter()
        controller = _controller(
            registry,
            adapter,
            config=EngineConfig(
                use_builtin_tools=False,
                parallel_tool_execution_enabled=True,
                parallel_tool_max_workers=2,
            ),
        )

        started = time.perf_counter()
        result, context, messages = _dispatch_many(
            controller,
            [
                ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="call-2", name="read_file", arguments={"path": "b.py"}),
            ],
        )
        elapsed = time.perf_counter() - started

        self.assertIsNone(result.exit_reason)
        self.assertTrue(result.parallel_executed)
        self.assertEqual(result.dispatched_count, 2)
        self.assertLess(elapsed, 0.15)
        self.assertEqual([m["content"] for m in messages if m.get("role") == "tool"], ["read:a.py", "read:b.py"])
        self.assertEqual(context.metadata["tool_parallel"]["executed"], True)
        self.assertIsNone(context.metadata["tool_parallel"]["fallback_reason"])

    def test_mixed_safe_unsafe_batch_falls_back_to_sequential(self) -> None:
        registry = ToolRegistry()
        read_tool = _ReadFileTool()
        write_tool = _WriteFileTool()
        registry.register(read_tool)
        registry.register(write_tool)
        adapter = _Adapter()
        controller = _controller(
            registry,
            adapter,
            config=EngineConfig(
                use_builtin_tools=False,
                parallel_tool_execution_enabled=True,
            ),
        )

        result, context, messages = _dispatch_many(
            controller,
            [
                ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="call-2", name="write_file", arguments={"path": "b.py"}),
            ],
        )

        self.assertFalse(result.parallel_executed)
        self.assertEqual(result.dispatched_count, 2)
        self.assertEqual(read_tool.calls, ["a.py"])
        self.assertEqual(write_tool.calls, ["b.py"])
        self.assertEqual([m["content"] for m in messages if m.get("role") == "tool"], ["read:a.py", "wrote:b.py"])
        self.assertEqual(
            context.metadata["tool_parallel_fallback_reason"],
            "tool-not-parallel-safe:write_file",
        )

    def test_plan_mode_blocked_write_does_not_enter_parallel_pool(self) -> None:
        registry = ToolRegistry()
        write_tool = _WriteFileTool()
        registry.register(_ReadFileTool())
        registry.register(write_tool)
        adapter = _Adapter()
        controller = _controller(
            registry,
            adapter,
            config=EngineConfig(
                use_builtin_tools=False,
                parallel_tool_execution_enabled=True,
            ),
        )
        set_mode("tools", SessionMode.PLAN)
        try:
            result, context, messages = _dispatch_many(
                controller,
                [
                    ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"}),
                    ToolCall(id="call-2", name="write_file", arguments={"path": "b.py"}),
                ],
            )
        finally:
            clear_mode("tools")

        tool_messages = [m for m in messages if m.get("role") == "tool"]
        self.assertFalse(result.parallel_executed)
        self.assertEqual(write_tool.calls, [])
        self.assertEqual(context.metadata["tool_parallel_executed"], False)
        self.assertEqual(
            context.metadata["tool_parallel_fallback_reason"],
            "tool-not-parallel-safe:write_file",
        )
        self.assertTrue(tool_messages[-1]["metadata"]["plan_mode_blocked"])

    def test_parallel_result_order_and_post_hooks_are_original_order(self) -> None:
        registry = ToolRegistry()
        registry.register(_ReadFileTool())
        adapter = _Adapter()
        controller = _controller(
            registry,
            adapter,
            config=EngineConfig(
                use_builtin_tools=False,
                parallel_tool_execution_enabled=True,
                parallel_tool_max_workers=2,
            ),
        )

        result, _context, messages = _dispatch_many(
            controller,
            [
                ToolCall(id="call-1", name="read_file", arguments={"path": "slow.py"}),
                ToolCall(id="call-2", name="read_file", arguments={"path": "fast.py"}),
            ],
        )

        self.assertTrue(result.parallel_executed)
        self.assertEqual(
            [m["tool_call_id"] for m in messages if m.get("role") == "tool"],
            ["call-1", "call-2"],
        )
        self.assertEqual(
            [hook["result"].tool_call_id for hook in adapter.post_hooks],
            ["call-1", "call-2"],
        )
        self.assertEqual(adapter.diagnostic_updates, 2)

    def test_parallel_worker_error_still_appends_ordered_tool_results(self) -> None:
        registry = ToolRegistry()
        registry.register(_ReadFileTool(fail_path="b.py"))
        adapter = _Adapter()
        controller = _controller(
            registry,
            adapter,
            config=EngineConfig(
                use_builtin_tools=False,
                parallel_tool_execution_enabled=True,
                parallel_tool_max_workers=2,
            ),
        )

        result, context, messages = _dispatch_many(
            controller,
            [
                ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="call-2", name="read_file", arguments={"path": "b.py"}),
            ],
        )

        tool_messages = [m for m in messages if m.get("role") == "tool"]
        self.assertTrue(result.parallel_executed)
        self.assertEqual([m["tool_call_id"] for m in tool_messages], ["call-1", "call-2"])
        self.assertFalse(tool_messages[0]["is_error"])
        self.assertTrue(tool_messages[1]["is_error"])
        self.assertEqual(context.metadata["tool_result_errors"], 1)
        self.assertTrue(tool_messages[1]["metadata"]["parallel_worker_error"])

    def test_parallel_cheap_tool_batch_surfaces_refund_signal(self) -> None:
        registry = ToolRegistry()
        registry.register(_ReadFileTool())
        adapter = _Adapter()
        adapter.cheap_names.add("read_file")
        controller = _controller(
            registry,
            adapter,
            config=EngineConfig(
                use_builtin_tools=False,
                parallel_tool_execution_enabled=True,
                parallel_tool_max_workers=2,
            ),
        )

        result, _context, _messages = _dispatch_many(
            controller,
            [
                ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="call-2", name="read_file", arguments={"path": "b.py"}),
            ],
        )

        self.assertTrue(result.parallel_executed)
        self.assertTrue(result.all_tools_cheap)


if __name__ == "__main__":
    unittest.main()
