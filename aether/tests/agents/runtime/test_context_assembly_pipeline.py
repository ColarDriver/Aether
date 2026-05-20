from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass
from typing import Any

from aether import AgentEngine
from aether.agents.middlewares.base import EngineMiddleware
from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.agents.runtime.context_assembly import (
    ContextAssemblyInput,
    ContextAssemblyPipeline,
)
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.control.interrupts import InterruptController
from aether.runtime.core.contracts import (
    EngineRequest,
    ExitReason,
    NormalizedResponse,
    TurnContext,
)
from aether.runtime.core.hooks import EngineHooks, HookOutcome
from aether.runtime.core.services import EngineServices
from aether.runtime.recovery.strategies import GenericBackoffStrategy
from aether.tools.base import ToolDescriptor
from aether.tools.registry import ToolRegistry


class _Provider(ModelProvider):
    provider_name = "test"
    api_mode = "chat"

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: Any = None,
        stream_silent_callback: Any = None,
    ) -> NormalizedResponse:
        del messages, tools, config, context, stream_callback, stream_silent_callback
        self.calls += 1
        return NormalizedResponse(content="ok")


class _RecordingMiddleware(EngineMiddleware):
    def __init__(self, calls: list[str], *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail

    def before_llm(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        del context
        self.calls.append("middleware_before_llm")
        if self.fail:
            raise RuntimeError("middleware exploded")
        return [*messages, {"role": "user", "content": "middleware"}]


@dataclass(slots=True)
class _Compaction:
    compressed_messages: list[dict[str, Any]]


class _Adapter:
    def __init__(self, calls: list[str], *, compact: bool = False) -> None:
        self.calls = calls
        self.compact = compact

    def maybe_compact_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
        trigger_reason: str,
    ) -> _Compaction | None:
        del context, trigger_reason
        self.calls.append("preflight_compaction")
        if not self.compact:
            return None
        return _Compaction([*messages, {"role": "user", "content": "compacted"}])

    def register_skill_nudge(self, context: TurnContext) -> None:
        del context
        self.calls.append("register_skill_nudge")

    def maybe_inject_skill_nudge(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        del context
        self.calls.append("skill_nudge")
        return messages

    def drain_pending_messages(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        del context
        self.calls.append("pending_messages")
        return messages

    def maybe_inject_diagnostic_attachment(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        del context
        self.calls.append("diagnostics")
        return [*messages, {"role": "user", "content": "<diagnostics>pending</diagnostics>"}]

    def maybe_inject_verifier_reminder(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        del context
        self.calls.append("verifier")
        return messages

    def maybe_inject_plan_mode_attachment(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
        *,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        del context, session_id
        self.calls.append("plan_mode")
        return messages

    def collect_pre_llm_hook_outcome(self, name: str, **kwargs: Any) -> HookOutcome:
        del name, kwargs
        self.calls.append("pre_llm_hook")
        return HookOutcome(inject_user_context="hook-context")

    def consume_messages_override(self, context: TurnContext) -> list[dict[str, Any]] | None:
        self.calls.append("messages_override")
        override = context.metadata.pop("_messages_override", None)
        return override if isinstance(override, list) else None

    def merge_memory_context_into_hook_outcome(
        self,
        messages: list[dict[str, Any]],
        outcome: HookOutcome,
        *,
        context: TurnContext,
    ) -> HookOutcome:
        del messages, context
        self.calls.append("memory_merge")
        return HookOutcome(
            inject_user_context=f"{outcome.inject_user_context}\nmemory-context",
            inject_system_addendum=outcome.inject_system_addendum,
            short_circuit_response=outcome.short_circuit_response,
        )

    def apply_hook_outcome_to_messages(
        self,
        messages: list[dict[str, Any]],
        outcome: HookOutcome,
        *,
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        del context
        self.calls.append("hook_application")
        return [
            *messages,
            {"role": "user", "content": outcome.inject_user_context or ""},
        ]

    def apply_collapse_view(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        del context
        self.calls.append("collapse_view")
        return [*messages, {"role": "user", "content": "collapsed-view"}]


def _services(middleware: MiddlewarePipeline) -> EngineServices:
    provider = _Provider()
    return EngineServices(
        provider=provider,
        tool_registry=ToolRegistry(),
        middleware_pipeline=middleware,
        interrupt_controller=InterruptController(),
        logger=logging.getLogger(__name__),
        recovery_strategy=GenericBackoffStrategy(),
    )


class ContextAssemblyPipelineTests(unittest.TestCase):
    def test_assembly_order_is_stable(self) -> None:
        calls: list[str] = []
        middleware = MiddlewarePipeline([_RecordingMiddleware(calls)])
        pipeline = ContextAssemblyPipeline(
            services=_services(middleware),
            hooks=EngineHooks(),
            adapter=_Adapter(calls),
        )
        context = TurnContext(session_id="ctx", iteration=1, metadata={})

        pipeline.assemble(
            ContextAssemblyInput(
                request=EngineRequest(session_id="ctx"),
                messages=[{"role": "user", "content": "hi"}],
                context=context,
                iteration=1,
            )
        )

        self.assertEqual(
            calls,
            [
                "preflight_compaction",
                "register_skill_nudge",
                "skill_nudge",
                "pending_messages",
                "diagnostics",
                "verifier",
                "plan_mode",
                "pre_llm_hook",
                "messages_override",
                "memory_merge",
                "hook_application",
                "middleware_before_llm",
                "collapse_view",
            ],
        )

    def test_memory_and_collapse_do_not_mutate_canonical_messages(self) -> None:
        calls: list[str] = []
        middleware = MiddlewarePipeline([_RecordingMiddleware(calls)])
        pipeline = ContextAssemblyPipeline(
            services=_services(middleware),
            hooks=EngineHooks(),
            adapter=_Adapter(calls),
        )
        base_messages = [{"role": "user", "content": "hi"}]

        result = pipeline.assemble(
            ContextAssemblyInput(
                request=EngineRequest(session_id="ctx"),
                messages=base_messages,
                context=TurnContext(session_id="ctx", iteration=1, metadata={}),
                iteration=1,
            )
        )

        canonical_text = "\n".join(str(m.get("content", "")) for m in result.canonical_messages)
        prepared_text = "\n".join(str(m.get("content", "")) for m in result.prepared_messages)
        self.assertIn("<diagnostics>", canonical_text)
        self.assertNotIn("memory-context", canonical_text)
        self.assertNotIn("middleware", canonical_text)
        self.assertNotIn("collapsed-view", canonical_text)
        self.assertIn("memory-context", prepared_text)
        self.assertIn("middleware", prepared_text)
        self.assertIn("collapsed-view", prepared_text)

    def test_preflight_compaction_updates_canonical_messages_once(self) -> None:
        calls: list[str] = []
        pipeline = ContextAssemblyPipeline(
            services=_services(MiddlewarePipeline()),
            hooks=EngineHooks(),
            adapter=_Adapter(calls, compact=True),
        )
        context = TurnContext(session_id="ctx", iteration=1, metadata={})

        first = pipeline.assemble(
            ContextAssemblyInput(
                request=EngineRequest(session_id="ctx"),
                messages=[{"role": "user", "content": "hi"}],
                context=context,
                iteration=1,
            )
        )
        second = pipeline.assemble(
            ContextAssemblyInput(
                request=EngineRequest(session_id="ctx"),
                messages=first.canonical_messages,
                context=context,
                iteration=2,
            )
        )

        self.assertEqual(calls.count("preflight_compaction"), 1)
        self.assertTrue(
            any(m.get("content") == "compacted" for m in first.canonical_messages)
        )
        self.assertTrue(
            any(m.get("content") == "compacted" for m in second.canonical_messages)
        )

    def test_messages_override_replaces_canonical_after_hook_observation(self) -> None:
        calls: list[str] = []
        pipeline = ContextAssemblyPipeline(
            services=_services(MiddlewarePipeline()),
            hooks=EngineHooks(),
            adapter=_Adapter(calls),
        )
        context = TurnContext(
            session_id="ctx",
            iteration=1,
            metadata={
                "_messages_override": [
                    {"role": "user", "content": "override"},
                ]
            },
        )

        result = pipeline.assemble(
            ContextAssemblyInput(
                request=EngineRequest(session_id="ctx"),
                messages=[{"role": "user", "content": "original"}],
                context=context,
                iteration=1,
            )
        )

        self.assertEqual(result.canonical_messages[0]["content"], "override")
        self.assertNotIn("_messages_override", context.metadata)

    def test_run_loop_maps_before_llm_exception_to_middleware_error(self) -> None:
        middleware = MiddlewarePipeline([_RecordingMiddleware([], fail=True)])
        engine = AgentEngine(
            _Provider(),
            config=EngineConfig(use_builtin_tools=False),
            middleware_pipeline=middleware,
        )

        result = engine.run_turn(EngineRequest(session_id="ctx-error", user_message="hi"))

        self.assertEqual(result.exit_reason, ExitReason.MIDDLEWARE_ERROR)
        self.assertEqual(provider_calls(engine), 0)


def provider_calls(engine: AgentEngine) -> int:
    provider = engine.services.provider
    return int(getattr(provider, "calls", 0))


if __name__ == "__main__":
    unittest.main()

