"""Tests for AgentEngine's internal post-tool diagnostic dispatch."""

from __future__ import annotations

import copy
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
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.core.hooks import EngineHooks
from aether.runtime.diagnostics.tracker import DiagnosticTracker
from aether.runtime.diagnostics.types import Diagnostic
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.builtins.write_file import WriteFileTool
from aether.tools.registry import ToolRegistry


class _RecordingProvider(ModelProvider):
    provider_name: str = "test-provider"
    api_mode: str = "chat"

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self.model: str = "test-model"
        self.responses: list[NormalizedResponse] = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del tools, config, context, stream_callback, stream_silent_callback
        self.calls.append({"messages": copy.deepcopy(messages)})
        if not self.responses:
            raise RuntimeError("no scripted response left")
        return self.responses.pop(0)


class _StubManager:
    def __init__(self) -> None:
        self.pull_calls: list[Path] = []
        self.change_calls: list[tuple[Path, str]] = []
        self.save_calls: list[tuple[Path, str | None]] = []

    def change_file(self, path: Path, content: str) -> None:
        self.change_calls.append((Path(path).resolve(), content))

    def save_file(self, path: Path, *, content: str | None = None) -> None:
        self.save_calls.append((Path(path).resolve(), content))

    def pull_diagnostics(self, path: Path, *, deadline: float) -> list[Diagnostic]:
        del deadline
        self.pull_calls.append(Path(path).resolve())
        return []


class _OrderingHooks(EngineHooks):
    __slots__ = ("_manager", "seen_change_counts")

    def __init__(self, manager: _StubManager) -> None:
        super().__init__()
        self._manager = manager
        self.seen_change_counts: list[int] = []

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
        del session_id, iteration, tool_call_id, tool_name, tool_args, result, elapsed_ms, context_metadata
        self.seen_change_counts.append(len(self._manager.change_calls))


class _NoMetadataTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="noop")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del call, context
        return ToolResult(tool_call_id="noop", name="noop", content="ok")


class _MissingPathTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="missing-path")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del call, context
        return ToolResult(
            tool_call_id="missing-path",
            name="missing-path",
            content="ok",
            metadata={"edited_paths": ["/tmp/definitely-not-here-aether.py"]},
        )


def _build_engine(
    *,
    tool: ToolExecutor,
    provider: _RecordingProvider,
    tracker: DiagnosticTracker | None,
    hooks: EngineHooks | None = None,
) -> AgentEngine:
    registry = ToolRegistry()
    registry.register(tool)
    return AgentEngine(
        provider,
        tool_registry=registry,
        diagnostic_tracker=tracker,
        hooks=hooks,
        config=EngineConfig(
            use_builtin_tools=False,
            verification_directive_enabled=False,
            faithful_reporting_enabled=False,
            verifier_gate_enabled=False,
            max_iterations=4,
        ),
    )


class InternalDiagnosticHookTests(unittest.TestCase):
    def test_real_write_file_dispatches_after_user_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = _StubManager()
            tracker = DiagnosticTracker(manager, settle_timeout_ms=20)
            hooks = _OrderingHooks(manager)
            provider = _RecordingProvider(
                [
                    NormalizedResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "out.py", "content": "x = 1\n"},
                            )
                        ],
                    ),
                    NormalizedResponse(content="done"),
                ]
            )
            engine = _build_engine(
                tool=WriteFileTool(default_cwd=Path(tmp)),
                provider=provider,
                tracker=tracker,
                hooks=hooks,
            )

            engine.run_turn(EngineRequest(session_id="diag-hook-1", user_message="go"))
            written = (Path(tmp) / "out.py").resolve()

        self.assertEqual(hooks.seen_change_counts, [0])
        self.assertEqual(manager.change_calls, [(written, "x = 1\n")])
        self.assertEqual(manager.save_calls, [(written, "x = 1\n")])

    def test_dispatch_skips_when_result_has_no_edited_paths(self) -> None:
        manager = _StubManager()
        tracker = DiagnosticTracker(manager, settle_timeout_ms=20)
        provider = _RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
                ),
                NormalizedResponse(content="done"),
            ]
        )
        engine = _build_engine(
            tool=_NoMetadataTool(),
            provider=provider,
            tracker=tracker,
        )

        engine.run_turn(EngineRequest(session_id="diag-hook-2", user_message="go"))

        self.assertEqual(manager.change_calls, [])
        self.assertEqual(manager.save_calls, [])

    def test_missing_edited_path_does_not_break_loop(self) -> None:
        manager = _StubManager()
        tracker = DiagnosticTracker(manager, settle_timeout_ms=20)
        provider = _RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="missing-path", arguments={})],
                ),
                NormalizedResponse(content="done"),
            ]
        )
        engine = _build_engine(
            tool=_MissingPathTool(),
            provider=provider,
            tracker=tracker,
        )

        result = engine.run_turn(EngineRequest(session_id="diag-hook-3", user_message="go"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(manager.change_calls, [])
        self.assertEqual(manager.save_calls, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
