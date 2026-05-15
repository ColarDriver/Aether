"""End-to-end pipeline test for diagnostic attachment injection.

Uses a mock LSPManager (returned by :class:`_StubManager`) so the test
never touches a real language server.  A scripted provider records every
LLM call so we can inspect the messages list of the *second* turn —
that's where the ``<diagnostics>`` user-role message must appear.
"""

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
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.diagnostics.tracker import DiagnosticTracker
from aether.runtime.diagnostics.types import Diagnostic
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.builtins.write_file import WriteFileTool
from aether.tools.registry import ToolRegistry


# ---------- scripted provider ----------


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


# ---------- fake LSP manager ----------


def _diag(message: str, *, line: int = 1, column: int = 1) -> Diagnostic:
    return Diagnostic(
        message=message,
        severity="error",
        line=line,
        column=column,
        source="pyright",
    )


class _StubManager:
    def __init__(self, queues: dict[Path, list[list[Diagnostic]]]) -> None:
        self._queues = queues
        self.change_calls: list[tuple[Path, str]] = []
        self.save_calls: list[tuple[Path, str | None]] = []
        self.pull_calls: list[Path] = []

    def change_file(self, path: Path, content: str) -> None:
        self.change_calls.append((Path(path), content))

    def save_file(self, path: Path, *, content: str | None = None) -> None:
        self.save_calls.append((Path(path), content))

    def pull_diagnostics(self, path: Path, *, deadline: float) -> list[Diagnostic]:
        del deadline
        self.pull_calls.append(Path(path))
        queue = self._queues.get(Path(path).resolve())
        if not queue:
            return []
        return queue.pop(0)


# ---------- a tool that reports edited_paths in its result metadata ----------


class _NoopTool(ToolExecutor):
    """Stand-in for a non-edit tool (e.g. ``grep``) — does NOT write
    ``edited_paths`` into its result metadata.  Used to keep the loop
    spinning without injecting fresh edits, so we can prove the tracker
    is not re-surfacing the same diagnostic on the next drain."""

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="noop")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="noop ok",
        )


class _EditingTool(ToolExecutor):
    """Stand-in for ``file_edit``.

    Declares which path it edited via metadata and freezes a tracker
    baseline before returning, so subsequent ``get_new_diagnostics``
    calls see only what changed afterwards.
    """

    def __init__(self, edited_path: Path, tracker: DiagnosticTracker | None) -> None:
        self._path = Path(edited_path).resolve()
        self._tracker = tracker

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="fake_edit")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        if self._tracker is not None:
            self._tracker.before_file_edited(self._path)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="ok",
            metadata={"edited_paths": [str(self._path)]},
        )


# ---------- helpers ----------


def _make_engine(
    *,
    tool: ToolExecutor,
    provider: _RecordingProvider,
    tracker: DiagnosticTracker | None,
    extra_tools: list[ToolExecutor] | None = None,
) -> AgentEngine:
    registry = ToolRegistry()
    registry.register(tool)
    for extra in extra_tools or []:
        registry.register(extra)
    return AgentEngine(
        provider,
        tool_registry=registry,
        diagnostic_tracker=tracker,
        config=EngineConfig(
            use_builtin_tools=False,
            verification_directive_enabled=False,
            faithful_reporting_enabled=False,
            max_iterations=4,
        ),
    )


def _find_diagnostics_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and "<diagnostics>" in content:
            return msg
    return None


def _count_diagnostic_blocks(messages: list[dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            count += content.count("<diagnostics>")
    return count


# ---------- tests ----------


class DiagnosticAttachmentPipelineTests(unittest.TestCase):
    edited_path: Path = Path("/tmp/aether-pipeline-test.py")

    def setUp(self) -> None:
        self.edited_path = Path("/tmp/aether-pipeline-test.py").resolve()

    def test_edit_then_next_turn_messages_contain_diagnostics(self) -> None:
        bad = _diag("undefined name 'foo'", line=2, column=4)
        manager = _StubManager(
            {
                self.edited_path: [
                    [],         # baseline before edit
                    [bad],      # after edit — drain on turn 2
                ]
            }
        )
        tracker = DiagnosticTracker(manager, settle_timeout_ms=20)
        provider = _RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="fake_edit", arguments={})],
                ),
                NormalizedResponse(content="all done"),
            ]
        )
        engine = _make_engine(
            tool=_EditingTool(self.edited_path, tracker),
            provider=provider,
            tracker=tracker,
        )

        engine.run_turn(EngineRequest(session_id="diag-1", user_message="go"))

        self.assertEqual(len(provider.calls), 2)
        # Turn 1 should NOT carry a diagnostics block.
        turn1 = provider.calls[0]["messages"]
        self.assertIsNone(_find_diagnostics_message(turn1))
        # Turn 2 must carry it, scoped to the edited path.
        turn2 = provider.calls[1]["messages"]
        diag_msg = _find_diagnostics_message(turn2)
        self.assertIsNotNone(diag_msg)
        assert diag_msg is not None
        self.assertEqual(diag_msg["role"], "user")
        self.assertEqual(diag_msg["metadata"]["source"], "diagnostics")
        self.assertIn("undefined name 'foo'", diag_msg["content"])
        self.assertIn(str(self.edited_path), diag_msg["content"])

    def test_real_write_file_tool_produces_next_turn_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            edited_path = (Path(tmp) / "real.py").resolve()
            bad = _diag("undefined name 'foo'", line=2, column=4)
            manager = _StubManager(
                {
                    edited_path: [
                        [],      # baseline before write
                        [bad],   # after internal didChange/didSave
                    ]
                }
            )
            tracker = DiagnosticTracker(manager, settle_timeout_ms=20)
            provider = _RecordingProvider(
                [
                    NormalizedResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "real.py", "content": "x = foo\n"},
                            )
                        ],
                    ),
                    NormalizedResponse(content="all done"),
                ]
            )
            engine = _make_engine(
                tool=WriteFileTool(default_cwd=Path(tmp)),
                provider=provider,
                tracker=tracker,
            )

            engine.run_turn(EngineRequest(session_id="diag-real-write", user_message="go"))

        self.assertEqual(len(provider.calls), 2)
        turn2 = provider.calls[1]["messages"]
        diag_msg = _find_diagnostics_message(turn2)
        self.assertIsNotNone(diag_msg)
        assert diag_msg is not None
        self.assertIn("undefined name 'foo'", diag_msg["content"])
        self.assertIn(str(edited_path), diag_msg["content"])

    def test_no_attachment_when_tracker_disabled(self) -> None:
        provider = _RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="fake_edit", arguments={})],
                ),
                NormalizedResponse(content="done"),
            ]
        )
        engine = _make_engine(
            tool=_EditingTool(self.edited_path, tracker=None),
            provider=provider,
            tracker=None,
        )
        engine.run_turn(EngineRequest(session_id="diag-2", user_message="go"))

        for turn in provider.calls:
            self.assertIsNone(_find_diagnostics_message(turn["messages"]))

    def test_no_attachment_when_lsp_returns_no_new_diagnostics(self) -> None:
        manager = _StubManager(
            {
                self.edited_path: [
                    [],   # baseline
                    [],   # after edit — still clean
                ]
            }
        )
        tracker = DiagnosticTracker(manager, settle_timeout_ms=20)
        provider = _RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="fake_edit", arguments={})],
                ),
                NormalizedResponse(content="done"),
            ]
        )
        engine = _make_engine(
            tool=_EditingTool(self.edited_path, tracker),
            provider=provider,
            tracker=tracker,
        )
        engine.run_turn(EngineRequest(session_id="diag-3", user_message="go"))

        for turn in provider.calls:
            self.assertIsNone(_find_diagnostics_message(turn["messages"]))

    def test_diagnostic_delivered_once(self) -> None:
        """Tracker dedups across turns: the same error must not be
        re-surfaced once the model has already seen it."""
        bad = _diag("ImportError: foo")
        # Three pull responses: baseline (during turn 1 dispatch),
        # bad on turn 2 PRE_LLM drain, bad still present on turn 3
        # PRE_LLM drain (must be filtered by the delivered set).
        manager = _StubManager(
            {
                self.edited_path: [
                    [],
                    [bad],
                    [bad],
                ]
            }
        )
        tracker = DiagnosticTracker(manager, settle_timeout_ms=20)
        # Turn 1 edits; turn 2 runs a non-edit tool (so ``_last_edited_paths``
        # stays empty and the producer falls back to the all-baselines
        # drain); turn 3 produces text and exits.
        provider = _RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="fake_edit", arguments={})],
                ),
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c2", name="noop", arguments={})],
                ),
                NormalizedResponse(content="done"),
            ]
        )
        engine = _make_engine(
            tool=_EditingTool(self.edited_path, tracker),
            extra_tools=[_NoopTool()],
            provider=provider,
            tracker=tracker,
        )
        engine.run_turn(EngineRequest(session_id="diag-4", user_message="go"))

        self.assertEqual(len(provider.calls), 3)
        # Turn 1: nothing edited yet → no block.
        self.assertEqual(_count_diagnostic_blocks(provider.calls[0]["messages"]), 0)
        # Turn 2: drain delivers the diagnostic exactly once.
        self.assertEqual(_count_diagnostic_blocks(provider.calls[1]["messages"]), 1)
        # Turn 3: the LSP still reports ``bad``, but the tracker's
        # delivered-set filters it; the engine must NOT inject a
        # second block.  The first block persists in history (that's
        # how user-role messages work in this engine), so the count
        # stays at 1 — not 2.
        self.assertEqual(_count_diagnostic_blocks(provider.calls[2]["messages"]), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
