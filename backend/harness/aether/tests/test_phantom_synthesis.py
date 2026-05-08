"""Sprint 1.5 / P0-9 — phantom-tool → ToolCall synthesis coverage.

Pins the contract of
:func:`aether.agents.core.phantom_tool.synthesize_tool_calls_from_phantom`
plus the engine-level wire-up that lifts synthesized calls back into
the dispatch path inline.

Three layers covered here:

1. **Pure synthesis** — given a parsed :class:`PhantomToolIntent` and a
   stub registry, every supported syntax (\u0060\u0060\u0060bash, ``<function=NAME>``,
   ``<functions.NAME:N>``, ``<invoke>``) round-trips into a ``ToolCall``
   with the right name + args.
2. **Name resolution** — fuzzy / namespace-aware mapping
   (``execute_command`` → ``shell``, ``Read_File`` → ``read_file``,
   unknown names → ``unresolved``).
3. **Run-loop integration** — the engine actually dispatches the
   synthesized calls, populates ``runtime.phantom_tool_synthesized``,
   and exits with ``TEXT_RESPONSE`` (not ``PHANTOM_TOOL_INTENT``) so
   the UI surfaces the soft "↻ synthesized N call(s)" hint instead
   of the loud "inline tool tags" warning.
"""

from __future__ import annotations

import unittest
from typing import Any, List

from aether import AgentEngine
from aether.agents.core.phantom_tool import (
    PhantomToolIntent,
    SynthesisOutcome,
    detect_phantom_tool_intent,
    synthesize_tool_calls_from_phantom,
)
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.builtins import build_default_tool_registry
from aether.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _SpyShell(ToolExecutor):
    """Records every dispatch so the integration tests can introspect."""

    def __init__(self, name: str = "shell") -> None:
        self._name = name
        self.calls: List[ToolCall] = []

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self._name,
            description="echo",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            required=["command"],
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        self.calls.append(call)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=f"(simulated: {call.arguments.get('command')})",
        )


def _registry_with(*tools: ToolExecutor) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


# ---------------------------------------------------------------------------
# Pure synthesis
# ---------------------------------------------------------------------------


class SynthesisFromIntentTests(unittest.TestCase):
    def test_bash_fence_synthesizes_shell_call(self) -> None:
        intent = PhantomToolIntent(shell_commands=["ls -la"])
        reg = build_default_tool_registry()
        out = synthesize_tool_calls_from_phantom(intent, reg)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(len(out.tool_calls), 1)
        call = out.tool_calls[0]
        self.assertEqual(call.name, "shell")
        self.assertEqual(call.arguments, {"command": "ls -la"})
        self.assertEqual(out.notes[0], "shell(ls -la)")
        self.assertEqual(out.unresolved, [])

    def test_function_eq_tag_with_json_body_round_trips(self) -> None:
        intent = detect_phantom_tool_intent(
            '<function=execute_command>{"command": "date"}</function>'
        )
        reg = build_default_tool_registry()
        out = synthesize_tool_calls_from_phantom(intent, reg)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.tool_calls[0].name, "shell")
        self.assertEqual(out.tool_calls[0].arguments, {"command": "date"})

    def test_functions_slot_tag_round_trips(self) -> None:
        intent = detect_phantom_tool_intent(
            '<functions.shell:0>{"command":"echo hi"}</functions.shell:0>'
        )
        reg = build_default_tool_registry()
        out = synthesize_tool_calls_from_phantom(intent, reg)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.tool_calls[0].name, "shell")
        self.assertEqual(out.tool_calls[0].arguments.get("command"), "echo hi")

    def test_invoke_tag_routes_to_named_tool(self) -> None:
        intent = detect_phantom_tool_intent(
            '<invoke name="read_file">'
            '<parameter name="path">README.md</parameter>'
            "</invoke>"
        )
        reg = build_default_tool_registry()
        out = synthesize_tool_calls_from_phantom(intent, reg)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.tool_calls[0].name, "read_file")
        self.assertEqual(out.tool_calls[0].arguments, {"path": "README.md"})

    def test_multiple_phantoms_yield_multiple_tool_calls(self) -> None:
        intent = detect_phantom_tool_intent(
            "I'll do two things.\n"
            "```bash\nls -la\n```\n"
            "<functions.read_file:1>{\"path\":\"/etc/hosts\"}</functions.read_file:1>"
        )
        reg = build_default_tool_registry()
        out = synthesize_tool_calls_from_phantom(intent, reg)
        self.assertIsNotNone(out)
        assert out is not None
        names = [c.name for c in out.tool_calls]
        self.assertIn("shell", names)
        self.assertIn("read_file", names)

    def test_empty_intent_returns_none(self) -> None:
        out = synthesize_tool_calls_from_phantom(
            PhantomToolIntent(),
            build_default_tool_registry(),
        )
        self.assertIsNone(out)

    def test_unresolved_name_returns_none_when_nothing_synthesized(self) -> None:
        intent = detect_phantom_tool_intent(
            "<function=do_magic>{}</function>"
        )
        out = synthesize_tool_calls_from_phantom(
            intent,
            build_default_tool_registry(),
        )
        self.assertIsNone(out)

    def test_partial_synthesis_returns_outcome_with_unresolved(self) -> None:
        intent = detect_phantom_tool_intent(
            "```bash\nls\n```\n"
            "<function=do_magic>{}</function>"
        )
        out = synthesize_tool_calls_from_phantom(
            intent,
            build_default_tool_registry(),
        )
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(len(out.tool_calls), 1)
        self.assertEqual(out.tool_calls[0].name, "shell")
        self.assertEqual(len(out.unresolved), 1)
        self.assertEqual(out.unresolved[0][0], "do_magic")

    def test_no_shell_tool_registered_skips_bash_fence(self) -> None:
        intent = PhantomToolIntent(shell_commands=["ls -la"])
        reg = ToolRegistry()  # empty
        out = synthesize_tool_calls_from_phantom(intent, reg)
        self.assertIsNone(out)


class FuzzyResolutionTests(unittest.TestCase):
    """The fuzzy resolver is the single defense between "model wrote
    a typo'd tool name" and a phantom turn."""

    def test_case_insensitive_match(self) -> None:
        intent = detect_phantom_tool_intent(
            '<function=Read_File>{"path":"x"}</function>'
        )
        reg = build_default_tool_registry()
        out = synthesize_tool_calls_from_phantom(intent, reg)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.tool_calls[0].name, "read_file")

    def test_namespace_prefixes_are_stripped(self) -> None:
        intent = detect_phantom_tool_intent(
            '<function=mcp__server__shell>{"command":"true"}</function>'
        )
        reg = build_default_tool_registry()
        out = synthesize_tool_calls_from_phantom(intent, reg)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.tool_calls[0].name, "shell")

    def test_aliased_shell_names_route_to_shell(self) -> None:
        for alias in ("execute_command", "run_command", "Bash", "terminal"):
            intent = detect_phantom_tool_intent(
                f'<function={alias}>{{"command":"true"}}</function>'
            )
            reg = build_default_tool_registry()
            out = synthesize_tool_calls_from_phantom(intent, reg)
            self.assertIsNotNone(out, alias)
            assert out is not None
            self.assertEqual(out.tool_calls[0].name, "shell", alias)

    def test_typo_does_not_resolve(self) -> None:
        intent = detect_phantom_tool_intent(
            '<function=read_files>{"path":"x"}</function>'
        )
        reg = build_default_tool_registry()
        out = synthesize_tool_calls_from_phantom(intent, reg)
        # ``read_files`` (with trailing s) is NOT resolved — we'd
        # rather surface the corrective retry path than dispatch the
        # wrong tool.
        self.assertIsNone(out)


# ---------------------------------------------------------------------------
# Run-loop integration
# ---------------------------------------------------------------------------


class SynthesisRunLoopTests(unittest.TestCase):
    def test_synthesis_dispatches_inline_and_loop_continues(self) -> None:
        spy = _SpyShell("shell")
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    content="先看看：```bash\nls -la /tmp\n```",
                    finish_reason="stop",
                ),
                NormalizedResponse(content="ok done.", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=5,
                phantom_tool_synthesis_enabled=True,
            ),
        )

        result = engine.run_loop(
            EngineRequest(session_id="synth-1", user_message="run something")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0].arguments, {"command": "ls -la /tmp"})
        self.assertEqual(result.metadata["runtime"]["phantom_tool_synthesized"], 1)
        self.assertEqual(result.metadata["runtime"]["phantom_tool_retries"], 0)
        notes = result.metadata["turn"].get("phantom_synth_notes")
        self.assertIsInstance(notes, list)
        assert isinstance(notes, list)
        self.assertGreaterEqual(len(notes), 1)
        self.assertIn("shell", notes[0])

    def test_synthesis_disabled_falls_back_to_corrective_retry(self) -> None:
        spy = _SpyShell("shell")
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    content="```bash\nls\n```",
                    finish_reason="stop",
                ),
                NormalizedResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="shell", arguments={"command": "ls"})
                    ],
                    finish_reason="tool_calls",
                ),
                NormalizedResponse(content="done.", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=5,
                phantom_tool_synthesis_enabled=False,
                max_phantom_tool_retries=2,
            ),
        )

        result = engine.run_loop(
            EngineRequest(session_id="synth-2", user_message="run")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(result.metadata["runtime"]["phantom_tool_synthesized"], 0)
        self.assertEqual(result.metadata["runtime"]["phantom_tool_retries"], 1)
        # The model self-corrected with structured tool_calls; spy fired once.
        self.assertEqual(len(spy.calls), 1)

    def test_synthesis_skips_when_no_matching_tool(self) -> None:
        # Empty tool registry except for read_file: bash fence has
        # nowhere to land, so the engine falls through to corrective.
        # We pin this with use_builtin_tools=False + a single
        # registered non-shell tool.
        from aether.tools.builtins.read_file import ReadFileTool

        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    content="```bash\nls\n```",
                    finish_reason="stop",
                ),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(ReadFileTool()),
            config=EngineConfig(
                max_iterations=5,
                use_builtin_tools=False,  # honour the explicit registry
                phantom_tool_synthesis_enabled=True,
                max_phantom_tool_retries=0,
            ),
        )

        result = engine.run_loop(
            EngineRequest(session_id="synth-3", user_message="run")
        )
        # Synthesis couldn't find a shell tool → it returned None →
        # the corrective-retry path took over → with retries=0 the
        # turn exits PHANTOM_TOOL_INTENT.
        self.assertEqual(result.exit_reason, ExitReason.PHANTOM_TOOL_INTENT)
        self.assertEqual(result.metadata["runtime"]["phantom_tool_synthesized"], 0)


if __name__ == "__main__":
    unittest.main()
