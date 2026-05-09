"""Sprint 1.5 — phantom-tool intent recovery.

Pins down the run-loop branch that catches "model wrote a tool
invocation in prose instead of populating ``tool_calls``" and either
nudges the model to retry or surfaces ``ExitReason.PHANTOM_TOOL_INTENT``
when the retry budget is exhausted.

Background:
``aether.agents.core.phantom_tool`` owns the detection regex (shared
with the CLI's diagnostic) and the corrective ``role=user`` message.
``AgentEngine._maybe_recover_phantom_tool_intent`` is the
decision-routine the run-loop calls in the no-tool-calls branch.

Five behaviours covered here:

1. **Pure detection** — the parser surfaces every fenced shell block,
   ``<function=NAME>`` inline tag, and ``<invoke>`` XML payload in
   source order, with sensible empty defaults for whitespace bodies.
2. **Single-attempt recovery** — phantom intent on iteration 1, the
   model self-corrects with structured ``tool_calls`` on iteration 2,
   the tool actually runs, and the turn finalises ``COMPLETED``.
3. **Multi-attempt recovery + counter** — phantom intent on every
   iteration up to ``max_phantom_tool_retries``, then the engine
   exits with ``PHANTOM_TOOL_INTENT`` and the turn metadata exposes
   the parsed attempts.
4. **Retry-budget guard** — ``max_phantom_tool_retries=0`` means the
   detector still surfaces the diagnostic but the loop exits
   immediately with ``PHANTOM_TOOL_INTENT``.  No infinite loops.
5. **Disabled mode** — ``phantom_tool_recovery_enabled=False`` falls
   straight through to ``TEXT_RESPONSE``, preserving today's
   behaviour for non-Kimi-class models that always populate
   ``tool_calls`` correctly.

The provider responses are scripted so the test pins down exactly
what the run-loop does with each shape, without depending on
floating model behaviour.
"""

from __future__ import annotations

import unittest
from typing import Any, List

from aether import AgentEngine
from aether.agents.core.phantom_tool import (
    PhantomToolIntent,
    build_corrective_user_message,
    detect_phantom_tool_intent,
)
from aether.config.schema import EngineConfig
from aether.models.provider.base import ModelProvider
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    StreamDeltaCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.config.schema import ModelCallConfig
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _RecordingProvider(ModelProvider):
    """Provider that emits scripted responses and records each call.

    Mirrors the helper in ``test_run_loop_truncated_tool_call.py`` —
    each test owns its own instance, so we duplicate the few-liner
    rather than reach across test files.
    """

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: Any = None,  # noqa: ARG002
    ) -> NormalizedResponse:
        if not self._responses:
            raise RuntimeError("RecordingProvider exhausted")
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "max_tokens": config.max_tokens,
            }
        )
        return self._responses.pop(0)


class _SpyShellTool(ToolExecutor):
    """A bash/shell-shaped tool that records every dispatch."""

    def __init__(self, name: str = "execute_command") -> None:
        self._name = name
        self.calls: list[ToolCall] = []

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name=self._name)

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        self.calls.append(call)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="(simulated stdout)",
        )


def _registry_with(*tools: ToolExecutor) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


# ---------------------------------------------------------------------------
# Pure detection unit
# ---------------------------------------------------------------------------


class DetectPhantomToolIntentTests(unittest.TestCase):
    """The regex-driven extraction is the foundation everything else
    hinges on; broken extraction → broken corrective messages →
    broken recovery.  These tests pin the corner cases that bit us
    when the user's terminal showed multiple ``\u0060\u0060\u0060bash`` blocks
    side-by-side with ``<function=NAME>`` tags."""

    def test_returns_empty_for_plain_prose(self) -> None:
        intent = detect_phantom_tool_intent("Hello! Nice to meet you.")
        self.assertTrue(intent.is_empty())
        self.assertEqual(intent.raw_intents_count, 0)

    def test_extracts_single_fenced_shell_block(self) -> None:
        intent = detect_phantom_tool_intent(
            "我来帮你看看这个项目是做什么的。让我先浏览一下项目结构和关键文件。"
            "```bash ls -la /workspace/Aether"
        )
        self.assertEqual(len(intent.shell_commands), 1)
        self.assertEqual(intent.shell_commands[0], "ls -la /workspace/Aether")
        self.assertEqual(intent.invoke_calls, [])

    def test_extracts_multiple_fenced_shell_blocks(self) -> None:
        intent = detect_phantom_tool_intent(
            "我先看看：```bash cd /workspace && ls -la\n\n"
            "实际上让我用更系统的方式查看：```bash\n"
            "cat /workspace/README.md 2>/dev/null || echo 'no readme'\n"
            "```\n"
            "让我执行这些命令。"
        )
        self.assertEqual(len(intent.shell_commands), 2)
        self.assertIn("cd /workspace && ls -la", intent.shell_commands[0])
        self.assertIn("cat /workspace/README.md", intent.shell_commands[1])

    def test_extracts_function_equals_inline_tag(self) -> None:
        intent = detect_phantom_tool_intent(
            "我来看看。<function=execute_command> ls -la /workspace/Aether/  "
            "<function=execute_command> cat /workspace/Aether/README.md"
        )
        self.assertEqual(len(intent.invoke_calls), 2)
        self.assertEqual(intent.invoke_calls[0][0], "execute_command")
        self.assertEqual(
            intent.invoke_calls[0][1].get("command"),
            "ls -la /workspace/Aether/",
        )

    def test_extracts_anthropic_invoke_xml(self) -> None:
        intent = detect_phantom_tool_intent(
            "Looking up.\n"
            '<function_calls><invoke name="read_file">'
            '<parameter name="path">README.md</parameter>'
            "</invoke></function_calls>"
        )
        self.assertEqual(len(intent.invoke_calls), 1)
        name, args = intent.invoke_calls[0]
        self.assertEqual(name, "read_file")
        self.assertEqual(args.get("path"), "README.md")

    def test_extracts_openai_tool_call_json(self) -> None:
        intent = detect_phantom_tool_intent(
            'Trying. <tool_call>{"name":"shell","arguments":{"cmd":"ls"}}</tool_call>'
        )
        self.assertEqual(len(intent.invoke_calls), 1)
        name, args = intent.invoke_calls[0]
        self.assertEqual(name, "shell")
        self.assertEqual(args.get("cmd"), "ls")

    def test_mixed_syntaxes_are_collected_into_one_intent(self) -> None:
        intent = detect_phantom_tool_intent(
            "I'll run two things.\n"
            "```bash\nls -la\n```\n"
            "<function=read_file> /workspace/README.md"
        )
        self.assertEqual(len(intent.shell_commands), 1)
        self.assertEqual(len(intent.invoke_calls), 1)
        self.assertEqual(intent.raw_intents_count, 2)

    def test_empty_or_whitespace_body_is_not_counted(self) -> None:
        intent = detect_phantom_tool_intent("Trying.\n```bash\n   \n```\nLet me retry.")
        self.assertTrue(intent.is_empty())


class CorrectiveMessageTests(unittest.TestCase):
    def test_corrective_message_is_role_user_with_attempt_index(self) -> None:
        intent = PhantomToolIntent(shell_commands=["ls -la"], invoke_calls=[])
        msg = build_corrective_user_message(intent, attempt_index=1)
        self.assertEqual(msg["role"], "user")
        self.assertIn("attempt 1", msg["content"])

    def test_corrective_message_lists_every_attempted_command(self) -> None:
        intent = PhantomToolIntent(
            shell_commands=["ls -la /workspace", "cat README.md"],
            invoke_calls=[("read_file", {"path": "/etc/passwd"})],
        )
        msg = build_corrective_user_message(intent, attempt_index=2)
        self.assertIn("ls -la /workspace", msg["content"])
        self.assertIn("cat README.md", msg["content"])
        self.assertIn("read_file", msg["content"])
        self.assertIn("path=/etc/passwd", msg["content"])

    def test_corrective_message_truncates_runaway_attempts(self) -> None:
        intent = PhantomToolIntent(
            shell_commands=[f"echo {i}" for i in range(20)],
            invoke_calls=[],
        )
        msg = build_corrective_user_message(intent, attempt_index=1)
        # Limit is 6, plus a "+ N more" line.
        self.assertIn("+14 more", msg["content"])


# ---------------------------------------------------------------------------
# Run-loop integration
# ---------------------------------------------------------------------------


class PhantomToolRecoveryRunLoopTests(unittest.TestCase):
    """Engine-level scenarios.

    Each test wires a ``ScriptedProvider`` so the model's behaviour is
    deterministic.  The provider returns a phantom-tool prose
    response on iteration 1 and the run-loop must either retry
    (sending a corrective message) or finalise PHANTOM_TOOL_INTENT,
    depending on the retry budget.
    """

    def test_single_attempt_self_corrects_and_dispatches_tool(self) -> None:
        spy = _SpyShellTool("execute_command")
        provider = _RecordingProvider(
            [
                # Iteration 1 — model writes the command in prose.
                NormalizedResponse(
                    content=(
                        "我来看看这个项目的结构。让我先浏览一下。"
                        "```bash ls -la /workspace/Aether"
                    ),
                    tool_calls=[],
                    finish_reason="stop",
                ),
                # Iteration 2 — model self-corrects with proper tool_calls.
                NormalizedResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="execute_command",
                            arguments={"command": "ls -la /workspace/Aether"},
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                # Iteration 3 — final answer once tool result is in.
                NormalizedResponse(
                    content="这个项目是 Aether harness 的实现。",
                    finish_reason="stop",
                ),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=6,
                max_phantom_tool_retries=2,
                # Force the corrective-message retry path so this
                # test pins down the *nudge → self-correct → real
                # tool_call* loop.  The (newer) synthesis branch has
                # its own dedicated tests in
                # ``test_phantom_synthesis.py``.
                phantom_tool_synthesis_enabled=False,
            ),
        )

        result = engine.run_turn(
            EngineRequest(session_id="phantom-1", user_message="项目是什么？")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(result.iterations, 3)
        # Tool was actually dispatched after the corrective nudge.
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0].name, "execute_command")
        # Exactly one corrective retry burned.
        self.assertEqual(result.metadata["runtime"]["phantom_tool_retries"], 1)
        # The corrective ``role=user`` message must have been sent on
        # iteration 2 — verify by inspecting the provider's recorded
        # transcript.
        iteration_2_messages = provider.calls[1]["messages"]
        last_user = next(
            (m for m in reversed(iteration_2_messages) if m["role"] == "user"), None
        )
        self.assertIsNotNone(last_user)
        assert last_user is not None
        self.assertIn("tool_calls", last_user["content"])
        self.assertIn("ls -la /workspace/Aether", last_user["content"])

    def test_retry_budget_exhaustion_exits_with_phantom_tool_intent(self) -> None:
        spy = _SpyShellTool()
        provider = ScriptedProvider(
            [
                # Three consecutive phantom-tool responses; the engine
                # only has 2 retries so iteration 3 finalises.
                NormalizedResponse(
                    content="先看看 ```bash ls /workspace",
                    finish_reason="stop",
                ),
                NormalizedResponse(
                    content="再看 ```bash ls /tmp",
                    finish_reason="stop",
                ),
                NormalizedResponse(
                    content="最后试试 ```bash ls /etc",
                    finish_reason="stop",
                ),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=6,
                max_phantom_tool_retries=2,
                # Disabled so the test pins the corrective-message
                # exhaustion path even though the spy tool is shell-
                # aliased and the synthesizer would otherwise rescue
                # the prose into structured calls.
                phantom_tool_synthesis_enabled=False,
            ),
        )

        result = engine.run_turn(
            EngineRequest(session_id="phantom-2", user_message="run something")
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.PHANTOM_TOOL_INTENT)
        # Tool was never dispatched — the model never produced a
        # structured tool_call, so by design no work happened.
        self.assertEqual(spy.calls, [])
        # Counter must reflect that we burned every available retry.
        self.assertEqual(result.metadata["runtime"]["phantom_tool_retries"], 2)
        # The parsed attempts must be exposed for the UI footer to
        # show "model tried these but never invoked them".
        attempts = result.metadata["turn"].get("phantom_tool_attempts")
        self.assertIsNotNone(attempts)
        assert isinstance(attempts, dict)
        self.assertGreaterEqual(attempts["raw_intents_count"], 1)

    def test_zero_retry_budget_exits_phantom_tool_intent_immediately(self) -> None:
        spy = _SpyShellTool()
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    content="我来跑一下 ```bash ls /workspace",
                    finish_reason="stop",
                ),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=6,
                max_phantom_tool_retries=0,
                # See sibling tests: synthesis would side-step the
                # zero-retry-budget guard, so disable to keep this
                # specific test deterministic.
                phantom_tool_synthesis_enabled=False,
            ),
        )

        result = engine.run_turn(
            EngineRequest(session_id="phantom-3", user_message="ls")
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.PHANTOM_TOOL_INTENT)
        self.assertEqual(spy.calls, [])
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.metadata["runtime"]["phantom_tool_retries"], 0)

    def test_disabled_mode_falls_through_to_text_response(self) -> None:
        spy = _SpyShellTool()
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    content="我来跑一下 ```bash ls /workspace",
                    finish_reason="stop",
                ),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=6,
                phantom_tool_recovery_enabled=False,
            ),
        )

        result = engine.run_turn(
            EngineRequest(session_id="phantom-4", user_message="ls")
        )

        # Old behaviour: prose response is treated as a normal text
        # answer; status COMPLETED, exit TEXT_RESPONSE.  Useful for
        # well-behaved providers that don't need this recovery.
        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(result.metadata["runtime"]["phantom_tool_retries"], 0)
        # The detector ran but the recovery branch was bypassed —
        # ``phantom_tool_attempts`` should NOT be set.
        self.assertNotIn(
            "phantom_tool_attempts", result.metadata["turn"]
        )

    def test_normal_text_response_does_not_trigger_recovery(self) -> None:
        """A polite greeting must never be flagged as phantom intent."""
        spy = _SpyShellTool()
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    content="你好！很高兴见到你。",
                    finish_reason="stop",
                ),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(max_iterations=6),
        )

        result = engine.run_turn(
            EngineRequest(session_id="phantom-5", user_message="你好")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(result.metadata["runtime"]["phantom_tool_retries"], 0)

    # ------------------------------------------------------------------
    # Sprint 1.5 / P0-9 — synthesis preserves the loop instead of
    # bouncing through corrective retries.  See ``test_phantom_synthesis``
    # for the full coverage; the tests below pin the run-loop-level
    # transitions only (counters, exit reason, metadata flow).
    # ------------------------------------------------------------------

    def test_synthesis_preserves_loop_with_zero_retries_burned(self) -> None:
        spy = _SpyShellTool("shell")
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
                max_iterations=6,
                phantom_tool_synthesis_enabled=True,
            ),
        )

        result = engine.run_turn(
            EngineRequest(session_id="phantom-synth-1", user_message="跑个命令")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        # Synthesis took the place of a corrective retry: counter is 0.
        self.assertEqual(result.metadata["runtime"]["phantom_tool_retries"], 0)
        self.assertEqual(result.metadata["runtime"]["phantom_tool_synthesized"], 1)
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0].arguments, {"command": "ls -la /tmp"})

    def test_synthesis_metadata_includes_per_call_notes(self) -> None:
        spy = _SpyShellTool("shell")
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    content="```bash\necho hi\n```",
                    finish_reason="stop",
                ),
                NormalizedResponse(content="ok.", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(max_iterations=6),
        )

        result = engine.run_turn(
            EngineRequest(session_id="phantom-synth-2", user_message="hi")
        )

        notes = result.metadata["turn"].get("phantom_synth_notes")
        self.assertIsInstance(notes, list)
        assert isinstance(notes, list)
        self.assertEqual(len(notes), 1)
        self.assertIn("shell", notes[0])
        self.assertIn("echo hi", notes[0])


if __name__ == "__main__":
    unittest.main()
