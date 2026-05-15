"""Tests for the PR 10.3 builder extensions: META_TOOLS injection,
model override, max_turns clamp, and skills hint injection."""

from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.agents.types import AgentTypeDefinition, AgentTypeRegistry
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.subagents.contracts import SubagentTask
from aether.subagents.default_builder import (
    META_TOOLS_DEFAULT_ALLOWED,
    DefaultSubagentBuilder,
    _filter_tool_registry,
    _resolve_model_alias,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _DummyTool(ToolExecutor):
    def __init__(self, name: str) -> None:
        self._descriptor = ToolDescriptor(name=name)

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        return ToolResult(tool_call_id=call.id, name=call.name, content="ok")


def _registry(*names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register(_DummyTool(name))
    return registry


def _build_parent(tool_registry: ToolRegistry) -> AgentEngine:
    return AgentEngine(
        ScriptedProvider([NormalizedResponse(content="parent")]),
        tool_registry=tool_registry,
        config=EngineConfig(use_builtin_tools=False),
        agent_type_registry=AgentTypeRegistry(search_paths=[]),
    )


def _typed_task(definition: AgentTypeDefinition, **metadata) -> SubagentTask:
    return SubagentTask(
        task_id="t-1",
        goal="g",
        request=EngineRequest(session_id="s", user_message="u"),
        metadata={"_agent_type_def": definition, **metadata},
    )


# ----------------------------------------------------------------- META TOOLS

class MetaToolFilteringTests(unittest.TestCase):
    def test_meta_tools_injected_into_implicit_allowlist(self) -> None:
        # Type only whitelists ``read_file``; meta tools should still
        # survive (so the child can task_stop / load skills / etc).
        source = _registry("read_file", "shell", "task_stop", "skill", "send_message")
        filtered = _filter_tool_registry(
            source,
            allowed_tools=("read_file",),
            disallowed_tools=(),
        )
        self.assertTrue(filtered.has("read_file"))
        self.assertTrue(filtered.has("task_stop"))
        self.assertTrue(filtered.has("skill"))
        self.assertTrue(filtered.has("send_message"))
        self.assertFalse(filtered.has("shell"))  # not in allowlist, not META

    def test_meta_tool_explicitly_denied_is_removed(self) -> None:
        # Explicit deny wins even for META tools — Verifier uses this
        # to forbid send_message ("verdict IS the message").
        source = _registry("read_file", "send_message", "skill")
        filtered = _filter_tool_registry(
            source,
            allowed_tools=("read_file",),
            disallowed_tools=("send_message",),
        )
        self.assertTrue(filtered.has("read_file"))
        self.assertTrue(filtered.has("skill"))           # META, not denied
        self.assertFalse(filtered.has("send_message"))   # META, but denied

    def test_no_allowlist_keeps_everything_minus_deny(self) -> None:
        source = _registry("read_file", "shell", "task")
        filtered = _filter_tool_registry(
            source,
            allowed_tools=None,
            disallowed_tools=("shell",),
        )
        self.assertTrue(filtered.has("read_file"))
        self.assertTrue(filtered.has("task"))
        self.assertFalse(filtered.has("shell"))

    def test_meta_tools_only_added_if_present_in_source(self) -> None:
        # If the parent never registered ``send_message``, the filter
        # must not synthesise it — only inject what already exists.
        source = _registry("read_file")
        filtered = _filter_tool_registry(
            source,
            allowed_tools=("read_file",),
            disallowed_tools=(),
        )
        for meta_name in META_TOOLS_DEFAULT_ALLOWED - {"read_file"}:
            self.assertFalse(filtered.has(meta_name))


# ----------------------------------------------------------------- MODEL

class ModelAliasTests(unittest.TestCase):
    def test_known_aliases_resolve_to_concrete_model_ids(self) -> None:
        self.assertEqual(_resolve_model_alias("sonnet"), "claude-sonnet-4-6")
        self.assertEqual(_resolve_model_alias("opus"), "claude-opus-4-7")
        self.assertEqual(_resolve_model_alias("haiku"), "claude-haiku-4-5-20251001")

    def test_alias_lookup_is_case_insensitive(self) -> None:
        self.assertEqual(_resolve_model_alias("SONNET"), "claude-sonnet-4-6")
        self.assertEqual(_resolve_model_alias(" Opus "), "claude-opus-4-7")

    def test_unknown_alias_passes_through_trimmed(self) -> None:
        self.assertEqual(
            _resolve_model_alias(" claude-sonnet-4-7 "), "claude-sonnet-4-7"
        )


class ModelOverrideTests(unittest.TestCase):
    def _definition(self, model: str | None = None) -> AgentTypeDefinition:
        return AgentTypeDefinition(
            agent_type="custom",
            description="x",
            tools=("read_file",),
            model=model,
        )

    def test_definition_model_writes_to_extra(self) -> None:
        parent = _build_parent(_registry("read_file"))
        task = _typed_task(self._definition(model="sonnet"))
        DefaultSubagentBuilder().build_child(parent, task, child_depth=1)
        self.assertEqual(
            task.request.model_config.extra.get("model"),
            "claude-sonnet-4-6",
        )

    def test_caller_model_override_wins(self) -> None:
        parent = _build_parent(_registry("read_file"))
        task = _typed_task(self._definition(model="sonnet"), model_override="haiku")
        DefaultSubagentBuilder().build_child(parent, task, child_depth=1)
        self.assertEqual(
            task.request.model_config.extra.get("model"),
            "claude-haiku-4-5-20251001",
        )

    def test_no_override_keeps_extra_clean(self) -> None:
        parent = _build_parent(_registry("read_file"))
        task = _typed_task(self._definition(model=None))
        DefaultSubagentBuilder().build_child(parent, task, child_depth=1)
        self.assertNotIn("model", task.request.model_config.extra)

    def test_inherit_sentinel_falls_through_to_definition(self) -> None:
        parent = _build_parent(_registry("read_file"))
        task = _typed_task(self._definition(model="opus"), model_override="inherit")
        DefaultSubagentBuilder().build_child(parent, task, child_depth=1)
        self.assertEqual(
            task.request.model_config.extra.get("model"),
            "claude-opus-4-7",
        )


# ----------------------------------------------------------------- MAX TURNS

class MaxTurnsClampTests(unittest.TestCase):
    def _build_child(self, definition: AgentTypeDefinition):
        parent = _build_parent(_registry("read_file"))
        task = _typed_task(definition)
        return DefaultSubagentBuilder().build_child(parent, task, child_depth=1)

    def test_max_turns_clamps_below_parent(self) -> None:
        defn = AgentTypeDefinition(
            agent_type="capped",
            description="x",
            tools=("read_file",),
            max_turns=5,
        )
        child = self._build_child(defn)
        self.assertEqual(child.config.max_iterations, 5)

    def test_max_turns_does_not_raise_above_parent(self) -> None:
        defn = AgentTypeDefinition(
            agent_type="generous",
            description="x",
            tools=("read_file",),
            max_turns=999,
        )
        child = self._build_child(defn)
        # Parent default is 32 (EngineConfig default), 999 > 32 → keep 32.
        self.assertEqual(child.config.max_iterations, 32)

    def test_no_max_turns_inherits_parent(self) -> None:
        defn = AgentTypeDefinition(
            agent_type="open",
            description="x",
            tools=("read_file",),
            max_turns=None,
        )
        child = self._build_child(defn)
        self.assertEqual(child.config.max_iterations, 32)

    def test_zero_max_turns_treated_as_unset(self) -> None:
        defn = AgentTypeDefinition(
            agent_type="zero",
            description="x",
            tools=("read_file",),
            max_turns=0,
        )
        child = self._build_child(defn)
        # 0 is invalid (would mean "no iterations") — fall through to parent.
        self.assertEqual(child.config.max_iterations, 32)


# ----------------------------------------------------------------- SKILLS HINT

class SkillsHintTests(unittest.TestCase):
    def _build(self, skills: tuple[str, ...]) -> SubagentTask:
        defn = AgentTypeDefinition(
            agent_type="hinted",
            description="x",
            tools=("read_file",),
            skills=skills,
        )
        parent = _build_parent(_registry("read_file"))
        task = _typed_task(defn)
        DefaultSubagentBuilder().build_child(parent, task, child_depth=1)
        return task

    def test_hint_appended_when_skills_set(self) -> None:
        task = self._build(skills=("tdd", "security-review"))
        sm = task.request.system_message or ""
        self.assertIn("<system-reminder>", sm)
        self.assertIn("- tdd", sm)
        self.assertIn("- security-review", sm)
        self.assertIn("`skill` tool", sm)

    def test_no_hint_when_skills_empty(self) -> None:
        task = self._build(skills=())
        sm = task.request.system_message or ""
        self.assertNotIn("<system-reminder>", sm)

    def test_hint_appended_after_existing_system_message(self) -> None:
        defn = AgentTypeDefinition(
            agent_type="hinted",
            description="x",
            tools=("read_file",),
            system_prompt="You are X.",
            skills=("tdd",),
        )
        parent = _build_parent(_registry("read_file"))
        task = SubagentTask(
            task_id="t-1",
            goal="g",
            request=EngineRequest(
                session_id="s",
                user_message="u",
                system_message="Parent prompt.",
            ),
            metadata={"_agent_type_def": defn},
        )
        DefaultSubagentBuilder().build_child(parent, task, child_depth=1)
        sm = task.request.system_message or ""
        # Order: type system_prompt, then parent prompt, then hint last.
        type_idx = sm.index("You are X.")
        parent_idx = sm.index("Parent prompt.")
        hint_idx = sm.index("<system-reminder>")
        self.assertLess(type_idx, parent_idx)
        self.assertLess(parent_idx, hint_idx)


# ----------------------------------------------------------------- AgentTool wiring

class AgentToolModelParameterTests(unittest.TestCase):
    def test_descriptor_exposes_model_param_with_enum(self) -> None:
        from aether.tools.builtins.agent_tool import AgentTool

        registry = AgentTypeRegistry(search_paths=[])
        tool = AgentTool(agent_type_registry=registry)
        props = tool.descriptor.parameters["properties"]
        self.assertIn("model", props)
        self.assertEqual(
            sorted(props["model"]["enum"]),
            sorted(["sonnet", "opus", "haiku", "inherit"]),
        )
        self.assertEqual(props["model"]["default"], "inherit")

    def test_model_argument_writes_metadata(self) -> None:
        # End-to-end: caller passes model="haiku"; builder reads it via
        # task.metadata["model_override"] and resolves the alias.
        from aether.tools.builtins.agent_tool import AgentTool

        registry = AgentTypeRegistry(search_paths=[])
        explore = registry.get("Explore")
        assert explore is not None

        captured: dict[str, SubagentTask] = {}

        class _CapturingManager:
            def run_task(self, *, parent, task: SubagentTask):
                captured["task"] = task
                from aether.subagents.contracts import SubagentResult, SubagentStatus
                return SubagentResult(
                    task_id=task.task_id,
                    status=SubagentStatus.COMPLETED,
                    summary="ok",
                    engine_result=None,
                )

        parent = _build_parent(_registry("read_file"))
        manager = _CapturingManager()
        tool = AgentTool(
            parent_agent=parent,
            subagent_manager=manager,
            agent_type_registry=registry,
        )
        ctx = TurnContext(session_id="s", iteration=0, metadata={})
        result = tool.execute(
            ToolCall(
                id="c1",
                name="task",
                arguments={
                    "subagent_type": "Explore",
                    "prompt": "find usages",
                    "model": "haiku",
                },
            ),
            ctx,
        )
        self.assertFalse(result.is_error, msg=result.content)
        self.assertEqual(
            captured["task"].metadata.get("model_override"),
            "haiku",
        )

    def test_inherit_model_does_not_set_override(self) -> None:
        from aether.tools.builtins.agent_tool import AgentTool

        registry = AgentTypeRegistry(search_paths=[])
        captured: dict[str, SubagentTask] = {}

        class _CapturingManager:
            def run_task(self, *, parent, task: SubagentTask):
                captured["task"] = task
                from aether.subagents.contracts import SubagentResult, SubagentStatus
                return SubagentResult(
                    task_id=task.task_id,
                    status=SubagentStatus.COMPLETED,
                    summary="ok",
                    engine_result=None,
                )

        parent = _build_parent(_registry("read_file"))
        tool = AgentTool(
            parent_agent=parent,
            subagent_manager=_CapturingManager(),
            agent_type_registry=registry,
        )
        ctx = TurnContext(session_id="s", iteration=0, metadata={})
        tool.execute(
            ToolCall(
                id="c1",
                name="task",
                arguments={
                    "subagent_type": "Explore",
                    "prompt": "find usages",
                    "model": "inherit",
                },
            ),
            ctx,
        )
        self.assertNotIn("model_override", captured["task"].metadata)

    def test_non_string_model_returns_error(self) -> None:
        from aether.tools.builtins.agent_tool import AgentTool

        registry = AgentTypeRegistry(search_paths=[])
        parent = _build_parent(_registry("read_file"))
        tool = AgentTool(
            parent_agent=parent,
            subagent_manager=object(),  # never reached
            agent_type_registry=registry,
        )
        ctx = TurnContext(session_id="s", iteration=0, metadata={})
        result = tool.execute(
            ToolCall(
                id="c1",
                name="task",
                arguments={
                    "subagent_type": "Explore",
                    "prompt": "go",
                    "model": 42,
                },
            ),
            ctx,
        )
        self.assertTrue(result.is_error)
        self.assertIn("'model' must be a string", result.content)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
