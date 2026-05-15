from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.agents.types import AgentTypeRegistry, VERIFIER_AGENT_TYPE
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse, ToolCall, ToolResult, TurnContext
from aether.subagents.contracts import SubagentTask
from aether.subagents.default_builder import DefaultSubagentBuilder
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


class AgentTypeBuilderTests(unittest.TestCase):
    def test_verifier_definition_filters_child_tools(self) -> None:
        type_registry = AgentTypeRegistry(search_paths=[])
        verifier = type_registry.get(VERIFIER_AGENT_TYPE)
        assert verifier is not None
        parent = AgentEngine(
            ScriptedProvider([NormalizedResponse(content="parent")]),
            tool_registry=_registry("read_file", "shell", "file_edit", "write_file", "task"),
            config=EngineConfig(use_builtin_tools=False),
            agent_type_registry=type_registry,
        )
        task = SubagentTask(
            task_id="verifier-task",
            goal="verify",
            request=EngineRequest(session_id="child-verifier", user_message="verify"),
            metadata={"_agent_type_def": verifier},
        )

        child = DefaultSubagentBuilder().build_child(parent, task, child_depth=1)

        self.assertTrue(child.services.tool_registry.has("read_file"))
        self.assertTrue(child.services.tool_registry.has("shell"))
        self.assertFalse(child.services.tool_registry.has("file_edit"))
        self.assertFalse(child.services.tool_registry.has("write_file"))
        self.assertFalse(child.services.tool_registry.has("task"))

    def test_agent_type_system_prompt_is_added_to_child_request(self) -> None:
        type_registry = AgentTypeRegistry(search_paths=[])
        verifier = type_registry.get(VERIFIER_AGENT_TYPE)
        assert verifier is not None
        parent = AgentEngine(
            ScriptedProvider([NormalizedResponse(content="parent")]),
            tool_registry=_registry("read_file", "shell"),
            config=EngineConfig(use_builtin_tools=False),
            agent_type_registry=type_registry,
        )
        task = SubagentTask(
            task_id="verifier-task",
            goal="verify",
            request=EngineRequest(
                session_id="child-verifier-prompt",
                user_message="verify",
                system_message="Parent supplied prompt.",
            ),
            metadata={"_agent_type_def": verifier},
        )

        DefaultSubagentBuilder().build_child(parent, task, child_depth=1)

        assert task.request.system_message is not None
        self.assertIn("independent verifier", task.request.system_message)
        self.assertTrue(task.request.system_message.endswith("Parent supplied prompt."))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
