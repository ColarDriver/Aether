from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from aether.agents.types import AgentTypeRegistry
from aether.config.schema import EngineConfig
from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.subagents.contracts import SubagentResult, SubagentStatus, SubagentTask
from aether.tools.builtins.agent_tool import AgentTool


def _ctx(
    *,
    parent: Any | None = None,
    manager: Any | None = None,
    config: EngineConfig | None = None,
    registry: AgentTypeRegistry | None = None,
) -> TurnContext:
    md: dict[str, Any] = {"_engine_config": config or EngineConfig()}
    if parent is not None:
        md["_parent_agent"] = parent
    if manager is not None:
        md["_subagent_manager"] = manager
    if registry is not None:
        md["_agent_type_registry"] = registry
    return TurnContext(session_id="ses-agent-type", iteration=0, metadata=md)


class _StubManager:
    def __init__(self) -> None:
        self.calls: list[SubagentTask] = []

    def run_task(self, *, parent, task: SubagentTask) -> SubagentResult:
        del parent
        self.calls.append(task)
        return SubagentResult(
            task_id=task.task_id,
            status=SubagentStatus.COMPLETED,
            summary="done",
            engine_result=None,
            duration_seconds=0.1,
            metadata={"subagent_id": "sub-1"},
        )


class AgentToolTypeResolutionTests(unittest.TestCase):
    def test_without_registry_keeps_legacy_behavior(self) -> None:
        manager = _StubManager()
        parent = SimpleNamespace(delegate_depth=0, subagent_manager=manager)
        tool = AgentTool()

        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "find code", "subagent_type": "Explore"}),
            _ctx(parent=parent, manager=manager),
        )

        self.assertFalse(result.is_error, result.content)
        self.assertEqual(manager.calls[0].metadata["subagent_type"], "Explore")

    def test_unknown_type_returns_error_when_registry_injected(self) -> None:
        registry = AgentTypeRegistry(search_paths=[])
        manager = _StubManager()
        parent = SimpleNamespace(delegate_depth=0, subagent_manager=manager)
        tool = AgentTool(agent_type_registry=registry)

        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "find code", "subagent_type": "NotARealType"}),
            _ctx(parent=parent, manager=manager, registry=registry),
        )

        self.assertTrue(result.is_error)
        self.assertIn("Available:", result.content)
        self.assertFalse(manager.calls)

    def test_known_type_attaches_definition_to_task_metadata(self) -> None:
        registry = AgentTypeRegistry(search_paths=[])
        manager = _StubManager()
        parent = SimpleNamespace(delegate_depth=0, subagent_manager=manager)
        tool = AgentTool(agent_type_registry=registry)

        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "find code", "subagent_type": "Explore"}),
            _ctx(parent=parent, manager=manager, registry=registry),
        )

        self.assertFalse(result.is_error, result.content)
        self.assertIn("_agent_type_def", manager.calls[0].metadata)
        definition = manager.calls[0].metadata["_agent_type_def"]
        self.assertEqual(definition.agent_type, "Explore")

    def test_descriptor_exposes_enum_with_builtin_names(self) -> None:
        registry = AgentTypeRegistry(search_paths=[])
        tool = AgentTool(agent_type_registry=registry)
        field = tool.descriptor.parameters["properties"]["subagent_type"]
        self.assertIn("enum", field)
        self.assertIn("general-purpose", field["enum"])
        self.assertIn("Explore", field["enum"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
