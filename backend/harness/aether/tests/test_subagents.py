from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.contracts import EngineRequest, NormalizedResponse
from aether.subagents import SubagentManager, SubagentStatus, SubagentTask


class SubagentTests(unittest.TestCase):
    def test_run_single_subagent_task(self) -> None:
        parent_provider = ScriptedProvider([NormalizedResponse(content="parent-ready")])
        manager = SubagentManager(max_concurrent_children=2, max_spawn_depth=2)
        parent = AgentEngine(provider=parent_provider, subagent_manager=manager)

        child_provider = ScriptedProvider([NormalizedResponse(content="child-done")])
        task = SubagentTask(
            task_id="t1",
            goal="do delegated work",
            request=EngineRequest(session_id="child-session-1", user_message="work"),
            provider=child_provider,
        )

        results = parent.run_subagents([task])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, SubagentStatus.COMPLETED)
        self.assertEqual(results[0].summary, "child-done")
        self.assertIsNotNone(results[0].engine_result)

    def test_depth_limit_guard(self) -> None:
        manager = SubagentManager(max_concurrent_children=2, max_spawn_depth=2)
        parent = AgentEngine(
            provider=ScriptedProvider([NormalizedResponse(content="unused")]),
            subagent_manager=manager,
            delegate_depth=2,
        )

        task = SubagentTask(
            task_id="t2",
            goal="too deep",
            request=EngineRequest(session_id="child-session-2", user_message="work"),
            provider=ScriptedProvider([NormalizedResponse(content="unused")]),
        )

        with self.assertRaises(RuntimeError):
            parent.run_subagents([task])


if __name__ == "__main__":
    unittest.main()
