from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.contracts import (
    EngineStatus,
    EngineRequest,
    NormalizedResponse,
    ToolCall,
    ToolResult,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class SumTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="sum")

    def execute(self, call: ToolCall, context) -> ToolResult:
        total = int(call.arguments["a"]) + int(call.arguments["b"])
        return ToolResult(tool_call_id=call.id, name=call.name, content=str(total))


class TrajectoryPersistenceTests(unittest.TestCase):
    def test_default_disabled_does_not_create_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_dir = Path(tmp) / "trajectories"
            engine = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="ok")]),
                config=EngineConfig(
                    use_builtin_tools=False,
                    trajectory_dir=trajectory_dir,
                ),
            )

            result = engine.run_turn(
                EngineRequest(session_id="traj-disabled", user_message="hello")
            )

            self.assertEqual(result.status, EngineStatus.COMPLETED)
            self.assertFalse(trajectory_dir.exists())
            self.assertEqual(
                result.metadata["trajectory"],
                {"saved": False, "path": None, "error": None},
            )

    def test_successful_turn_writes_sessions_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_dir = Path(tmp) / "trajectories"
            engine = AgentEngine(
                ScriptedProvider(
                    [
                        NormalizedResponse(
                            content="answer",
                            metadata={"reasoning_content": "thinking"},
                        )
                    ]
                ),
                config=EngineConfig(
                    use_builtin_tools=False,
                    save_trajectories=True,
                    trajectory_dir=trajectory_dir,
                ),
            )

            result = engine.run_turn(
                EngineRequest(session_id="traj-success", user_message="hello")
            )

            self.assertEqual(result.status, EngineStatus.COMPLETED)
            meta = result.metadata["trajectory"]
            self.assertTrue(meta["saved"])
            path = Path(meta["path"])
            self.assertEqual(path, trajectory_dir / "sessions" / "traj-success.jsonl")
            payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(payload["completed"])
            self.assertEqual(payload["session_id"], "traj-success")
            self.assertEqual(payload["conversations"][0], {"from": "human", "value": "hello"})
            self.assertIn("<think>\nthinking\n</think>", payload["conversations"][1]["value"])
            self.assertIn("answer", payload["conversations"][1]["value"])

    def test_failed_turn_writes_failed_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_dir = Path(tmp) / "trajectories"
            engine = AgentEngine(
                ScriptedProvider([]),
                config=EngineConfig(
                    use_builtin_tools=False,
                    save_trajectories=True,
                    trajectory_dir=trajectory_dir,
                ),
            )

            result = engine.run_turn(
                EngineRequest(session_id="traj-failed", user_message="hello")
            )

            self.assertEqual(result.status, EngineStatus.FAILED)
            meta = result.metadata["trajectory"]
            self.assertTrue(meta["saved"])
            path = Path(meta["path"])
            self.assertEqual(path, trajectory_dir / "failed" / "traj-failed.jsonl")
            payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(payload["completed"])
            self.assertEqual(payload["conversations"][0], {"from": "human", "value": "hello"})

    def test_tool_calls_and_tool_results_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_dir = Path(tmp) / "trajectories"
            registry = ToolRegistry()
            registry.register(SumTool())
            engine = AgentEngine(
                ScriptedProvider(
                    [
                        NormalizedResponse(
                            content="",
                            tool_calls=[
                                ToolCall(id="call-1", name="sum", arguments={"a": 2, "b": 3})
                            ],
                            metadata={"reasoning_content": "use sum"},
                        ),
                        NormalizedResponse(content="done"),
                    ]
                ),
                tool_registry=registry,
                config=EngineConfig(
                    max_iterations=4,
                    use_builtin_tools=False,
                    save_trajectories=True,
                    trajectory_dir=trajectory_dir,
                ),
            )

            result = engine.run_turn(
                EngineRequest(session_id="traj-tool", user_message="calculate")
            )

            self.assertEqual(result.status, EngineStatus.COMPLETED)
            payload = json.loads(Path(result.metadata["trajectory"]["path"]).read_text(encoding="utf-8").splitlines()[0])
            values = [item["value"] for item in payload["conversations"]]
            self.assertTrue(any("<tool_call>" in value and '"name":"sum"' in value for value in values))
            self.assertTrue(any("<tool_response>" in value and '"content":"5"' in value for value in values))

    def test_write_failure_is_reported_without_failing_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_path = Path(tmp) / "not_a_directory"
            trajectory_path.write_text("occupied", encoding="utf-8")
            engine = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="ok")]),
                config=EngineConfig(
                    use_builtin_tools=False,
                    save_trajectories=True,
                    trajectory_dir=trajectory_path,
                ),
            )

            result = engine.run_turn(
                EngineRequest(session_id="traj-write-fail", user_message="hello")
            )

            self.assertEqual(result.status, EngineStatus.COMPLETED)
            self.assertFalse(result.metadata["trajectory"]["saved"])
            self.assertIsNone(result.metadata["trajectory"]["path"])
            self.assertIsNotNone(result.metadata["trajectory"]["error"])


if __name__ == "__main__":
    unittest.main()
