from __future__ import annotations

from pathlib import Path
import threading
import time

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tools.parallel_scheduler import ToolExecutionScheduler, paths_overlap


def _call(index: int, name: str = "read_file", **arguments) -> ToolCall:
    return ToolCall(id=f"c{index}", name=name, arguments=dict(arguments))


def _context() -> TurnContext:
    return TurnContext(session_id="s", iteration=1, metadata={})


def test_safe_read_only_batch_runs_in_parallel() -> None:
    scheduler = ToolExecutionScheduler(max_workers=2)
    calls = [_call(1, path="a.py"), _call(2, path="b.py")]
    plan = scheduler.plan(calls, context=_context(), cwd=Path("/repo"))

    assert plan.mode == "parallel"
    started: list[float] = []
    lock = threading.Lock()

    def execute(call: ToolCall) -> ToolResult:
        with lock:
            started.append(time.perf_counter())
        time.sleep(0.08)
        return ToolResult(tool_call_id=call.id, name=call.name, content=call.id)

    t0 = time.perf_counter()
    results = scheduler.execute_parallel(plan, context=_context(), execute=execute)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.15
    assert [item.result.content for item in results] == ["c1", "c2"]
    assert len(started) == 2


def test_unsafe_mixed_batch_chooses_sequential() -> None:
    scheduler = ToolExecutionScheduler()
    plan = scheduler.plan(
        [_call(1, "read_file", path="a.py"), _call(2, "write_file", path="b.py")],
        context=_context(),
        cwd=Path("/repo"),
    )

    assert plan.mode == "sequential"
    assert plan.reason.startswith("tool-not-parallel-safe:write_file")


def test_path_overlap_blocks_parallelism() -> None:
    scheduler = ToolExecutionScheduler()
    plan = scheduler.plan(
        [_call(1, "list_dir", path="src"), _call(2, "read_file", path="src/app.py")],
        context=_context(),
        cwd=Path("/repo"),
    )

    assert plan.mode == "sequential"
    assert plan.reason.startswith("path-overlap")


def test_result_order_is_stable_when_second_finishes_first() -> None:
    scheduler = ToolExecutionScheduler(max_workers=2)
    calls = [_call(1, path="a.py"), _call(2, path="b.py")]
    plan = scheduler.plan(calls, context=_context(), cwd=Path("/repo"))

    def execute(call: ToolCall) -> ToolResult:
        if call.id == "c1":
            time.sleep(0.08)
        return ToolResult(tool_call_id=call.id, name=call.name, content=call.id)

    results = scheduler.execute_parallel(plan, context=_context(), execute=execute)

    assert [item.index for item in results] == [0, 1]
    assert [item.result.content for item in results] == ["c1", "c2"]


def test_worker_exception_becomes_error_result() -> None:
    scheduler = ToolExecutionScheduler(max_workers=2)
    calls = [_call(1, path="a.py"), _call(2, path="b.py")]
    plan = scheduler.plan(calls, context=_context(), cwd=Path("/repo"))

    def execute(call: ToolCall) -> ToolResult:
        if call.id == "c2":
            raise RuntimeError("boom")
        return ToolResult(tool_call_id=call.id, name=call.name, content="ok")

    results = scheduler.execute_parallel(plan, context=_context(), execute=execute)

    assert results[0].result.content == "ok"
    assert results[1].result.is_error is True
    assert results[1].result.metadata["parallel_worker_error"] is True


def test_permission_required_batch_is_sequential() -> None:
    scheduler = ToolExecutionScheduler()
    plan = scheduler.plan(
        [_call(1, path="a.py"), _call(2, path="b.py")],
        context=_context(),
        cwd=Path("/repo"),
        permission_required={"c2"},
    )

    assert plan.mode == "sequential"
    assert plan.reason == "permission-required"


def test_paths_overlap_rules() -> None:
    assert paths_overlap(Path("/repo/a"), Path("/repo/a/b")) is True
    assert paths_overlap(Path("/repo/a"), Path("/repo/c")) is False
