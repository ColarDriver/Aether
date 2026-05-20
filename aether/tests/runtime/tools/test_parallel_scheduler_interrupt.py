from __future__ import annotations

from pathlib import Path
import threading
import time

from aether.runtime.control.interrupt_signal import InterruptSignal
from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tools.parallel_scheduler import ToolExecutionScheduler


def _call(index: int, name: str = "read_file", **arguments) -> ToolCall:
    return ToolCall(id=f"c{index}", name=name, arguments=dict(arguments))


def _context(signal: InterruptSignal | None = None) -> TurnContext:
    return TurnContext(session_id="s", iteration=1, metadata={}, interrupt_signal=signal)


def test_interrupt_before_scheduling_chooses_sequential() -> None:
    signal = InterruptSignal()
    signal.abort("user-stop")
    scheduler = ToolExecutionScheduler(max_workers=2)

    plan = scheduler.plan(
        [_call(1, path="a.py"), _call(2, path="b.py")],
        context=_context(signal),
        cwd=Path("/repo"),
    )

    assert plan.mode == "sequential"
    assert plan.reason == "interrupted-before-start"


def test_execute_parallel_interrupted_before_start_returns_stub_results() -> None:
    scheduler = ToolExecutionScheduler(max_workers=2)
    context = _context()
    plan = scheduler.plan(
        [_call(1, path="a.py"), _call(2, path="b.py")],
        context=context,
        cwd=Path("/repo"),
    )
    signal = InterruptSignal()
    signal.abort("user-stop")

    executed: list[str] = []
    results = scheduler.execute_parallel(
        plan,
        context=_context(signal),
        execute=lambda call: executed.append(call.id) or ToolResult(call.id, call.name, "ok"),
    )

    assert executed == []
    assert [item.result.metadata["interrupted"] for item in results] == [True, True]
    assert [item.result.metadata["tool_executed"] for item in results] == [False, False]


def test_interrupt_during_execution_cancels_pending_futures_and_preserves_order() -> None:
    signal = InterruptSignal()
    scheduler = ToolExecutionScheduler(max_workers=1)
    calls = [_call(1, path="a.py"), _call(2, path="b.py"), _call(3, path="c.py")]
    context = _context(signal)
    plan = scheduler.plan(calls, context=context, cwd=Path("/repo"))
    started = threading.Event()
    executed: list[str] = []

    def execute(call: ToolCall) -> ToolResult:
        executed.append(call.id)
        started.set()
        assert signal.wait(1.0)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="stopped",
            is_error=True,
            metadata={"interrupted": True},
        )

    worker = threading.Thread(
        target=lambda: setattr(
            worker_result,
            "items",
            scheduler.execute_parallel(plan, context=context, execute=execute),
        ),
    )
    worker_result = type("_Result", (), {"items": []})()
    worker.start()
    assert started.wait(1.0)
    signal.abort("user-stop")
    worker.join(1.0)

    assert not worker.is_alive()
    results = worker_result.items
    assert [item.call.id for item in results] == ["c1", "c2", "c3"]
    assert executed == ["c1"]
    assert results[0].result.metadata["interrupted"] is True
    assert results[1].result.metadata["interrupted"] is True
    assert results[2].result.metadata["interrupted"] is True
    assert results[1].result.metadata["tool_executed"] is False
    assert results[2].result.metadata["tool_executed"] is False


def test_permission_required_batch_falls_back_to_sequential() -> None:
    scheduler = ToolExecutionScheduler(max_workers=2)

    plan = scheduler.plan(
        [_call(1, path="a.py"), _call(2, path="b.py")],
        context=_context(),
        cwd=Path("/repo"),
        permission_required={"c2"},
    )

    assert plan.mode == "sequential"
    assert plan.reason == "permission-required"


def test_executor_shutdown_after_worker_error() -> None:
    scheduler = ToolExecutionScheduler(max_workers=2)
    context = _context()
    plan = scheduler.plan(
        [_call(1, path="a.py"), _call(2, path="b.py")],
        context=context,
        cwd=Path("/repo"),
    )

    def execute(call: ToolCall) -> ToolResult:
        if call.id == "c1":
            raise RuntimeError("boom")
        time.sleep(0.01)
        return ToolResult(tool_call_id=call.id, name=call.name, content="ok")

    results = scheduler.execute_parallel(plan, context=context, execute=execute)

    assert [item.result.tool_call_id for item in results] == ["c1", "c2"]
    assert results[0].result.is_error is True
    assert results[0].result.metadata["parallel_worker_error"] is True
