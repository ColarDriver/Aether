# PR 18.2 - Parallel Scheduler Core

## Goal

Build a scheduler that decides whether a batch can run concurrently and executes
safe batches while preserving output order.

## New Module

Add:

- `aether/runtime/tools/parallel_scheduler.py`

Recommended types:

- `ToolExecutionScheduler`
- `ToolExecutionPlan`
- `ToolExecutionItem`
- `ToolExecutionResult`

## Safety Rules

Parallelize only when:

- batch size > 1
- every tool is marked parallel safe
- no interactive tools
- no permission prompts are pending in the first version
- path-scoped tools do not overlap
- context interrupt signal is not already aborted

Path overlap rule:

- `/repo/a` overlaps `/repo/a/b`
- `/repo/a` does not overlap `/repo/c`
- relative paths must be normalized against current cwd without requiring file
  existence

## Ordering

Execution may be concurrent, but returned results must preserve original
tool-call order.

## Tests

Add:

- `aether/tests/runtime/tools/test_parallel_scheduler.py`

Cover:

- Safe read-only batch runs in parallel.
- Unsafe mixed batch chooses sequential.
- Path overlap blocks parallelism.
- Result order is stable.
- Exceptions become tool results in the same shape as sequential dispatch.

## Acceptance

- Scheduler is unit-tested without changing engine dispatch yet.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/runtime/tools/parallel_scheduler.py`
- `aether/tests/runtime/tools/test_parallel_scheduler.py`

### Scheduler Contract

Recommended shape:

```python
@dataclass(slots=True)
class ScheduledToolCall:
    index: int
    call: ToolCall
    capabilities: ToolCapabilities
    scope_path: Path | None = None

@dataclass(slots=True)
class ToolExecutionPlan:
    mode: Literal["sequential", "parallel"]
    reason: str
    calls: list[ScheduledToolCall]

class ToolExecutionScheduler:
    def plan(self, calls: list[ToolCall], *, context: TurnContext, cwd: Path) -> ToolExecutionPlan: ...
    def execute_parallel(...): ...
```

Keep planning separate from execution so ToolDispatchController can inspect and
record fallback reasons.

### Execution Boundary

The scheduler should not know permission policy. It receives already-approved
or known-safe calls. ToolDispatchController remains responsible for permission
and plan-mode gates.

Execution callback shape:

```python
Callable[[ToolCall], ToolResult]
```

The scheduler can call this callback in worker threads and collect results.

### Worker Model

Use `ThreadPoolExecutor` for blocking tools. Set `max_workers` from config with
a conservative default, for example 4.

Do not use global shared executor in first version. Per-batch executor with
bounded workers is easier to clean up and test.

### Ordering

Return results as:

```python
list[tuple[int, ToolResult]]
```

sorted by original index. ToolDispatchController then appends messages in that
order.

### Exception Handling

If a tool callback raises:

- convert to a `ToolResult` with `is_error=True`
- include stable error text
- do not crash sibling tool results
- record exception metadata without traceback unless debug mode already does so

### Tests

Use fake sleeping tools:

- two safe calls complete faster in parallel than sequential threshold
- result order remains original order even if second finishes first
- exception from one worker becomes error result
- unsafe plan does not call executor
- path overlap rejects parallel mode
- max worker count is respected

### Review Checklist

- Scheduler has no imports from `AgentEngine`.
- Scheduler has no permission prompts.
- Scheduler can be tested with fake callbacks only.
