# PR 18.4 - Tool Dispatch Integration

## Goal

Integrate the scheduler into `ToolDispatchController`.

## Target Files

Modify:

- `aether/agents/runtime/tool_dispatch.py`
- `aether/runtime/tools/capabilities.py`
- `aether/runtime/tools/parallel_scheduler.py`

## Integration Boundary

`ToolDispatchController` still owns:

- schema repair/dedup/delegate cap
- plan-mode blocker
- permission gate
- post-tool hooks
- metadata recording
- appending tool result messages

Scheduler owns:

- concurrent execution of already-approved safe calls
- result ordering
- worker cleanup

## Metadata

Add:

- `tool_parallel_enabled`
- `tool_parallel_batch_size`
- `tool_parallel_executed`
- `tool_parallel_fallback_reason`
- `tool_parallel_elapsed_ms`

Metadata must be public-safe and not include live worker objects.

## Tests

Add/update:

- `aether/tests/agents/runtime/test_tool_dispatch_controller.py`
- `aether/tests/engine/test_engine.py`
- `aether/tests/tools/test_builtin_tools.py`

Cover:

- Safe read-only calls parallelize.
- Mixed safe/unsafe falls back.
- Result transcript order unchanged.
- Post-tool hooks still fire in transcript order.
- Cheap-tool refund remains correct.

## Acceptance

- Parallel dispatch is enabled behind config flag first:
  `EngineConfig.parallel_tool_execution_enabled`.
- Default may be off for one PR if needed, then enabled after acceptance.

## Detailed Implementation Notes

### Config

Add to `EngineConfig`:

- `parallel_tool_execution_enabled: bool = False` for first integration PR
- `parallel_tool_max_workers: int = 4`

Only flip default to true after PR18.5 proves regression coverage, or keep it
false if product risk is still high.

### Integration Point

In `ToolDispatchController.dispatch`, integrate after:

- `prepare_tool_calls`
- schema repair
- plan-mode blocker checks
- permission decision for the batch or individual calls

and before executing safe tool callbacks.

Do not bypass:

- `fire_post_tool_hook`
- `accumulate_edited_paths`
- `maybe_mark_verifier_invoked`
- `dispatch_internal_diagnostic_update`
- `record_tool_result_error`
- `append_tool_result_message`

If parallel execution returns raw `ToolResult`s, run the same post-processing in
original call order.

### Result Shape

`ToolDispatchResult` may gain fields:

- `parallel_executed: bool`
- `parallel_fallback_reason: str | None`
- `parallel_batch_size: int`

Keep existing fields stable.

### Metadata

Write public-safe metadata:

```python
context.metadata["tool_parallel"] = {
    "enabled": True,
    "executed": True,
    "batch_size": 3,
    "fallback_reason": None,
    "elapsed_ms": 42,
}
```

If older flat metadata names are easier for current observability, mirror them,
but keep the nested shape documented.

### Post Hook Ordering

Even if execution is parallel, call post-tool hooks in original order. This
preserves downstream diagnostics and transcript assumptions.

### Tests

Update `test_tool_dispatch_controller.py` with fake tools:

- safe read-only parallel batch
- mixed safe/unsafe fallback
- permission fallback
- one parallel tool fails, another succeeds
- post hooks called in original order
- appended messages in original order
- cheap-tool refund still sees all tools cheap when appropriate

### Review Checklist

- Config flag can disable the feature completely.
- Sequential path remains the default until acceptance.
- Existing permission tests pass unchanged.
