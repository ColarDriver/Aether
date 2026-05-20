# PR 15.1 - Turn Runner Skeleton

## Goal

Move the main loop skeleton from `AgentEngine.run_loop()` into a runtime runner
without changing behavior.

## New Module

Add:

- `aether/agents/runtime/turn_runner.py`

Recommended types:

- `TurnRunner`
- `TurnRunInput`
- `TurnRunResult`
- `TurnRunnerAdapter`
- `LegacyTurnRunnerAdapter`

## Boundary

The runner owns:

- Loop state transitions.
- Iteration budget consumption/refund.
- Calling context assembly.
- Calling provider invocation with recovery.
- Calling tool dispatch.
- Selecting terminal exit reason.
- Calling lifecycle finalization.

The runner does not own:

- Provider payload conversion.
- Tool registry dispatch internals.
- Context assembly details.
- Response finalization internals after PR15.2.
- Stream callback construction after PR15.4.

## Migration Strategy

1. Create `TurnRunner` with a legacy adapter that calls current private methods.
2. Move the loop body mechanically.
3. Leave `AgentEngine.run_loop()` as:
   - set current session id
   - call runner
   - cleanup
4. Preserve state machine transitions exactly.

## Tests

Add:

- `aether/tests/agents/runtime/test_turn_runner.py`

Cover:

- Text response path.
- Tool response path.
- Provider error path.
- Interrupt before provider call.
- Max iteration exit.

## Acceptance

- `run_loop()` public behavior unchanged.
- No gateway/TUI schema changes.
- Existing engine tests pass.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/agents/runtime/turn_runner.py`
- `aether/tests/agents/runtime/test_turn_runner.py`

### Initial Runner Contract

Use a request/result shape that mirrors the current `run_loop` locals:

```python
@dataclass(slots=True)
class TurnRunInput:
    request: EngineRequest

@dataclass(slots=True)
class TurnRunResult:
    result: EngineResult
```

The first PR can keep `TurnRunResult` tiny because finalization still returns an
`EngineResult`. Do not over-design for streaming or async yet.

### Adapter Boundary

The runner should receive explicit dependencies:

- `SessionLifecycleController`
- `ContextAssemblyPipeline`
- `RecoveryController`
- `ToolDispatchController`
- a legacy adapter for private engine helpers that have not moved yet
- `EngineConfig`
- `EngineServices`

Avoid passing the entire `AgentEngine` except through a `LegacyTurnRunnerAdapter`
whose methods are narrowly named. This makes later PRs able to delete adapter
methods one by one.

### Mechanical Migration Steps

1. Copy the existing `run_loop` body into `TurnRunner.run`.
2. Replace `self.config` and `self.services` references with injected
   dependencies.
3. Replace calls to still-engine-private helpers with adapter methods.
4. Leave `AgentEngine.run_loop` responsible only for:
   - setting `_current_session_id`
   - creating stream callbacks if PR15.4 has not happened yet
   - calling runner
   - cleanup in `finally`
5. Run tests before changing any response behavior.

### Adapter Method List

Expect the first adapter to include methods for:

- `is_interrupted`
- `handle_pipeline_error`
- `pop_context_response`
- `invoke_provider_with_recovery`
- `accumulate_usage`
- `safe_call_hook`
- `handle_length_finish_reason`
- `handle_length_with_tool_calls`
- `validate_tool_call_arguments`
- `append_assistant_tool_message`
- `maybe_recover_phantom_tool_intent`
- `dispatch_synthesized_tool_calls`
- `finalize_empty_response`

This list should shrink in PR15.2-15.4.

### Tests

Use `ScriptedProvider` and small fake tools. Cover:

- one text response
- one tool call followed by text response
- provider error converted to failed result
- interrupt before first provider call
- max-iterations path with no final text

Do not assert private implementation details like adapter call counts unless
needed to prevent regressions.

### Review Checklist

- No response behavior changed in this PR.
- `run_loop` is shorter but still public-compatible.
- Runner tests do not require network or TUI.
