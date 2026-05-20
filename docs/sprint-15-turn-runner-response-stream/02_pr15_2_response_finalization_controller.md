# PR 15.2 - Response Finalization Controller

## Goal

Extract response terminal handling from the loop.

## New Module

Add:

- `aether/agents/runtime/response_finalization.py`

Recommended types:

- `ResponseFinalizationController`
- `ResponseFinalizationInput`
- `ResponseFinalizationResult`
- `ResponseFinalizationAdapter`

## Scope

Move or delegate these behaviors:

- Combine truncated prefix parts with final content.
- Store assistant text response in messages.
- Empty response recovery/finalization.
- Phantom tool intent detection and retry/synthesized dispatch decision.
- Final exit reason selection for text/empty/phantom cases.

## Not In Scope

- Length continuation and truncated tool-call JSON recovery. That is PR15.3.
- Provider error recovery. That remains recovery controller.
- Tool dispatch implementation.

## Tests

Add:

- `aether/tests/agents/runtime/test_response_finalization_controller.py`

Cover:

- Plain text final response.
- Empty response finalization.
- Empty response recovery retry.
- Phantom tool retry.
- Phantom tool exhausted exit reason.
- Synthesized tool dispatch request shape.

## Acceptance

- The no-tool-call branch in the runner becomes a call to the controller.
- Existing empty response, phantom tool, and max-iteration tests pass.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/agents/runtime/response_finalization.py`
- `aether/tests/agents/runtime/test_response_finalization_controller.py`

### Controller Contract

Recommended result shape:

```python
@dataclass(slots=True)
class ResponseFinalizationResult:
    action: Literal["finalize", "continue", "dispatch_synthesized"]
    messages: list[dict[str, Any]]
    final_response: str | None = None
    error_text: str | None = None
    exit_reason: ExitReason | None = None
    synthesized_response: NormalizedResponse | None = None
```

The runner should own loop continuation and state transitions. The controller
should describe what happened.

### Behaviors to Move

Move the no-tool-call branch in phases:

1. Prefix/suffix combination from `truncated_response_prefix_parts`.
2. Construction of `response_to_store`.
3. Phantom intent check.
4. Empty response finalization.
5. Final assistant message append if currently part of finalization.

If `_maybe_recover_phantom_tool_intent` is too coupled to engine helpers, keep it
behind a narrow adapter for this PR and move only the orchestration. Do not
rewrite phantom parsing in this PR.

### Metadata Rules

Preserve keys currently used by tests:

- `truncated_response_prefix_parts`
- `partial`
- `length_exit_reason`
- phantom retry counters
- empty response retry counters

If new metadata is needed, nest it under `response_finalization` and ensure
`turn_metadata.py` knows whether it is public or internal.

### Synthetic Dispatch Boundary

When phantom recovery synthesizes tool calls, the controller should not execute
tools. It should return `action="dispatch_synthesized"` and the synthetic
`NormalizedResponse`. The runner then calls the existing dispatch path. This
keeps tool permission, plan blocker, post hooks, and diagnostics in one place.

### Tests

Use direct controller tests for:

- normal text response becomes final response
- content prefix is combined exactly once
- empty response returns the current exit reason
- phantom retry returns `continue`
- phantom exhausted returns terminal phantom exit reason
- synthesized tool call returns `dispatch_synthesized`

Then run existing:

- `aether/tests/engine/test_empty_response_pipeline.py`
- `aether/tests/engine/test_phantom_synthesis.py`
- `aether/tests/engine/test_run_loop_phantom_tool.py`

### Review Checklist

- No tool execution inside finalization controller.
- No provider calls inside finalization controller.
- No state machine mutation inside finalization controller unless hidden behind
  a temporary legacy adapter.
