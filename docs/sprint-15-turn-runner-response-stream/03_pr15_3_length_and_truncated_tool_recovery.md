# PR 15.3 - Length and Truncated Tool Recovery

## Goal

Extract finish_reason=`length` handling and truncated tool-call argument
recovery from the runner.

## New Module

Either extend `response_finalization.py` or add:

- `aether/agents/runtime/response_repair.py`

Recommended types:

- `LengthContinuationController`
- `ToolCallArgumentRepairController`

## Scope

Move:

- `_handle_length_finish_reason`
- `_handle_length_with_tool_calls`
- `_validate_tool_call_arguments`
- assistant/tool error injection for invalid JSON arguments
- rollback to last assistant message for truncated calls
- metadata counters for length/truncation/invalid JSON

## Boundary

This controller decides one of:

- continue with modified messages
- inject tool argument errors and continue
- finalize with a specific exit reason
- allow normal tool dispatch

It does not execute tools.

## Tests

Add:

- `aether/tests/agents/runtime/test_response_repair_controller.py`

Cover:

- Length prose continuation.
- Length tool-call truncation retry.
- Truncated tool call exhausted.
- Invalid JSON error injection.
- Retry counters and metadata remain stable.

## Acceptance

- Runner no longer directly parses or repairs tool-call JSON.
- Existing run-loop truncated tool and invalid JSON tests pass.

## Detailed Implementation Notes

### Files to Add

Prefer one module:

- `aether/agents/runtime/response_repair.py`
- `aether/tests/agents/runtime/test_response_repair_controller.py`

If the module grows too large, split into:

- `length_continuation.py`
- `tool_call_repair.py`

Do not split prematurely if shared metadata counters make the call sites harder
to follow.

### Controller Decisions

The controller should return one of these decisions:

- `proceed_to_tool_dispatch`
- `continue_loop`
- `finalize`
- `inject_tool_errors`

Recommended shape:

```python
@dataclass(slots=True)
class ResponseRepairResult:
    action: ResponseRepairAction
    messages: list[dict[str, Any]]
    final_response: str | None = None
    exit_reason: ExitReason | None = None
    error_text: str | None = None
    injection_messages: list[dict[str, Any]] = field(default_factory=list)
```

### Behaviors to Move

Move:

- `finish_reason == "length"` prose continuation
- `finish_reason == "length"` with tool calls
- invalid JSON argument detection
- truncated argument retry
- tool error injection after persistent invalid JSON
- rollback to last valid assistant message
- visible text preservation for truncated output

### History Safety Rules

The controller must respect these existing safety properties:

- If a tool call is detected as truncated, do not append the malformed assistant
  tool-call message to canonical history.
- If invalid JSON is repairable by model feedback, append assistant tool call
  plus tool error stubs exactly as existing code does.
- If visible text exists before a truncated tool call, preserve it as partial
  final response only in the existing terminal path.

### Budget Rules

The controller should not consume or refund iteration budget directly. It should
return an action. The runner then checks `budget.exhausted` in the same places
as today.

### Tests

Controller-level tests should build `NormalizedResponse` fixtures:

- length text with content
- length text with no content
- length with valid tool call
- length with incomplete JSON tool args
- non-length invalid JSON tool args
- max retry exceeded

Regression tests:

- `aether/tests/engine/test_run_loop_length_continuation.py`
- `aether/tests/engine/test_run_loop_truncated_tool_call.py`
- `aether/tests/engine/test_json_tolerance_pr4.py`
- `aether/tests/engine/test_unicode_payload_recovery.py`

### Review Checklist

- No direct provider invocation in repair controller.
- No direct tool dispatch in repair controller.
- Existing metadata counters are preserved.
- Error injected tool messages keep the same content shape.
