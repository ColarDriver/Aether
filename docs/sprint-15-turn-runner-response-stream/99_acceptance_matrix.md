# Sprint 15 - Acceptance Matrix

| Scenario | 15.1 Runner | 15.2 Finalization | 15.3 Repair | 15.4 Stream | 15.5 Acceptance |
|---|---|---|---|---|---|
| Text response | runner owns loop | finalizes text | - | stream optional | regression pass |
| Empty response | runner delegates | handles empty | - | - | regression pass |
| Phantom tool | runner delegates | detects/retries | - | - | regression pass |
| Length prose | runner delegates | - | continues | stream stable | regression pass |
| Truncated tool call | runner delegates | - | retries/refuses | silent tokens stable | regression pass |
| Invalid JSON args | runner delegates | - | injects errors | - | regression pass |
| Tool call | runner dispatches | - | validates before dispatch | - | regression pass |
| Interrupt | runner exits | finalization skipped | repair skipped | callback interrupt | regression pass |
| Max iterations | runner detects | final summary stable | repair budget stable | - | regression pass |

## Unit Test Map

| File | Purpose |
|---|---|
| `aether/tests/agents/runtime/test_turn_runner.py` | loop orchestration and state transitions |
| `aether/tests/agents/runtime/test_response_finalization_controller.py` | text, empty, phantom, synthesized dispatch decisions |
| `aether/tests/agents/runtime/test_response_repair_controller.py` | length continuation, truncated tool call, invalid JSON |
| `aether/tests/agents/runtime/test_stream_controller.py` | visible/silent stream callbacks and interrupt |
| existing `aether/tests/engine/test_empty_response_pipeline.py` | empty-response behavior parity |
| existing `aether/tests/engine/test_run_loop_truncated_tool_call.py` | truncated tool-call parity |
| existing `aether/tests/engine/test_streaming_generate.py` | streaming parity |

## Implementation Status

| Area | Owner After Sprint 15 | Notes |
|---|---|---|
| Turn orchestration | `aether/agents/runtime/turn_runner.py` | `AgentEngine.run_loop()` is now a facade that sets `_current_session_id`, calls `TurnRunner.run(...)`, and clears the active session id. |
| Text/empty/phantom finalization | `aether/agents/runtime/response_finalization.py` | The controller decides whether to finalize, continue, or dispatch synthesized tool calls. Detailed empty/phantom helpers remain delegated through a narrow adapter. |
| Length/truncated/invalid JSON repair | `aether/agents/runtime/response_repair.py` | The controller owns the decision path before tool dispatch. Deep repair helpers stay in `AgentEngine` for behavior parity. |
| Visible/silent streaming callbacks | `aether/agents/runtime/stream_controller.py` | The runner receives callback objects from `StreamController`; visible and silent callbacks both check interrupts. |
| Tool dispatch | existing `aether/agents/runtime/tool_dispatch.py` plus legacy adapter | Real tool execution stays in the existing dispatch controller. Synthesized phantom dispatch still uses the legacy engine helper. |

## Line Count Audit

Command:

```bash
wc -l aether/agents/core/agent.py aether/agents/runtime/*.py
```

| File set | `master` before sprint | Current branch |
|---|---:|---:|
| `aether/agents/core/agent.py` | 6200 | 5677 |
| New Sprint 15 runtime controllers | 0 | 960 |
| Runtime package total | 2172 | 3132 |

Current detailed output:

```text
  5677 aether/agents/core/agent.py
     2 aether/agents/runtime/__init__.py
   279 aether/agents/runtime/context_assembly.py
   270 aether/agents/runtime/provider_invocation.py
   704 aether/agents/runtime/recovery_controller.py
   159 aether/agents/runtime/response_finalization.py
   205 aether/agents/runtime/response_repair.py
   286 aether/agents/runtime/session_lifecycle.py
   164 aether/agents/runtime/stream_controller.py
   631 aether/agents/runtime/tool_dispatch.py
   432 aether/agents/runtime/turn_runner.py
  8809 total
```

## Deferred Engine Helpers

These private methods intentionally remain in `aether/agents/core/agent.py` for
this sprint:

| Method | Reason It Stayed |
|---|---|
| `_build_stream_callback`, `_build_stream_silent_callback` | Compatibility wrappers for direct tests and for `_finalize_empty_response` fallback emission. They now delegate to `StreamController`. |
| `_finalize_empty_response` | Still combines empty-response classification, fallback streaming, codex intermediate ack handling, thinking prefill cleanup, and transcript mutation. The controller owns the decision boundary; deeper migration should be separate. |
| `_maybe_recover_phantom_tool_intent` | Depends on existing phantom retry counters, synthesized tool-call construction, and metadata names. The controller now isolates the call site. |
| `_dispatch_synthesized_tool_calls` | Shares dispatch bookkeeping with tool-result append, diagnostics, permission abort, steering, and state transitions. It stayed delegated to preserve ordering. |
| `_handle_length_finish_reason`, `_handle_length_with_tool_calls`, `_validate_tool_call_arguments` | These helpers still contain the detailed recovery mechanics. `ResponseRepairController` now owns when each path is selected. |

`LegacyTurnRunnerAdapter` is reduced to engine-owned concerns that were not part
of this sprint's extraction boundary: turn nudges, interrupt checks, pipeline
error handling, provider recovery, usage accumulation, hook calls, assistant
tool message append, and synthesized dispatch.

## Manual Checklist

- Plain text answer still streams and finalizes.
- Tool-use turn still appends assistant tool call and tool results correctly.
- Max-iteration turn still produces existing summary behavior.
- Interrupt during streaming still stops the turn.
- A malformed tool-call argument still routes through repair instead of crashing.

## Non-Regression Rules

- Do not change `AgentEngine` public API.
- Do not change tool result message shape.
- Do not change TUI stream event schema.
- Do not mutate provider-prepared messages into canonical transcript.
- Do not consume/refund iteration budget from multiple places.

## Verification

Run on `turn-runner-response-stream`:

```text
python -m pytest aether/tests/agents        # 131 passed
python -m pytest aether/tests/engine        # 169 passed
python -m pytest aether/tests/runtime       # 574 passed
python -m pytest aether/tests/gateway       # 224 passed
python -m pytest aether/tests/subagents     # 46 passed
uv run pyright aether/agents/core/agent.py aether/agents/runtime
# 0 errors, 0 warnings, 0 informations
```

No TUI or gateway wire schema changes were made in Sprint 15.
