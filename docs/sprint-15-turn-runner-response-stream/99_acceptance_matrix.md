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
