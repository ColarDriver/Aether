# Sprint 15 - Turn Runner, Response Finalization, and Stream Controller

## Decision

This is one sprint, but not one PR.

Turn runner extraction, response finalization, and stream handling all sit in
the same `AgentEngine.run_loop()` path. They should be planned together so the
state transitions stay coherent, but implemented in separate PRs.

## Motivation

Sprint 13 added controller boundaries, but `aether/agents/core/agent.py` still
contains a large run loop and many behavioral helpers. The next structural goal
is to make `AgentEngine` a facade and move turn execution into testable runtime
components.

Current high-weight responsibilities still in `AgentEngine` include:

- Loop state orchestration.
- Iteration budget consumption/refund.
- Length continuation.
- Truncated tool call retry/refusal.
- Phantom tool recovery.
- Empty response finalization.
- Stream callback wrapping.
- Silent token streaming.
- Result finalization fallback for max iterations.

## Goals

- Introduce a `TurnRunner` or `LoopOrchestrator`.
- Extract stream callback construction into `StreamController`.
- Extract text/empty/length/phantom response finalization into
  `ResponseFinalizationController`.
- Keep public `AgentEngine.run_loop()`, `run_turn()`, and `resume()` unchanged.
- Reduce `AgentEngine` line count through actual responsibility movement.

## Non-Goals

- Do not change provider transport behavior.
- Do not change tool dispatch policy.
- Do not change context compression strategy.
- Do not change TUI rendering.
- Do not add credential runtime.

## PR Roadmap

| PR | File | Boundary |
|---|---|---|
| 15.1 | `01_pr15_1_turn_runner_skeleton.md` | Move loop skeleton into runtime runner with legacy adapters |
| 15.2 | `02_pr15_2_response_finalization_controller.md` | Extract text/empty/phantom finalization |
| 15.3 | `03_pr15_3_length_and_truncated_tool_recovery.md` | Extract length continuation and truncated tool-call paths |
| 15.4 | `04_pr15_4_stream_controller.md` | Extract visible/silent stream callbacks and interrupt handling |
| 15.5 | `05_pr15_5_facade_cleanup_and_acceptance.md` | Remove dead wrappers, add acceptance, document line-count delta |

## Completion Criteria

- `AgentEngine.run_loop()` becomes a thin call into `TurnRunner`.
- Response finalization behavior has focused tests.
- Stream callback behavior has focused tests.
- Engine regression suite remains green.

## Current Aether Anchors

The code still concentrated in `aether/agents/core/agent.py` includes:

- `AgentEngine.run_loop`
- `_build_stream_callback`
- `_build_stream_silent_callback`
- `_handle_length_finish_reason`
- `_handle_length_with_tool_calls`
- `_validate_tool_call_arguments`
- `_maybe_recover_phantom_tool_intent`
- `_dispatch_synthesized_tool_calls`
- `_finalize_empty_response`
- `_append_assistant_tool_message`
- `_append_final_assistant_message`
- `_resolve_terminal_exit_reason`

Sprint 15 should not move all of these in one PR. Move the runner skeleton
first, then response finalization, then repair, then stream callbacks.

## Target End State

The desired end shape is:

```text
AgentEngine.run_loop(request)
  -> TurnRunner.run(request)
       -> SessionLifecycleController.prepare_turn
       -> ContextAssemblyPipeline.assemble
       -> RecoveryController.invoke_with_recovery
       -> ResponseRepairController.inspect
       -> ToolDispatchController.dispatch
       -> ResponseFinalizationController.finalize
       -> SessionLifecycleController.finalize_turn
```

`AgentEngine` remains the public facade and dependency owner. It should not
continue to contain the details of response repair or stream callback behavior.

## State Machine Invariants

Preserve the current `LoopState` transitions. At minimum these transitions must
remain observable in the same scenarios:

- `PREPARE`
- `PRE_LLM`
- `LLM_CALL`
- `POST_LLM`
- `TOOL_DISPATCH`
- `TOOL_EXECUTE`
- `CHECK_EXIT`
- `FINALIZE`
- `FAILED`
- `INTERRUPTED`

If a controller needs to choose a transition, return a structured result and let
`TurnRunner` apply the transition. Do not let multiple controllers mutate the
state machine independently unless the old code already did so and the
migration is mechanical.

## Message Ownership Rules

- `messages` is the canonical transcript for the turn.
- `prepared_messages` is provider-bound and must not be appended to transcript.
- Tool result messages must preserve existing shape.
- Synthetic tool dispatch must append messages in the same order as real tool
  dispatch.
- Response repair must not poison canonical history with malformed tool calls
  when the current behavior rolls back.

## Risk Areas

- Cheap-tool budget refund can be broken if runner and tool dispatch disagree on
  `iterations`.
- Truncated tool-call recovery can accidentally append malformed assistant tool
  calls to history.
- Stream callback extraction can break TUI token counters even if final
  responses still pass tests.
- Phantom tool recovery can loop if retry counters move to the wrong metadata
  namespace.

These risks are why Sprint 15 is one sprint but five PRs.
