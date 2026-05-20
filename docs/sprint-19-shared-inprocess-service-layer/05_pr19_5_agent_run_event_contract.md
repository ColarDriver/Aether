# PR 19.5 - Agent Run Event Contract

## Goal

Define transport-neutral agent run request/result/event contracts and lock down
current gateway event compatibility before moving run lifecycle code.

## Current Problem

`agent_methods.py` currently owns both business lifecycle and JSON-RPC event
translation. It emits TUI-consumed gateway events such as text deltas, reasoning
deltas, stream progress, status changes, loop-state changes, iteration
start/end, tool start/result, token usage, done, cancelled, and error. Moving
this code without golden tests risks breaking the TUI even if the engine still
runs.

## Changes

Add `aether/services/runs/events.py` with service-level event dataclasses:

- `RunStarted`
- `AssistantDelta`
- `ReasoningDelta`
- `SilentProgress`
- `RunStatusChanged`
- `LoopStateChanged`
- `IterationStarted`
- `IterationFinished`
- `ToolStarted`
- `ToolFinished`
- `PermissionRequested`
- `TokenUsageUpdated`
- `RunFinished`
- `RunFailed`
- `RunCancelled`

Add `aether/services/runs/contracts.py`:

- `AgentRunRequest`
- `AgentRunOptions`
- `AgentRunResult`
- `AgentRunStatus`
- `AgentRunSnapshot`
- `AgentRunCancelRequest`
- `RunEvent`

Contract requirements:

- Events are transport-neutral and do not import gateway protocol classes.
- Events include enough fields for the gateway to recreate existing JSON-RPC
  event payloads.
- Sequence semantics are explicit for visible and silent deltas.
- Tool events preserve `tool_call_id`, `tool_name`, `arguments`, result
  content, `is_error`, metadata, and iteration.
- Permission events expose request metadata but not UI behavior.
- Final result exposes the same information currently returned by
  `agent.run`, but as a service result.

Add gateway compatibility fixture tests before implementation migration:

- Capture current event wire shapes from `agent_methods.py`.
- Build a mapper from service events to gateway events.
- Verify mapped events exactly match current gateway event names and public
  fields.

This PR may add an adapter helper such as:

- `aether/gateway/handlers/run_event_adapter.py`

It must only map service events to gateway protocol models. It must not start
runs or contain business lifecycle behavior.

## Tests

Add:

- `aether/tests/services/test_agent_run_contracts.py`
- `aether/tests/gateway/test_agent_run_event_compat.py`

Cover:

- every service event can be serialized to public-safe data
- service events do not import gateway protocol classes
- service event -> gateway event mapping preserves current event type strings
- text/reasoning/silent progress sequence numbers match current behavior
- tool event public fields match existing TUI expectations
- done/cancelled/error payload fields match current gateway events
- metadata is sanitized with the same rules as current `_safe_metadata`

## Migration Notes

- Do not move `agent_run` yet.
- Do not change `AgentEngine`.
- Add golden tests first so PR 19.6 and PR 19.8 can refactor safely.
- Gateway event adapter may temporarily live under gateway because it is a wire
  mapper, not business logic.

## Risks

- Service events that mirror gateway events too closely can become disguised
  wire schemas. Keep names transport-neutral, but include enough data to map
  losslessly.
- Missing sequence semantics will break streaming UI behavior. Make sequence
  ownership explicit.

## Non-Goals

- Do not start or cancel real runs in this PR.
- Do not migrate `agent_methods.py`.
- Do not change TUI event payloads.
- Do not implement HTTP/SSE/WebSocket event framing.
- Do not move permission UI logic into services.

## Acceptance

- Agent run service contracts exist and are gateway-independent.
- Current gateway event shape is captured by tests before migration.
- A service-event-to-gateway-event mapper can reproduce current TUI event
  payloads.
- Later run lifecycle refactors have a compatibility harness.
