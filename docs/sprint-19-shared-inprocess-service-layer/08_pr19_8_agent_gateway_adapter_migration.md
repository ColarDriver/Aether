# PR 19.8 - Agent Gateway Adapter Migration

## Goal

Migrate `agent_methods.py` to `AgentRunService` after service event contracts
and core run lifecycle tests are in place.

## Current Problem

`agent_methods.py` is the highest-risk gateway handler. It owns run lifecycle,
streaming, event mapping, permission bridge wiring, cancellation, and final
response mapping. Migrating it before PR 19.5 and PR 19.6 would make it too easy
to break TUI streaming or permission behavior.

## Changes

Refactor `agent_methods.py` into a thin adapter:

- parse and validate JSON-RPC params
- build `AgentRunRequest`
- create gateway prompter implementations
- call `AgentRunService.start`
- map service events through the PR 19.5 gateway event adapter
- map `AgentRunResult` to the existing `agent.run` response dict
- call `AgentRunService.cancel` from `agent.cancel`
- map service errors to existing `GatewayError` or error response behavior

Keep gateway-owned code where it belongs:

- JSON-RPC request id -> run id mapping
- `notify("event", ...)` transport emission
- gateway prompter bridge implementation
- exact gateway protocol model serialization

Move or wrap service-owned code:

- run handle lifecycle
- provider/engine construction
- session persistence
- event sequencing
- cancellation state
- final result lookup

## Tests

Update or add:

- `aether/tests/gateway/test_agent_run_streaming.py`
- `aether/tests/gateway/test_agent_cancel.py`
- `aether/tests/gateway/test_run_params.py`
- `aether/tests/gateway/test_service_adapter_compat.py`
- `aether/tests/services/test_agent_run_service.py`

Cover:

- `agent.run` response fields remain unchanged
- event type names and fields remain unchanged
- text, reasoning, silent progress, status, loop-state, iteration, tool,
  token-usage, done, cancelled, and error events map correctly
- permission requests still flow through the gateway prompter bridge
- tool permission prompts still flow through the gateway permission bridge
- `agent.cancel` returns `{"ok": True}` and aborts the correct run
- prompt disconnect behavior matches current gateway behavior
- run already active maps to existing error code/data
- result persistence remains identical

## Migration Notes

- Keep the old helper functions until tests pass, then delete or move them
  deliberately.
- Do not combine this with low-risk handler migrations.
- Compare event payloads against golden fixtures created in PR 19.5.
- If a compatibility mismatch appears, prefer fixing the adapter mapping over
  changing service contracts or TUI expectations.

## Risks

- Streaming order and sequence counters are fragile.
- Permission and approval bridges are interactive and easy to regress.
- Running runs are long-handler tasks; do not block the dispatcher or leak
  background runs.
- Error behavior currently surfaces some failures as run results rather than
  worker crashes. Preserve that distinction.

## Non-Goals

- Do not change `AgentEngine`.
- Do not change TUI event fields.
- Do not introduce HTTP/SSE/WebSocket endpoints.
- Do not move gateway prompter UI behavior into services.
- Do not change provider streaming implementation.

## Acceptance

- `agent_methods.py` is a JSON-RPC adapter over `AgentRunService`.
- TUI streaming, permission prompts, cancellation, and final responses are
  wire-compatible with current behavior.
- Service tests own run lifecycle behavior.
- Gateway tests own event and response compatibility.
