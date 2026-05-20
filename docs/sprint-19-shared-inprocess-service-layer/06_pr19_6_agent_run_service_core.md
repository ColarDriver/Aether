# PR 19.6 - Agent Run Service Core

## Goal

Move agent run lifecycle into `AgentRunService` while keeping gateway wire
behavior unchanged.

## Current Problem

`agent_methods.py` currently does all of the following:

- parse run params
- resolve and validate sessions
- create and register `RunHandle`
- build provider, engine config, tool registry, skill catalog, agent type
  registry, task store, and subagent manager
- bridge engine stream callbacks into gateway events
- bridge tool/iteration middleware into gateway events
- wire approval and permission prompters
- run `AgentEngine.run_loop`
- persist results back into the session record
- map final result, cancellation, and errors into gateway responses

Those are reusable run lifecycle concerns. Future HTTP/WebSocket/SSE adapters
would have to copy the same logic unless it lives in a service.

## Changes

Add `AgentRunService` under `aether/services/runs/service.py`:

- `start(request: AgentRunRequest, sink: RunEventSink | None = None) -> AgentRunResult`
- `cancel(request: AgentRunCancelRequest) -> bool`
- `status(run_id_or_session_id: str) -> AgentRunSnapshot | None`
- `final_result(run_id_or_session_id: str) -> AgentRunResult | None`

Add `RunEventSink` protocol:

- `emit(event: RunEvent) -> None`

Add `aether/services/runs/builder.py`:

- provider builder from service/provider contracts
- engine config builder
- tool registry builder hook
- skill catalog builder
- agent type registry builder
- task store builder
- subagent manager builder
- `EngineRequest` builder

Service responsibilities:

- session lookup through `SessionService`
- provider/model runtime resolution through provider services
- run handle registry ownership or a neutral wrapper around current
  `aether.gateway.run_handle`
- in-process event emission using PR 19.5 service events
- approval/permission prompt bridge injection via adapter-provided prompter
  protocols
- final result persistence through `SessionService`
- cancellation with the existing interrupt signal semantics

Adapter responsibilities:

- parse JSON-RPC params
- provide gateway prompter implementations
- map service events to gateway events
- map service result/errors to existing gateway response shape

Keep the old gateway handler path working during this PR if needed. A safe
intermediate shape is to have `agent_methods.py` call the new service internally
behind the existing wire adapter after service tests pass.

## Tests

Add:

- `aether/tests/services/test_agent_run_service.py`
- `aether/tests/services/test_agent_run_service_events.py`
- `aether/tests/services/test_agent_run_service_cancel.py`

Cover:

- run loads the expected session
- missing session raises `ServiceNotFoundError`
- session with missing provider/model raises validation/service error
- fake provider run emits `RunStarted` before deltas and `RunFinished` last
- assistant and reasoning deltas preserve ordering
- silent progress events preserve sequence semantics
- tool start/finish events preserve identifiers and metadata
- cancellation aborts the run handle interrupt signal
- final result can be read after completion
- result persistence updates session messages, model/provider/base URL, and
  system prompt exactly like current `agent_methods._persist_result`
- prompt disconnect maps to a service failure without crashing the worker

## Migration Notes

- Use fake providers and fake sinks in service tests.
- Do not require real gateway transport in service tests.
- Keep prompter protocols generic. Gateway prompters implement them later.
- If `RunHandle` remains in `aether.gateway.run_handle` temporarily, wrap it
  behind a service-owned interface and move it in a later cleanup PR.

## Risks

- Run lifecycle is the highest-risk part of Sprint 19. Do not combine this PR
  with gateway event shape changes.
- Threading and cancellation behavior can regress if `RunHandle` ownership is
  moved too aggressively.
- AgentEngine construction has many defaults. Builder tests should pin the
  current config toggles used by gateway runs.

## Non-Goals

- Do not change `AgentEngine`.
- Do not change provider streaming internals.
- Do not change gateway event fields.
- Do not add HTTP/SSE/WebSocket infrastructure.
- Do not move permission UI logic into the service.

## Acceptance

- AgentRunService can run an agent turn with fake dependencies.
- Service events follow the PR 19.5 contract.
- Cancellation and final-result semantics are service-owned.
- Existing gateway behavior can be preserved by an adapter over the service.
