# PR 19.9 - CLI and Web Adapter Readiness

## Goal

Prove the new service layer is useful outside the stdio gateway by adding
adapter-readiness checks and optionally thin CLI commands that call services
directly.

## Current Problem

After gateway handlers migrate to services, Aether still needs proof that CLI
and future Web adapters can reuse services without importing gateway handlers or
copying business logic.

## Changes

Add CLI readiness rules:

- CLI commands call services directly.
- CLI commands format output for terminal users.
- CLI commands do not import `aether.gateway.handlers` or JSON-RPC schemas.
- CLI commands do not duplicate business rules already owned by services.

Candidate thin CLI commands if product scope allows:

- `aether sessions list`
- `aether config show`
- `aether providers`
- `aether doctor`

If adding commands is too broad for this sprint, add internal CLI service tests
and document command follow-ups instead.

Add future Web adapter principles:

- HTTP handlers call services and map service results to HTTP responses.
- WebSocket/SSE stream endpoints forward `AgentRunService` events.
- Web adapter code maps service contracts to HTTP/stream frames.
- Web adapter code does not import gateway handlers.
- Web adapter code does not inspect runtime internals for display data.

Add a service consumer example module only if useful:

- `aether/cli/service_adapters.py` or command-local helper functions

Keep it thin. It should not become a second service layer.

## Tests

Run:

- `python -m pytest aether/tests/cli`
- `python -m pytest aether/tests/gateway`
- `python -m pytest aether/tests/services`

Add targeted tests:

- CLI modules do not import gateway handlers.
- CLI service consumers can list sessions/providers/tools through services.
- Future Web adapter documentation references service contracts rather than
  gateway handlers.
- Service contracts are serializable to JSON-compatible data without gateway
  protocol classes.

Manual acceptance:

- Start the TUI and verify command catalog loads.
- Create/list/resume sessions through the existing TUI path.
- Use `/model` to list and switch models.
- Start an agent run and observe streaming output.
- Cancel a run and verify consistent cancellation state.
- List tools and skills through existing surfaces.
- Inspect health/status output through the chosen adapter.

## Migration Notes

- Adding CLI commands is optional; adapter-readiness tests are required.
- Do not make CLI depend on gateway handlers for serviceized behavior.
- Do not introduce an HTTP server just to prove Web readiness.

## Risks

- CLI command additions can expand scope. Keep commands thin or defer them.
- Web readiness can become vague. Tie it to concrete service contracts and
  import-boundary tests.

## Non-Goals

- Do not implement a production Web UI.
- Do not add a production HTTP server.
- Do not require all candidate CLI commands if service migration is still
  stabilizing.
- Do not change existing TUI workflows.
- Do not introduce a second service layer for Web.

## Acceptance

- CLI code can consume services without gateway imports.
- Future Web adapter rules are concrete and tied to service contracts.
- Existing TUI/gateway workflows still pass.
- No adapter owns reusable business behavior.

## Implementation Evidence

- `aether/tests/cli/test_cli_service_boundaries.py` guards CLI modules against
  importing `aether.gateway.handlers`.
- The same test proves a non-gateway consumer can create/list sessions and read
  providers/tools through services directly.
- Future HTTP handlers should mirror the gateway adapters: validate transport
  params, call services, serialize dataclasses to HTTP response bodies.
- Future WebSocket/SSE run streams should consume `AgentRunService` with a
  `RunEventSink`, then frame service events at the transport edge. They should
  not import `aether.gateway.handlers.agent_methods`.
