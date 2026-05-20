# PR 19.1 - Service Layer Contract and Boundaries

## Goal

Create the shared service-layer skeleton, common contracts, common errors, and
import guardrails before any gateway handler is migrated.

## Current Problem

Gateway handlers currently mix three concerns:

- JSON-RPC parameter validation and wire serialization.
- Business behavior such as session persistence, provider discovery, model
  readiness, run lifecycle, and tool/skill catalog shaping.
- TUI-specific event compatibility.

If service extraction starts without guardrails, the new service packages can
accidentally import gateway schemas or UI concepts and simply move the coupling
instead of removing it.

## Changes

Add service skeleton packages:

```text
aether/services/common/
aether/services/sessions/
aether/services/config/
aether/services/providers/
aether/services/tools/
aether/services/skills/
aether/services/diagnostics/
aether/services/health/
aether/services/runs/
```

Each package must include:

- `contracts.py` for transport-neutral request/result/event contracts.
- `service.py` for business operations when applicable.
- `__init__.py` for stable public exports.

Add `aether/services/common/errors.py`:

- `ServiceError`
- `ServiceValidationError`
- `ServiceNotFoundError`
- `ServiceConflictError`
- `ServiceUnavailableError`
- `ServiceExecutionError`

Error contracts must include a stable `code: str`, human-readable `message`,
and optional public-safe `details: dict[str, object]`. They must not embed
`GatewayError` or gateway protocol error codes. Gateway adapters map service
errors to existing gateway errors.

Add service import rules:

- Services may import `aether.runtime`, `aether.config`, `aether.models`,
  `aether.tools`, `aether.agents`, and neutral persistence helpers.
- Services must not import `aether.gateway.handlers`,
  `aether.gateway.protocol`, TUI/Ink/React code, HTTP/WebSocket server code, or
  CLI parser modules.
- Services may temporarily call pure helper modules under `aether.cli` only
  when those modules already own storage behavior and do not import parser/UI
  code. Any such dependency must be documented and isolated behind the service.

Add an import guard test:

- `aether/tests/services/test_service_import_boundaries.py`

The guard should walk `aether/services/**/*.py` and fail if a service imports
forbidden gateway/UI/transport modules.

Add service package export tests:

- `aether/tests/services/test_service_exports.py`

This should import each public service package and verify the expected service
and contract names are exported.

## Tests

Add:

- `python -m pytest aether/tests/services/test_service_import_boundaries.py`
- `python -m pytest aether/tests/services/test_service_exports.py`
- `uv run pyright aether/services`

Cover:

- Forbidden imports are detected.
- Service packages import without side effects.
- Existing `aether/services/compact` still imports through its current public
  exports.
- No gateway handler imports are introduced into `aether/services`.

## Migration Notes

- This PR should not change gateway behavior.
- This PR can add empty service skeletons and common error types only.
- If a later PR needs a new common error, add it in that PR with tests.
- Keep service exceptions separate from gateway protocol exceptions.

## Risks

- Overly strict import guards can block legitimate runtime dependencies. Keep
  the deny-list focused on transport/UI modules instead of blocking all gateway
  package imports blindly.
- Too many empty skeleton files can create false confidence. Every later PR must
  add service behavior and tests before migrating a handler.

## Non-Goals

- Do not migrate gateway handlers.
- Do not implement session/provider/run business logic yet.
- Do not change RPC names, request fields, response fields, or TUI event shapes.
- Do not define HTTP routes, WebSocket frames, or SSE frames.
- Do not rewrite or move `aether/services/compact`.

## Acceptance

- Service packages have a consistent structure.
- Common service errors exist and are gateway-independent.
- Static tests prevent services from importing gateway handlers or UI/transport
  code.
- Reviewers can enforce the service/adapter boundary before behavior moves.
