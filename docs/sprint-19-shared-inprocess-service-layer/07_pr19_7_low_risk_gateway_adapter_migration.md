# PR 19.7 - Low-Risk Gateway Adapter Migration

## Goal

Migrate low-risk gateway handlers to services while preserving every existing
RPC method name, request field, response field, and error mapping.

## Current Problem

Gateway handlers for sessions, prefs, providers, and tools currently contain
business shaping logic. Once services exist and are tested, these handlers
should become thin adapters so gateway is no longer the source of truth.

## Changes

Migrate in this order:

1. `prefs_methods.py` -> `PrefsService`
2. `session_methods.py` -> `SessionService`
3. `tools_methods.py` -> `ToolService`
4. `providers_methods.py` -> `ProviderService`, `AuthService`,
   `ModelSelectionService`
5. optional status/health handler additions only if an equivalent handler
   already exists

Adapter responsibilities:

- validate JSON-RPC params
- call the appropriate service
- convert service dataclasses into existing Pydantic gateway schemas or dicts
- map service errors to existing `GatewayError` codes
- keep existing wire schemas unchanged

Handler code must not duplicate service business rules after migration. If a
handler needs a behavior branch, first decide whether it is transport-specific
validation/serialization or missing service behavior.

## Tests

Update or add:

- `aether/tests/gateway/test_prefs_methods.py`
- `aether/tests/gateway/test_session_methods.py`
- `aether/tests/gateway/test_tools_methods.py`
- `aether/tests/gateway/test_providers_methods.py`
- `aether/tests/gateway/test_service_adapter_compat.py`

Cover:

- session RPC response compatibility for create/list/resume/update/delete/current
- transcript compatibility including tool-call normalization
- prefs RPC get/set/all compatibility
- providers list/models/runtime/credentials/auxiliary-slots compatibility
- tools list compatibility
- common service errors map to existing gateway error codes
- gateway handlers no longer import low-level persistence helpers directly when
  a service owns that behavior

## Migration Notes

- Migrate one handler file per commit.
- Run `python -m pytest aether/tests/gateway/<handler-test>.py` after each
  file.
- Keep `commands_methods.py` unmigrated unless there is a clear service
  dependency; it is mostly a static command catalog.
- Do not touch `agent_methods.py` in this PR.

## Risks

- Compatibility regressions are easy because the TUI consumes exact field
  names. Use before/after response fixtures for each migrated handler.
- Provider discovery has network fallback paths. Gateway tests should mock
  service outputs rather than make live network calls.

## Non-Goals

- Do not migrate `agent_methods.py`.
- Do not change gateway protocol names.
- Do not remove gateway handlers.
- Do not require TUI to call services directly.
- Do not place TUI-only fallback or formatting logic in services.

## Acceptance

- Low-risk gateway handlers are thin adapters over services.
- Existing gateway tests pass.
- New adapter compatibility tests prove wire response parity.
- Business behavior is tested at the service layer.
