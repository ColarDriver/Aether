# PR 19.10 - Acceptance and Hardening

## Goal

Close Sprint 19 with regression tests, import-boundary tests, compatibility
fixtures, and documentation proving the service layer is stable.

## Current Problem

Service extraction is only successful if behavior remains stable while the
business boundary moves. Without final hardening, Aether could end with services
that exist but are not consistently used, handlers that still duplicate logic,
or future adapters that still depend on gateway internals.

## Changes

Add final acceptance checks:

- service import boundary guard
- service public export guard
- gateway adapter compatibility fixtures
- agent event golden fixture comparison
- CLI no-gateway-handler-import guard
- documentation map from old gateway responsibilities to new services

Audit service usage:

- `session_methods.py` uses `SessionService`
- `prefs_methods.py` uses `PrefsService`
- `providers_methods.py` uses provider/auth/model services
- `tools_methods.py` uses tool/skill/diagnostic/health services as applicable
- `agent_methods.py` uses `AgentRunService`
- `commands_methods.py` may remain catalog-only

Document known intentional exceptions:

- gateway protocol serialization stays in gateway
- gateway prompter bridge stays in gateway
- command catalog may stay in gateway if it has no reusable business logic
- `aether/services/compact` keeps its existing design

Decide follow-up cleanup:

- whether to move pure session persistence out of `aether.cli.sessions`
- whether to move run handle registry out of `aether.gateway.run_handle`
- whether to add production HTTP/WebSocket adapters
- whether to add CLI commands for serviceized surfaces

## Tests

Run:

- `python -m pytest aether/tests/services`
- `python -m pytest aether/tests/gateway`
- `python -m pytest aether/tests/cli`
- `python -m pytest aether/tests/agents`
- `python -m pytest aether/tests/tools`
- `uv run pyright aether/services aether/gateway/handlers aether/cli`

Add or update:

- `aether/tests/services/test_service_import_boundaries.py`
- `aether/tests/services/test_service_exports.py`
- `aether/tests/gateway/test_service_adapter_compat.py`
- `aether/tests/gateway/test_agent_run_event_compat.py`
- `aether/tests/cli/test_cli_service_boundaries.py`

Manual checks:

- TUI starts normally.
- Command catalog loads.
- Session create/list/resume works.
- `/model` lists models and persists selection.
- Agent run streams text and reasoning.
- Tool events render.
- Permission modal still appears and resolves.
- `agent.cancel` cancels active run.
- Tools/skills/status surfaces still work.

## Migration Notes

- This PR should mainly harden and document. Avoid new behavior unless required
  to close a regression.
- If a handler still contains business logic, either move it to a service or
  document why it is transport-only.
- Keep acceptance evidence in `99_acceptance_matrix.md`.

## Risks

- Broad final test suites may expose unrelated failures. Separate Sprint 19
  regressions from pre-existing failures.
- Import guard false positives can slow development. Keep exceptions explicit
  and reviewed.

## Non-Goals

- Do not add new product features.
- Do not change gateway schemas during hardening.
- Do not expand service contracts unless a compatibility bug requires it.
- Do not implement a Web server.

## Acceptance

- All Sprint 19 service packages are import-safe and publicly exported.
- Gateway handlers are adapters over services or documented catalog-only
  exceptions.
- Existing TUI/gateway/CLI behavior is preserved by tests.
- Future CLI/Web adapters can reuse services without gateway internals.
- `99_acceptance_matrix.md` records final automated and manual evidence.
