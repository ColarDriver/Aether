# PR20.2 - REST API Services

## Goal

Expose Aether's service layer through REST endpoints for the browser console.
This PR turns the Python web backend into a useful local API without yet running
agent turns over WebSocket.

## Current Problem

The TUI gets session, provider, model, tool, skill, and health data through
gateway RPC methods. A web app should not call gateway handlers directly, and it
should not duplicate the business logic that Sprint 19 moved into
`aether/services/*`.

Hermes has broad REST coverage, but many endpoints map to features Aether does
not own. This PR implements only service-backed Aether resources.

## Required Routes

### Sessions

- `GET /api/sessions?limit=50`
  - Calls `SessionService.list(limit=limit)`.
  - Returns `{ "sessions": [...] }`.
- `POST /api/sessions`
  - Body: `{ "provider": "...", "model": "...", "base_url": null, "system_prompt": null }`.
  - Calls `SessionService.create(...)`.
  - Returns created `SessionInfo`.
- `GET /api/sessions/current`
  - Calls `SessionService.current()`.
  - Returns `{ "session": null }` or `{ "session": ..., "messages": [] }`.
- `POST /api/sessions/{session_id}/resume`
  - Calls `SessionService.resume(...)`.
  - Returns session info plus transcript messages.
- `DELETE /api/sessions/{session_id}`
  - Calls `SessionService.delete(...)`.
  - Returns 204 when deleted or 404 when missing.
- `GET /api/sessions/{session_id}/messages`
  - Calls `SessionService.transcript(...)`.
  - Returns `{ "session_id": "...", "messages": [...] }`.

### Config and Preferences

- `GET /api/config`
  - Calls `ConfigService.effective()`.
- `GET /api/config/paths`
  - Calls config path service methods if available.
- `GET /api/prefs`
  - Calls `PrefsService` read APIs.

Only expose public-safe values. Do not return raw API keys.

### Providers and Models

- `GET /api/providers`
  - Calls `ProviderService.list_providers()`.
- `GET /api/providers/current`
  - Calls `ProviderService.runtime_current(...)`.
- `GET /api/providers/{provider}/models?base_url=...`
  - Calls `ProviderService.list_models(...)`.
- `POST /api/model/select`
  - Calls `ModelSelectionService.select(...)`.
  - Updates current session when requested.
- `GET /api/model/auxiliary`
  - Calls `ProviderService.auxiliary_slots()`.

### Tools, Skills, Diagnostics, Health

- `GET /api/tools`
  - Calls `ToolService.list_tools()`.
- `GET /api/tools/groups`
  - Calls `ToolService.list_groups()`.
- `GET /api/skills`
  - Calls `SkillService.list_skills()`.
- `GET /api/skills/{name}`
  - Calls `SkillService.get_skill(...)`.
- `GET /api/diagnostics`
  - Calls `DiagnosticsService.status()`.
- `GET /api/health`
  - Already added in PR20.1; keep stable.

### Runs Read Model

- `GET /api/runs/{run_or_session_id}`
  - Calls `AgentRunService.status(...)`.
  - Returns snapshot or 404.
- `POST /api/runs/{session_id}/cancel`
  - Calls `AgentRunService.cancel(...)`.
  - Returns `{ "cancelled": true }`.

This is a read/cancel surface only. Starting streamed runs belongs to PR20.3.

## Serialization Rules

- Put shared conversion helpers in `aether/web/serializers.py`.
- Serialize dataclasses through `dataclasses.asdict`.
- Serialize `StrEnum` and normal enums to `.value`.
- Omit non-JSON-safe values.
- Preserve snake_case keys to match Python services and existing gateway shapes.
- Keep frontend-specific computed fields in the frontend unless they are needed
  by multiple clients.

## Error Mapping

`aether/web/errors.py` should map service errors consistently:

- `ServiceValidationError` -> 400
- `ServiceNotFoundError` -> 404
- `ServiceConflictError` -> 409
- generic `ServiceError` -> 500 with public message

Responses should use:

```json
{
  "error": {
    "code": "SERVICE_CODE",
    "message": "...",
    "details": {}
  }
}
```

Do not leak stack traces or secret values.

## Tests

Add `aether/tests/web/test_web_rest_services.py`:

- Sessions list/create/current/resume/delete/messages use `SessionService`.
- Provider list/current/models serialize catalog and discovery metadata.
- Model select returns readiness and missing credential data.
- Tools and groups render enabled tools.
- Skills list/detail handle missing skill as 404.
- Diagnostics and health render public-safe status.
- Service exceptions map to expected HTTP statuses.
- Auth middleware protects non-public endpoints.

## Non-Goals

- Do not implement WebSocket streaming.
- Do not add full-text session search until Aether owns a search/index service.
- Do not add Hermes plugin, cron, profile, theme, analytics, or log routes.
- Do not mutate raw `.env` secrets through the web API.

## Acceptance

- REST routes are thin adapters over `aether/services/*`.
- Route tests pass with real or fake services.
- Existing gateway tests still prove TUI RPC compatibility.
- The API is sufficient for the initial React shell to load sessions, provider
  state, tools, skills, and health.
