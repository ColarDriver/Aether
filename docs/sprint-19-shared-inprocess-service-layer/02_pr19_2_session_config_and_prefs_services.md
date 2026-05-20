# PR 19.2 - Session, Config, and Prefs Services

## Goal

Move low-risk state capabilities behind services: session lifecycle, session
read models, effective config reads, environment path reporting, and scoped
preferences.

## Current Problem

`session_methods.py` converts persisted `SessionRecord` objects into gateway
wire shapes and directly calls `aether.cli.sessions`. `prefs_methods.py`
directly calls `aether.cli.prefs`. Future CLI/Web surfaces would either import
gateway handlers or duplicate the conversion logic.

## Changes

Add `aether/services/sessions/contracts.py`:

- `SessionInfo`
- `TranscriptMessage`
- `TranscriptToolCall`
- `SessionCreateRequest`
- `SessionUpdateRequest`
- `SessionResumeRequest`
- `SessionDeleteRequest`
- `SessionRenameRequest`
- `SessionExportRequest`
- `SessionExportResult`
- `SessionListResult`
- `SessionCurrentResult`

Add `SessionService`:

- `create(request: SessionCreateRequest) -> SessionInfo`
- `list(limit: int | None = None) -> SessionListResult`
- `resume(session_id_or_prefix: str) -> SessionCurrentResult`
- `current() -> SessionCurrentResult | None`
- `update(request: SessionUpdateRequest) -> SessionInfo`
- `delete(request: SessionDeleteRequest) -> bool`
- `rename(request: SessionRenameRequest) -> SessionInfo`
- `export(request: SessionExportRequest) -> SessionExportResult`
- `transcript(session_id_or_prefix: str) -> list[TranscriptMessage]`

SessionService must preserve current behavior:

- Reuse existing session persistence from `aether.cli.sessions` for the first
  implementation.
- Preserve current transcript normalization behavior from
  `session_methods._to_transcript`, including malformed tool-call JSON fallback
  under `{"__raw__": ...}`.
- Preserve plan-mode mode resolution:
  `runtime.session.session_state.get_mode(session_id)` wins over stored
  `record.mode` when active.
- New session creation clears in-process plan mode and plan artifact for the
  explicit or generated session id.
- Current session tracking remains compatible with
  `aether.gateway.handlers.state` until a neutral current-session holder exists.

Add `aether/services/config/contracts.py`:

- `EffectiveConfig`
- `ConfigPaths`
- `EnvironmentPathStatus`
- `ConfigDefaults`

Add `ConfigService`:

- `effective() -> EffectiveConfig`
- `paths() -> ConfigPaths`
- `defaults() -> ConfigDefaults`
- `environment_paths() -> list[EnvironmentPathStatus]`

This service is read-only in Sprint 19. It does not implement a new config file
format.

Add `PrefsService` in `aether/services/config/prefs.py`:

- `get(key: str) -> object | None`
- `set(key: str, value: object) -> None`
- `delete(key: str) -> bool`
- `all() -> dict[str, object]`
- `get_last_model(provider: str) -> str | None`
- `set_last_model(provider: str, model: str) -> None`

PrefsService must reuse current persistence from `aether.cli.prefs` and
preserve the existing scoped key behavior.

Gateway migration is deferred to PR 19.7. This PR should add service behavior
and service tests first.

## Tests

Add:

- `aether/tests/services/test_session_service.py`
- `aether/tests/services/test_config_service.py`
- `aether/tests/services/test_prefs_service.py`

Cover:

- session create/list/current/resume/update/delete/rename/export
- explicit session id creation clears stale plan mode and plan artifact
- transcript conversion preserves assistant tool calls, tool messages, metadata,
  errors, and malformed JSON fallback
- session prefix resolution matches current gateway behavior
- list limit semantics match `session.list`
- prefs get/set/delete/all round trip
- last model preference round trip by provider
- effective config/defaults/path read models contain public-safe values
- env path reporting redacts sensitive values and does not expose API keys

## Migration Notes

- Do not migrate `session_methods.py` or `prefs_methods.py` in this PR.
- Add service tests that describe current behavior before changing handlers.
- Keep any dependency on `aether.cli.sessions` and `aether.cli.prefs` isolated
  inside services so future storage relocation is one internal refactor.
- If `SessionInfo` overlaps gateway Pydantic schemas, keep it as a separate
  service dataclass and map it in the gateway adapter later.

## Risks

- Session conversion has many compatibility details; missing metadata or
  tool-call normalization will break TUI transcript rendering.
- Current-session state is currently gateway-owned. Moving it too early could
  change behavior. Keep the gateway state bridge until PR 19.7.
- ConfigService can become too broad. Keep Sprint 19 read-only.

## Non-Goals

- Do not change session storage format.
- Do not change session RPC names or response fields.
- Do not make TUI read session files directly.
- Do not implement full config-file migration.
- Do not implement provider switching beyond prefs support.
- Do not migrate gateway handlers in this PR.

## Acceptance

- Session, config, and prefs business behavior is available through services.
- Service tests prove parity with existing session/prefs behavior.
- Gateway behavior remains unchanged until the adapter migration PR.
- Future CLI commands can reuse services without importing gateway handlers.
