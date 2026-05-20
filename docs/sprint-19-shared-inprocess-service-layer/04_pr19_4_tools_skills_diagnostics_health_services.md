# PR 19.4 - Tools, Skills, Diagnostics, and Health Services

## Goal

Move read-only display models for tools, skills, diagnostics, and health into
shared services before gateway handlers are migrated.

## Current Problem

Tool catalogs, skill discovery, diagnostics, and health/status data are useful
to the TUI, CLI, and future Web surfaces. Today those surfaces would either call
gateway handlers or inspect runtime internals directly. That couples display
surfaces to implementation details and makes future parity work brittle.

## Changes

Add `aether/services/tools/contracts.py`:

- `ToolSummary`
- `ToolCatalog`
- `ToolGroup`
- `ToolAvailability`

Add `ToolService`:

- `list_tools() -> ToolCatalog`
- `list_groups() -> list[ToolGroup]`
- `get_tool(name: str) -> ToolSummary | None`

ToolService must preserve current built-in descriptor behavior from
`aether.tools.registry.build_default_tool_registry` and current gateway
`tools.list` output compatibility after adapter migration. It should not
implement enable/disable state unless backed by existing config.

Add `aether/services/skills/contracts.py`:

- `SkillSummary`
- `SkillCatalogResult`
- `SkillSource`

Add `SkillService`:

- `list_skills() -> SkillCatalogResult`
- `get_skill(name: str) -> SkillSummary | None`

SkillService must reuse `aether.runtime.tools.skill_catalog` and
`build_default_skill_catalog`. It should expose only public-safe source
metadata.

Add `aether/services/diagnostics/contracts.py`:

- `DiagnosticSummary`
- `DiagnosticFileSummary`
- `DiagnosticsStatus`
- `LspStatus`

Add `DiagnosticsService`:

- `status() -> DiagnosticsStatus`
- `recent() -> list[DiagnosticSummary]`
- `lsp_status() -> LspStatus`

DiagnosticsService must handle missing trackers gracefully and must not expose
live runtime objects.

Add `aether/services/health/contracts.py`:

- `HealthStatus`
- `ServiceStatus`
- `RuntimeStatus`

Add `HealthService`:

- `status() -> HealthStatus`
- aggregate Python/runtime version
- aggregate provider readiness through `AuthService`
- aggregate diagnostics readiness through `DiagnosticsService`
- report service availability without requiring gateway transport state

Gateway migration is deferred to PR 19.7.

## Tests

Add:

- `aether/tests/services/test_tools_service.py`
- `aether/tests/services/test_skills_service.py`
- `aether/tests/services/test_diagnostics_service.py`
- `aether/tests/services/test_health_service.py`

Cover:

- tool list includes existing built-in tool descriptors
- tool grouping is deterministic
- tool summaries do not expose executor instances
- skill list matches current skill discovery behavior
- missing optional skill metadata is tolerated
- diagnostics status works when no tracker is configured
- health includes runtime version and provider/auth readiness
- health output is public-safe and does not contain secrets

## Migration Notes

- Do not change `tools_methods.py` in this PR.
- Do not add tool enable/disable state unless it already exists in config.
- Keep read models compact; detailed runtime objects stay inside runtime
  packages.
- HealthService should aggregate other services; it should not duplicate their
  business logic.

## Risks

- Tool descriptors may include fields intended for model schemas, not display.
  ToolService should shape a display-safe summary.
- Skill discovery can touch filesystem paths. Return safe path/source metadata
  only, and avoid loading unnecessary file contents.
- HealthService can become a dumping ground. Keep it to high-level readiness.

## Non-Goals

- Do not implement full `tools enable/disable`.
- Do not implement skill install/update/audit.
- Do not add a plugin marketplace.
- Do not require an LSP tracker when none exists.
- Do not expose runtime-private objects through service contracts.
- Do not migrate gateway handlers in this PR.

## Acceptance

- Tool, skill, diagnostic, and health read behavior is available through
  services.
- Service tests cover existing display behavior and missing-runtime fallbacks.
- Gateway behavior remains unchanged until PR 19.7.
- Future TUI/Web/CLI display surfaces can use services instead of scanning
  runtime internals.
