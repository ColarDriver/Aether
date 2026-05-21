# PR20.6 - Settings, Models, Skills, Tools, and Health Views

## Goal

Build the browser console views around chat: provider/model selection, auxiliary
slots, tool catalog, skill catalog, config/preferences, diagnostics, docs, workspace, analytics, and health.

## Current Problem

The web console must be more than a chat surface. A user needs to understand the
runtime state that affects agent behavior: provider readiness, selected model,
auxiliary model slots, enabled tools, skills, diagnostics, local documentation, workspace files, analytics, and configuration.

Hermes has many dashboard pages. Aether should implement the subset backed by
existing services and defer Hermes-only product areas until Aether has matching
service ownership.

## Required Views

### Provider and Model View

Components:

- `ProviderSettings.tsx`
- `ProviderList.tsx`
- `ModelPicker.tsx`
- `AuxiliarySlotsPanel.tsx`
- `CredentialStatusBadge.tsx`

Behavior:

- Load providers from `GET /api/providers`.
- Load current runtime from `GET /api/providers/current`.
- Load model lists with discovery metadata from
  `GET /api/providers/{provider}/models`.
- Select model through `POST /api/model/select`.
- Show missing credential state without exposing secret values.
- Show auxiliary slots from `GET /api/model/auxiliary`.

### Tools View

Components:

- `ToolsView.tsx`
- `ToolGroup.tsx`
- `ToolDetail.tsx`

Behavior:

- Load `GET /api/tools/groups`.
- Group by filesystem, shell, web, subagent, interaction, planning, skills,
  diagnostics, memory, other.
- Show name, description, required params, and enabled status.

### Skills View

Components:

- `SkillsView.tsx`
- `SkillList.tsx`
- `SkillDetail.tsx`

Behavior:

- Load `GET /api/skills`.
- Show name, description, when-to-use, source, path, and version.
- Detail route/panel loads `GET /api/skills/{name}` if needed.

### Config and Preferences View

Components:

- `SettingsView.tsx`
- `ConfigSummary.tsx`
- `PrefsSummary.tsx`

Behavior:

- Load `GET /api/config`.
- Load `GET /api/config/paths`.
- Load `GET /api/prefs`.
- Render read-only summary first.
- Mutating config in-browser is deferred unless service contracts already
  support safe write operations.

### Health and Diagnostics View

Components:

- `DiagnosticsView.tsx`
- `HealthPanel.tsx`
- `ServiceStatusList.tsx`

Behavior:

- Load `GET /api/health`.
- Load `GET /api/diagnostics`.
- Show provider auth readiness, diagnostics enabled state, Python runtime, and
  service availability.
- Avoid stack traces and raw secret values.

### Docs View

Components:

- `DocsView.tsx`
- shared `MarkdownRenderer.tsx`

Behavior:

- Load `GET /api/docs` for the markdown index.
- Read selected files through `GET /api/docs/{doc_path}`.
- Render local markdown content without iframe or remote documentation coupling.
- Keep path display visible so implementation plans can be referenced precisely.

### Workspace View

Components:

- `WorkspaceView.tsx`

Behavior:

- Load `GET /api/workspace/tree` for safe project-root directory browsing.
- Read selected files through `GET /api/workspace/file`.
- Search paths through `GET /api/workspace/search`.
- Render markdown with the shared markdown renderer and text/code files in a stable preview pane.
- Never expose excluded directories or raw `.env` secret files.

### Analytics View

Components:

- `AnalyticsView.tsx`

Behavior:

- Load `GET /api/analytics`.
- Show session, message, tool-call, model, daily token, and top-session summaries.
- Aggregate only local session metadata exposed by Aether services.

## Navigation

Add app-level navigation:

- Chat
- Sessions
- Models
- Tools
- Skills
- Diagnostics
- Logs
- Analytics
- Docs
- Workspace
- Environment
- Settings

Use tabs or sidebar items; avoid nested card-heavy layouts. The active route
should be visible and keyboard-accessible.

## Tests

Frontend tests:

- Provider view renders current runtime and missing credentials.
- Model picker calls model-select API.
- Auxiliary slots render inherited/custom state.
- Tools view groups tools and exposes searchable detail/schema inspection.
- Skills view renders details and missing state.
- Diagnostics view renders service statuses.
- Docs view renders markdown index and selected content.
- Workspace view browses directories, previews files, and searches paths.
- Analytics view renders usage summary and model/session tables.
- Settings view does not show secret values.

Backend tests from PR20.2 should already cover the data routes.

## Non-Goals

- Do not implement secret editing in this PR.
- Do not implement Hermes plugin hub, dashboard themes, cron, profiles, or PTY.
- Do not add model benchmark dashboards.

## Acceptance

- User can inspect and change the active model/provider from the browser.
- User can inspect tools, skills, health, diagnostics, logs, docs, workspace files, and analytics.
- No browser view exposes raw credentials.
- Web UI stays operational and compact across desktop and narrow widths.
