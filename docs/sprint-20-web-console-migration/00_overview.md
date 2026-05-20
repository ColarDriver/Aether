# Sprint 20 - Web Console Migration

## Background

Aether now has a TypeScript Ink TUI, a stdio JSON-RPC gateway, a transport-neutral
service layer under `aether/services/*`, provider transport abstractions,
credential/runtime helpers, auxiliary slots, and an `AgentRunService` that emits
service-level run events. What Aether does not yet have is a browser console.

The target work is to migrate the useful web-related implementation patterns from
`/workspace/hermes-agent` and `/workspace/cc-haha` into Aether:

- Frontend implementation in TypeScript.
- Backend implementation in Python.
- A clean branch and staged commits.
- Architecture that fits Aether rather than a direct copy of either reference.

Sprint 20 is the implementation sprint for that web console. It builds on Sprint
19: the web backend must call `aether/services/*` directly and must not route
through gateway handlers as a hidden business API.

## Reference Findings

### Hermes Agent

Hermes has the closest Python web backend reference:

- `hermes_cli/web_server.py` is a large FastAPI app.
- It serves a Vite React SPA and REST endpoints under `/api/*`.
- It includes local security controls: ephemeral session token injection, token
  middleware, localhost CORS, and Host-header validation.
- It exposes sessions, config, model info/options/set, skills, toolsets,
  dashboard themes/plugins, analytics, logs, PTY WebSocket, JSON-RPC WebSocket,
  event fan-out, and static SPA mounting.
- `tui_gateway/ws.py` shows how Hermes adapts an existing stdio JSON-RPC server
  onto WebSocket without changing method handlers.
- `web/src/lib/api.ts` and `web/src/lib/gatewayClient.ts` show browser clients
  for REST and JSON-RPC WebSocket surfaces.

Hermes is valuable for server shape, local-dashboard security, SPA mounting, and
browser API ergonomics. It is not a good direct copy target because its server is
monolithic and includes Hermes-specific plugins, cron jobs, profile management,
OAuth flows, analytics, and PTY embedding that Aether does not currently own.

### cc-haha

cc-haha has the strongest browser chat UX reference:

- `desktop/src/api/client.ts` has a small typed REST client with base URL,
  bearer token support, abort timeouts, and diagnostic reporting.
- `desktop/src/api/websocket.ts` manages per-session WebSocket connections,
  ping, reconnect, and pending send queues.
- `desktop/src/components/chat/*` contains mature chat components: message list,
  assistant/user messages, streaming indicator, permission dialog, tool call
  blocks, tool result blocks, code viewer, diff viewer, markdown renderer, and
  task/status surfaces.
- `desktop/src/components/layout/*` and `desktop/src/stores/*` provide useful
  patterns for sidebar navigation, status bar, tabs, session state, chat state,
  provider state, and UI state.

cc-haha is valuable for UI state and component behavior. Its backend is
TypeScript and should not be migrated as Aether backend code. Its frontend should
be treated as a UX/component reference rather than copied wholesale.

### Aether

Aether already has the runtime foundations that a web console should reuse:

- `aether/services/sessions` owns session lifecycle and transcript read models.
- `aether/services/config` and `aether/services/config/prefs.py` expose config
  and preference state.
- `aether/services/providers` exposes provider catalog, auth readiness, live
  model discovery, model selection, and auxiliary slots.
- `aether/services/tools`, `aether/services/skills`, `aether/services/health`,
  and `aether/services/diagnostics` expose read-only console data.
- `aether/services/runs` owns `AgentRunService`, run contracts, run snapshots,
  cancellation, and transport-neutral run events.
- `aether/gateway/handlers/prompter_bridge.py` is a useful approval/permission
  protocol reference, but web approval prompts should be implemented as a web
  transport bridge rather than importing gateway reverse-RPC internals.

## Goals

- Add a Python FastAPI web backend under `aether/web`.
- Add a standalone TypeScript browser app under `web/`.
- Reuse `aether/services/*` for all business behavior.
- Provide REST endpoints for status, sessions, config/prefs, providers/models,
  tools, skills, diagnostics, and run snapshots.
- Provide a browser run stream over WebSocket, with text/reasoning deltas, tool
  events, token usage, status updates, final results, errors, and cancellation.
- Provide web-native approval and tool-permission prompts so agent runs can ask
  the browser user for decisions.
- Preserve Aether's existing TUI and gateway behavior.
- Keep frontend UX dense, operational, and chat-first rather than a marketing
  dashboard.
- Serve the built SPA from the Python web backend for local use.
- Include automated tests and manual acceptance coverage.

## Non-Goals

- Do not copy Hermes-specific cron, profile, plugin hub, theme extension,
  OAuth provider install, analytics, or PTY dashboard features unless Aether
  first grows matching service ownership.
- Do not use a TypeScript backend for Aether.
- Do not make the web backend call gateway handlers for business behavior.
- Do not replace the existing TUI or stdio gateway.
- Do not implement remote multi-user auth, hosted accounts, or cloud sync.
- Do not expose secrets in REST responses.
- Do not add browser-only state into `aether/services/*`.

## Architecture

```text
Browser React app
  -> REST client            -> aether/web/routes/*.py -> aether/services/*
  -> run WebSocket client   -> aether/web/ws/runs.py  -> AgentRunService
  -> approval UI decisions  -> aether/web/prompts.py  -> engine prompter protocols

Existing TUI
  -> stdio JSON-RPC gateway -> aether/gateway/handlers -> aether/services/*
```

The web backend is an adapter layer. It owns HTTP/WebSocket framing, request
validation, JSON serialization, token/host checks, and static SPA serving. It
does not own session persistence, provider selection, model discovery, run
execution, tools, skills, health, or diagnostics behavior.

## Target Directory Shape

```text
aether/web/
  __init__.py
  app.py
  entry.py
  security.py
  serializers.py
  errors.py
  static.py
  routes/
    __init__.py
    health.py
    sessions.py
    config.py
    providers.py
    tools.py
    skills.py
    diagnostics.py
    runs.py
  ws/
    __init__.py
    runs.py
    prompts.py
    events.py

web/
  package.json
  tsconfig.json
  vite.config.ts
  index.html
  src/
    main.tsx
    App.tsx
    styles.css
    api/
      client.ts
      types.ts
      runSocket.ts
    stores/
      appStore.ts
      sessionStore.ts
      chatStore.ts
      providerStore.ts
    components/
      layout/
      chat/
      settings/
      shared/
```

## Public API Shape

Initial REST endpoints:

- `GET /api/status`
- `GET /api/health`
- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/current`
- `POST /api/sessions/{session_id}/resume`
- `DELETE /api/sessions/{session_id}`
- `GET /api/sessions/{session_id}/messages`
- `GET /api/config`
- `GET /api/prefs`
- `GET /api/providers`
- `GET /api/providers/current`
- `GET /api/providers/{provider}/models`
- `POST /api/model/select`
- `GET /api/model/auxiliary`
- `GET /api/tools`
- `GET /api/tools/groups`
- `GET /api/skills`
- `GET /api/diagnostics`
- `GET /api/runs/{run_or_session_id}`
- `POST /api/runs/{session_id}/cancel`

Initial WebSocket endpoint:

- `WS /api/runs/ws`

The WebSocket carries JSON messages rather than terminal/PTY output. It is the
browser equivalent of `AgentRunService` event streaming:

- client: `run.start`, `run.cancel`, `permission.respond`, `approval.respond`,
  `ping`
- server: `ready`, `run.started`, `assistant.delta`, `reasoning.delta`,
  `silent.progress`, `run.status`, `loop.state`, `iteration.started`,
  `iteration.finished`, `tool.started`, `tool.finished`, `token.usage`,
  `permission.requested`, `approval.requested`, `run.finished`, `run.failed`,
  `run.cancelled`, `error`, `pong`

## Frontend UX

The web app starts as a real console, not a landing page:

- Left sidebar: sessions, current provider/model, runtime status.
- Main center: chat transcript, streaming assistant output, tool/diff blocks,
  permission/approval surfaces.
- Bottom composer: model-aware input, run/stop controls, lightweight mode
  indicator.
- Right/secondary panels: tools, skills, provider/model settings, diagnostics.

The visual direction should be quiet and operational:

- Dense but readable spacing.
- Stable panels and scroll containers.
- No marketing hero sections.
- No decorative gradients/orbs.
- No terminal emulation unless a later sprint explicitly implements PTY mode.

## Security Rules

- Default bind should be loopback.
- All mutable `/api/*` and WebSocket traffic should require an ephemeral local
  web token unless explicitly disabled for test app creation.
- Accept local origins only by default.
- Validate Host headers for loopback binds.
- Redact credential values in all responses.
- Do not persist browser auth tokens in source-controlled files.

## Roadmap

| PR | File | Scope |
|---|---|---|
| 20.1 | `01_pr20_1_web_backend_foundation.md` | FastAPI app factory, security, CLI entry, static hooks |
| 20.2 | `02_pr20_2_rest_api_services.md` | REST endpoints over service layer |
| 20.3 | `03_pr20_3_run_streaming_and_approvals.md` | WebSocket run stream, prompt bridge, cancel |
| 20.4 | `04_pr20_4_web_frontend_shell.md` | Vite/React app shell, API client, stores |
| 20.5 | `05_pr20_5_chat_transcript_tools_diff.md` | Chat, streaming markdown, tools, diff, permissions |
| 20.6 | `06_pr20_6_settings_models_skills_health.md` | Settings/providers/models/skills/tools/health pages |
| 20.7 | `07_pr20_7_tests_dev_server_acceptance.md` | Tests, scripts, acceptance, packaging polish |

## Acceptance Summary

- `aether-web` starts a local Python server.
- The browser app can create/resume sessions and show transcripts.
- A browser user can run the agent and see streaming text, reasoning, tool
  events, token usage, and final state.
- Tool permission and plan approval prompts work in the browser.
- Model/provider, tools, skills, health, and diagnostics surfaces are visible.
- Existing TUI/gateway tests continue to pass.
- Web backend tests prove service-layer usage and error mapping.
- Web frontend tests prove API client, run socket state, transcript rendering,
  permission dialogs, and layout basics.
