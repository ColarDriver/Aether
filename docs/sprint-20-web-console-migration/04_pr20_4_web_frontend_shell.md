# PR20.4 - Web Frontend Shell

## Goal

Add a standalone TypeScript React/Vite browser app under `web/` and connect it
to the Python web backend through typed REST and WebSocket clients.

## Current Problem

Aether's only TypeScript UI is the Ink TUI under `tui/`. It is not structured as
a browser app and should not be reused directly. Hermes has a Vite app, and
cc-haha has strong browser/desktop component patterns. Aether needs a clean web
frontend that fits its own API and runtime.

## Required Files

```text
web/
  package.json
  tsconfig.json
  tsconfig.node.json
  vite.config.ts
  index.html
  src/
    main.tsx
    App.tsx
    styles.css
    api/
      client.ts
      runSocket.ts
      types.ts
    stores/
      appStore.ts
      sessionStore.ts
      chatStore.ts
      providerStore.ts
    components/
      layout/
        AppShell.tsx
        Sidebar.tsx
        StatusBar.tsx
        TopBar.tsx
      shared/
        Button.tsx
        Spinner.tsx
        EmptyState.tsx
        Modal.tsx
```

## Dependencies

Use a conservative browser stack:

- React
- React DOM
- TypeScript
- Vite
- Zustand or a small local store pattern
- lucide-react for icons
- markdown renderer dependency can wait until PR20.5 if needed

Keep this separate from `tui/package.json`. Do not convert the repo root into a
workspace unless needed later.

## App Shell

The first screen should be the console itself:

- Left sidebar:
  - Aether identity
  - session list
  - new session button
  - provider/model summary
  - health indicator
- Main area:
  - chat transcript placeholder
  - composer placeholder
  - run connection status
- Secondary area or top tabs:
  - Tools
  - Skills
  - Settings
  - Diagnostics

Do not build a marketing landing page.

## API Client

Create `web/src/api/client.ts` inspired by cc-haha:

- `getBaseUrl()`
- `setBaseUrl(...)`
- `setAuthToken(...)`
- `request<T>(...)`
- `ApiError`
- abort timeout support
- JSON error parsing
- token header `X-Aether-Session-Token`
- bearer fallback only if the backend supports it

Create `web/src/api/types.ts` from Aether web payloads:

- `SessionInfo`
- `TranscriptMessage`
- `ProviderSummary`
- `ProviderRuntimeStatus`
- `ModelSummary`
- `ToolSummary`
- `SkillSummary`
- `HealthStatus`
- `RunEvent`
- `PermissionRequest`
- `ApprovalRequest`

## Run Socket Client

Create `web/src/api/runSocket.ts` inspired by cc-haha:

- connect to `/api/runs/ws`
- reconnect with bounded exponential backoff
- ping interval
- pending message queue while reconnecting
- event listener registration
- `startRun(...)`
- `cancelRun(...)`
- `respondPermission(...)`
- `respondApproval(...)`

The run socket should be independent of React components so stores and tests can
exercise it directly.

## State Stores

Minimum stores:

- `appStore`
  - backend status
  - health
  - active view
  - connection state
- `sessionStore`
  - sessions
  - current session
  - transcript loading
- `chatStore`
  - streaming run state
  - messages
  - in-flight assistant text
  - tool blocks
  - pending prompts
- `providerStore`
  - providers
  - current provider/model
  - auxiliary slots

## Tests

Add frontend tests with Vitest once the package is scaffolded:

- API client builds URLs and headers correctly.
- API client maps JSON errors into `ApiError`.
- Run socket builds WS URL with token.
- Run socket queues messages before open.
- App shell renders sidebar, main area, and status bar.
- Stores can load sessions/providers using mocked API functions.

## Non-Goals

- Do not implement the full chat renderer in this PR.
- Do not implement settings pages beyond placeholders.
- Do not implement browser routing if a local state router is enough.
- Do not copy cc-haha's entire desktop shell.

## Acceptance

- `cd web && npm install && npm run build` succeeds.
- The app can load backend status and list sessions.
- The app has a stable responsive shell with no landing page.
- Backend can serve the built app once PR20.1 static hooks are wired.
