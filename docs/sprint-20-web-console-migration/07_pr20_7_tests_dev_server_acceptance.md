# PR20.7 - Tests, Dev Server, Packaging, and Acceptance

## Goal

Harden the web console for local development and merge readiness: complete test
coverage, dev scripts, static serving, acceptance documentation, and regression
checks against the existing TUI/gateway.

## Required Backend Checks

Run and keep passing:

- `python -m pytest aether/tests/web`
- `python -m pytest aether/tests/services`
- `python -m pytest aether/tests/gateway`
- `python -m pytest aether/tests/cli`
- `python -m pytest aether/tests/agents`
- `python -m pytest aether/tests/tools`
- `uv run pyright aether/web aether/services aether/gateway/handlers`

Backend test coverage should include:

- app creation
- auth middleware
- Host header validation
- health/status routes
- sessions REST
- providers/models REST
- tools/skills/diagnostics/docs REST
- run status/cancel REST
- run WebSocket ready/start/cancel
- prompt request/response bridge
- disconnect cleanup
- service error mapping
- static SPA mount path

## Required Frontend Checks

Add scripts:

- `npm run dev`
- `npm run build`
- `npm run test`
- `npm run typecheck`

Frontend tests should cover:

- API client headers/errors/timeouts
- run WebSocket URL, queued send, reconnect, and ping behavior
- app shell layout
- session store
- chat store event reducer
- message rendering
- markdown/code/table rendering
- diff viewer
- permission dialog
- approval dialog
- provider/model/settings/docs/analytics views

## Static Serving

Wire Python static serving:

- Dev mode can point users to Vite directly.
- Built mode serves `web/dist` or an explicit `--web-dist` path.
- `index.html` receives:
  - `window.__AETHER_BASE_PATH__`
  - `window.__AETHER_SESSION_TOKEN__`
- Unknown non-API routes fall back to SPA index.
- Missing build directory returns a clear 404/instruction rather than a stack
  trace.

## Dev Workflow

Document two local modes:

1. Python backend + Vite frontend:
   - `uv run aether-web --port 9120`
   - `cd web && npm run dev`
   - Vite proxies `/api` to Python.
2. Single Python server with built SPA:
   - `cd web && npm run build`
   - `uv run aether-web --web-dist web/dist`

## Manual Acceptance Script

1. Start `uv run aether-web --port 9120`.
2. Start `cd web && npm run dev`.
3. Open the Vite URL.
4. Confirm status/health loads.
5. Create a new session using the current configured provider/model.
6. Send `你好`.
7. Confirm user message appears immediately.
8. Confirm assistant deltas stream into one assistant message.
9. Trigger or simulate a tool permission prompt.
10. Confirm the permission dialog shows preview/diff and allow/deny works.
11. Trigger or simulate plan approval.
12. Confirm markdown plan approval renders and approve/reject works.
13. Confirm token usage and run status update during the run.
14. Cancel an active run and verify final state is coherent.
15. Resume a previous session and verify transcript renders.
16. Open Models, Tools, Skills, Diagnostics, Logs, Docs, Analytics, Environment, and Settings views.
17. Confirm existing TUI still starts and can run a turn.

## Regression Rules

- Existing TUI terminal rendering must not change in this sprint.
- Gateway JSON-RPC wire schemas must not change.
- `aether/services/*` must stay transport-neutral.
- Web routes must not import `aether.gateway.handlers`.
- Browser code must not assume Node-only APIs.
- No raw credentials in frontend logs, API payloads, or screenshots.

## Acceptance

- Backend and frontend automated checks pass.
- Manual acceptance script is recorded in `99_acceptance_matrix.md`.
- Branch has staged commits per PR phase.
- Final web console can be run locally and is usable for core Aether workflows.
