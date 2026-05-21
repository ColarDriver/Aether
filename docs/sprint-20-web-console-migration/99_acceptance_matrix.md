# Sprint 20 - Acceptance Matrix

| # | Scenario | 20.1 | 20.2 | 20.3 | 20.4 | 20.5 | 20.6 | 20.7 |
|---|---|---|---|---|---|---|---|---|
| E1 | Web backend imports | app factory | route services mounted | WS module imports | API client target | chat consumes | settings consumes | pyright |
| E2 | Local server starts | `aether-web` | REST routers included | WS route included | Vite proxy docs | chat URL works | views load | dev script |
| E3 | Local dashboard security | token/host/CORS | protected REST | protected WS | token client | prompt safety | no secret UI | regression |
| E4 | Status and health | `/api/status`, `/api/health` | health serialization | run health unaffected | shell badge | chat status | diagnostics view | manual |
| E5 | Sessions | foundation DI | list/create/resume/delete/messages | run start validates | session store | transcript merge | session nav | tests |
| E6 | Providers/models | service DI | provider/model REST | run uses session model | provider store | composer model label | model view | tests |
| E7 | Tools and skills | service DI | REST catalogs | tool events | store types | tool blocks | catalog views | tests |
| E8 | Agent run streaming | app loop ready | run snapshot route | `run.start` stream | run socket | chat stream | status panels | manual |
| E9 | Cancellation | app loop ready | REST cancel | WS cancel | stop button | cancelled state | status updates | tests |
| E10 | Permissions | security | error mapping | prompt broker | socket responder | permission dialog | no secret previews | tests |
| E11 | Plan approval/questions | security | error mapping | approval prompter | socket responder | approval dialog | mode display | tests |
| E12 | Markdown/code/diff | static assets | transcript payload | tool metadata | app shell | renderers | detail panels | visual check |
| E13 | Frontend build | static hook | API shapes | WS types | Vite build | chat components | settings components | CI |
| E14 | Existing TUI regression | no changes | service reuse only | no gateway breakage | separate package | separate UI | separate UI | full tests |

## Required Files

| File | Purpose |
|---|---|
| `00_overview.md` | architecture, reference analysis, goals, boundaries |
| `01_pr20_1_web_backend_foundation.md` | FastAPI foundation and local security |
| `02_pr20_2_rest_api_services.md` | service-backed REST endpoints |
| `03_pr20_3_run_streaming_and_approvals.md` | WebSocket run stream and prompt bridge |
| `04_pr20_4_web_frontend_shell.md` | Vite/React shell, clients, stores |
| `05_pr20_5_chat_transcript_tools_diff.md` | chat, markdown, tools, diffs, prompts |
| `06_pr20_6_settings_models_skills_health.md` | provider/settings/catalog/diagnostic views |
| `07_pr20_7_tests_dev_server_acceptance.md` | automated/manual acceptance and packaging |
| `99_acceptance_matrix.md` | scenario-to-PR verification map |
| `README.md` | sprint index |

## Backend Implementation Evidence

Required before Sprint 20 completion:

- `aether/web/app.py`
- `aether/web/entry.py`
- `aether/web/security.py`
- `aether/web/errors.py`
- `aether/web/serializers.py`
- `aether/web/static.py`
- `aether/web/routes/health.py`
- `aether/web/routes/sessions.py`
- `aether/web/routes/config.py`
- `aether/web/routes/providers.py`
- `aether/web/routes/tools.py`
- `aether/web/routes/skills.py`
- `aether/web/routes/diagnostics.py`
- `aether/web/routes/runs.py`
- `aether/web/ws/runs.py`
- `aether/web/ws/prompts.py`
- `aether/web/ws/events.py`
- `aether/tests/web/*`

## Frontend Implementation Evidence

Required before Sprint 20 completion:

- `web/package.json`
- `web/vite.config.ts`
- `web/tsconfig.json`
- `web/index.html`
- `web/src/main.tsx`
- `web/src/App.tsx`
- `web/src/styles.css`
- `web/src/api/client.ts`
- `web/src/api/runSocket.ts`
- `web/src/api/types.ts`
- `web/src/stores/*`
- `web/src/components/layout/*`
- `web/src/components/chat/*`
- `web/src/components/settings/*`
- frontend tests for API, socket, stores, chat, diff, prompts, and settings

## Automated Check Matrix

| Check | Required For |
|---|---|
| `python -m pytest aether/tests/web` | web backend |
| `python -m pytest aether/tests/services` | service layer unaffected |
| `python -m pytest aether/tests/gateway` | TUI/gateway unaffected |
| `python -m pytest aether/tests/cli` | CLI unaffected |
| `python -m pytest aether/tests/agents` | runtime unaffected |
| `python -m pytest aether/tests/tools` | tools unaffected |
| `uv run pyright aether/web aether/services aether/gateway/handlers` | Python typing |
| `cd web && npm run build` | frontend production build |
| `cd web && npm run test` | frontend unit tests |
| `cd web && npm run typecheck` | frontend typing |

## Final Manual Acceptance

- Start backend and frontend in dev mode.
- Load the browser console.
- Create a session.
- Send a message.
- Observe streaming assistant output.
- Observe token/status updates.
- Approve or deny a tool permission.
- Approve or reject a plan approval.
- Cancel an active run.
- Resume a previous session.
- Inspect provider/model settings.
- Inspect tools and skills.
- Inspect diagnostics and health.
- Build the SPA and serve it from `aether-web`.
- Start the existing TUI and verify it still works.


## Current Implementation Evidence

Implemented on branch `web-console-migration`:

- PR20.1 backend foundation: FastAPI app factory, local auth/Host/CORS middleware, `aether-web` entrypoint, static SPA bootstrap hooks.
- PR20.2 REST services: sessions, config/prefs, providers/models, tools, skills, diagnostics, health, logs, and run status/cancel routes over `aether/services/*`.
- PR20.3 run WebSocket: structured `/api/runs/ws` protocol, event mapping, cancellation, web prompt broker, approval and permission responders.
- PR20.4 frontend shell: standalone Vite/React TypeScript app, API client, run socket, stores, sidebar, status bar, and console shell.
- PR20.5 chat surface: transcript loading, persisted tool/diff reconstruction, composer, optimistic user messages, assistant deltas, tool blocks, token usage, permission modal, approval modal, markdown plan/table/code rendering, and diff viewer.
- PR20.6 console views: provider/model selection, tools, skills, diagnostics, logs, and read-only settings views.

Latest verification performed during implementation:

- `python -m pytest aether/tests/web`
- `python -m pytest aether/tests/services`
- `python -m pytest aether/tests/gateway`
- `cd web && npm test` (12 files / 18 tests, including layout, provider/diagnostics/logs views, markdown table/code, approval dialog, and persisted tool reconstruction coverage)
- `cd web && npm run build`

Remaining hardening before declaring the whole web migration complete:

- Add browser E2E/screenshot coverage for the full chat and settings workflows.
- Consider replacing the local lightweight markdown renderer with a full GFM/highlight pipeline if web output needs task lists, nested tables, or language-grade syntax highlighting.
- Decide whether Hermes-only dashboard domains such as plugins, analytics, logs, profiles, cron, and PTY should stay out of Aether or receive their own future service-backed sprints.
