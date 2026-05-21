# PR22.3 - Slash Execution And Timeline Notices

## Scope

Do not let completed slash commands fall through as ordinary agent prompts.
Execute supported commands through web-native services and render durable
timeline feedback.

## Execution Contract

Add a frontend execution layer similar to hermes `slashExec.ts`, but backed by
Aether REST/WebSocket APIs instead of gateway JSON-RPC.

Recommended result shape:

```ts
type WebSlashResult =
  | { type: 'notice'; text: string }
  | { type: 'send'; message: string }
  | { type: 'handled' }
  | { type: 'error'; message: string }
```

## Initial Commands

- `/help`: render command catalog as a system notice.
- `/session`: render current session metadata.
- `/sessions`: switch/open sessions view or render concise session list.
- `/tools`: render tool catalog summary.
- `/model`: open model view or render current model.
- `/plan`: call web plan APIs, update session mode, show current plan, and let
  `/plan <description>` continue as an agent run in plan mode.
- unknown commands: render a clear error notice.

## Store Work

- Add a web chat store method for local timeline notices.
- Notices must be visible in the same `ChatTimeline` as model output.
- Notices must be session-scoped.

## Tests

- `/help` does not call `agent.run`.
- unsupported slash commands create a visible error notice.
- command-generated `send` still starts an agent run.

## Current Implementation Evidence

Implemented on branch `web-console-migration`:

- Frontend execution:
  - `web/src/components/chat/slashExecute.ts`
  - `web/src/components/chat/ChatView.tsx`
  - `web/src/stores/sessionStore.ts`
- Web plan APIs:
  - `aether/web/routes/plan.py`
  - `aether/web/app.py`
  - `aether/services/sessions/service.py`
  - `web/src/api/client.ts`
  - `web/src/api/types.ts`

`/plan` behavior:

- `/plan`: enter plan mode when needed, then show current plan metadata/content.
- `/plan <description>`: enter plan mode when needed and return a `send` result
  so the description starts a normal agent run under plan-mode guardrails.
- `/plan open`: render the current plan content/path in the web timeline.
