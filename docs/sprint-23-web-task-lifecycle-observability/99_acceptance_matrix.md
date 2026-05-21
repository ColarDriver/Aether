# Sprint 23 - Acceptance Matrix

| Scenario | Backend | Frontend | Tests |
|---|---|---|---|
| Per-session task list | `/api/sessions/{id}/tasks` | `taskStore.loadSessionTasks` | Python route test |
| Global task list | `/api/tasks` | `api.tasks` | Python route test |
| Task detail output tail | `TaskService.get_task` | `api.taskDetail` | Python route test |
| Active task visibility | status and progress fields | `SessionTaskBar` auto-expands | component test |
| Terminal history | terminal status classification | collapsible bar | component test |
| Error handling | service validation/not-found | API error propagation | Python route test |
| Task notification transcript | XML parser | `TaskNotificationBlock` | render/model tests |
| Task detail inspection | `/api/tasks/{id}` | `TaskDetailDialog` | component test |

## Current Evidence

Implemented on branch `web-console-migration`:

- `aether/services/tasks/*`
- `aether/web/routes/tasks.py`
- `aether/web/app.py` task service wiring
- `web/src/api/client.ts` task methods
- `web/src/api/types.ts` task contracts
- `web/src/stores/taskStore.ts`
- `web/src/components/chat/SessionTaskBar.tsx`
- `web/src/components/chat/TaskDetailDialog.tsx`
- `web/src/components/chat/ChatView.tsx` integration
- `web/src/chat-rendering/content.ts` task-notification parser
- `web/src/chat-rendering/normalizeTranscript.ts` task-notification block mapping
- `web/src/components/chat/blocks/TaskNotificationBlock.tsx`

Verification performed:

- `python -m pytest aether/tests/web/test_web_rest_services.py`
- `python -m pytest aether/tests/web`
- `cd web && npm test -- --run src/components/chat/SessionTaskBar.test.tsx src/components/chat/ChatView.test.tsx`
- `cd web && npm test`
- `cd web && npm run typecheck`
- `cd web && npm run build`
- `uv run pyright aether/services/tasks aether/web`
- `git diff --check`
