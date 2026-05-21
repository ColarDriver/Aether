# PR23.1 - Task Service and Session Task Bar

## Intent

Make subagent task lifecycle visible in Aether web through a clean service
boundary and a browser-native session task bar.

## Backend Changes

Add `aether/services/tasks/`:

- `contracts.py`
  - `TaskSummary`
  - `TaskListResult`
- `service.py`
  - `TaskService.list_tasks(...)`
  - `TaskService.get_task(...)`

`TaskService` rules:

- Read from `TaskStore`; do not duplicate persistence.
- Support `session_id`, `active_only`, `limit`, and `include_output_tail`.
- Keep `limit` bounded to avoid accidentally reading very large stores.
- Return public-safe service errors:
  - empty task ID -> validation error,
  - invalid limit -> validation error,
  - missing task -> not found.

Add `aether/web/routes/tasks.py`:

- `GET /api/tasks`
- `GET /api/sessions/{session_id}/tasks`
- `GET /api/tasks/{task_id}`

Wire the service into `aether/web/app.py` through `WebServices` and dependency
injection so tests can provide an isolated `TaskStore`.

## Frontend Changes

Add task API types to `web/src/api/types.ts`:

- `TaskStatus`
- `TaskSummary`
- `TaskListResult`

Add client methods in `web/src/api/client.ts`:

- `api.tasks(...)`
- `api.sessionTasks(sessionId, ...)`
- `api.taskDetail(taskId)`

Add `web/src/stores/taskStore.ts`:

- `tasksBySession`
- loading/error maps keyed by session ID
- `loadSessionTasks(sessionId)`

Add `web/src/components/chat/SessionTaskBar.tsx`:

- progress summary,
- active/completed/error/warn/pending status rendering,
- auto-expand when tasks are active,
- collapsible terminal history,
- compact metadata: subagent type, model, background, iterations, tool calls,
  token total, summary/error.

Integrate it into `ChatView` directly under the chat header. Poll while the
session has an active run or active tasks.

## Tests

Python:

- route list filtered by session,
- detail includes output tail,
- active-only filtering,
- missing task -> 404,
- invalid limit -> 400.

TypeScript:

- renders active/completed tasks with progress,
- terminal task history collapses/expands,
- terminal status classifier covers completed/failed/running.

## Follow-Ups

- Add a task detail drawer with output tail and result JSON.
- Add browser task stop action backed by `task_stop`.
- Add inline task summaries for transcript-local task notifications.
- Add WebSocket push for task lifecycle changes to reduce polling.
