# Sprint 23 - Web Task Lifecycle Observability

## Goal

Expose Aether subagent/task lifecycle state in the web console so users can see
background and foreground delegation progress without reading raw tool JSON or
task-store files.

This sprint builds on Sprint 21's chat renderer. It does not replace the chat
timeline; it adds a runtime observability surface tied to the current session.

## Current State

Aether already has a Python task runtime:

- `aether/runtime/tasks/TaskStore` persists task records, output tails, pending
  messages, progress counters, and final result metadata.
- `aether/subagents/manager.py` creates task records for async subagents and
  updates status/progress through lifecycle hooks.
- `task_output`, `task_stop`, and `send_message` use the same task-store
  contracts.

The web gap is that none of this is visible in the browser:

- no REST route exposes task records,
- no TypeScript API type models task status,
- no frontend store keeps session task state fresh,
- no chat-adjacent UI shows running/completed subagent work.

## Reference Analysis

cc-haha has two relevant UI patterns:

- `SessionTaskBar`: a compact per-session task summary with progress and an
  expandable task list.
- `InlineTaskSummary`: a transcript-local summary after a completed task group.

Aether should adopt the semantic shape but keep its own contracts:

- the backend source of truth is Aether `TaskStore`, not cc-haha's
  `useCLITaskStore`,
- the browser implementation uses Aether CSS variables and lucide icons,
- the first pass is read-only observability; stop/output actions can land in a
  follow-up PR after the base surface is stable.

## Scope

Included:

- Python `TaskService` that wraps `TaskStore` read APIs.
- FastAPI routes for global task list, per-session task list, and task detail.
- TypeScript task API types and client methods.
- Zustand task store for current-session polling.
- `SessionTaskBar` browser component with active/completed/error states.
- Python and TypeScript tests.

Excluded for this sprint:

- editing task records,
- stopping tasks from the browser,
- streaming task output panes,
- A2A mailbox UI,
- desktop notifications,
- terminal/Ink rendering.

## Acceptance

- `/api/sessions/{session_id}/tasks` returns only tasks for that session.
- `/api/tasks/{task_id}` returns task metadata plus output tail.
- Missing task IDs and invalid limits return normal service errors.
- The chat page shows a session task bar when the current session has tasks.
- Running tasks auto-expand the bar and keep polling while active.
- Terminal task history is collapsible and does not interfere with chat scroll.
