# PR23.3 - Task Detail Drawer

## Intent

Let users inspect a subagent task from the web chat surface without leaving the
session context or reading task-store files manually.

PR23.1 exposed task detail through `GET /api/tasks/{task_id}`. This PR consumes
that endpoint in the browser.

## UX

The `SessionTaskBar` task rows become clickable. Selecting a row opens a modal
detail surface containing:

- task prompt and ID,
- status, subagent type, session, model, background flag,
- iterations, tool calls, token total,
- started/finished timestamps,
- worktree and result paths when present,
- summary,
- error,
- output tail,
- metadata JSON.

The dialog is read-only. It does not stop or mutate tasks.

## Implementation

Frontend files:

- `web/src/components/chat/SessionTaskBar.tsx`
  - accepts `onOpenTask`,
  - renders task rows as accessible buttons.
- `web/src/components/chat/TaskDetailDialog.tsx`
  - calls `api.taskDetail(taskId)`,
  - uses the initial list row as an optimistic preview while loading,
  - renders output tail through the shared `CodeBlock`.
- `web/src/components/chat/ChatView.tsx`
  - owns selected task state,
  - closes task detail when the session changes.
- `web/src/components/chat/blocks/TaskNotificationBlock.tsx`
  - exposes the same detail action for completed async task notifications in
    the transcript.

Styles live in `web/src/styles.css` and reuse the existing modal shell.

## Tests

- `SessionTaskBar` calls `onOpenTask` when a task row is selected.
- `TaskDetailDialog` loads detail and displays output tail/metadata.
- `TaskDetailDialog` displays a readable error when the detail API fails.
- `ChatTimeline` routes task-notification detail clicks to the same selected
  task state.

## Follow-Ups

- Add a split-pane or drawer variant for large output tails.
- Add a task stop action only after the web service owns a reliable live
  subagent-manager handle.
- Add result JSON download/open affordance if result files become browser
  readable through workspace-safe routes.
