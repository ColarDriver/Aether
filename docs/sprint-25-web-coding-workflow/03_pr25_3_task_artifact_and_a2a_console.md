# PR25.3 - Task Artifact And A2A Console

## Objective

Deepen the web UI for subagents and tasks from readable summaries into an
operable task console. Users should inspect nested task transcripts, send
follow-up messages, stop tasks, route artifacts safely, and see A2A-style
communication when backend event data exists.

## Current State

Already present or partially implemented:

- session task bar and inline task summary;
- task notification blocks;
- task detail dialog;
- parent/child task hierarchy;
- observer message streams from `messages.jsonl`;
- pending and delivered parent-to-child messages;
- descendant child stream filtering/collapse;
- duration/provider metadata;
- task follow-up messages;
- detail-level stop and session-bar quick stop;
- `result.json` artifact rendering with copy/open affordances.

Main gaps:

- nested child transcript controls are still shallow;
- artifact routing is limited;
- A2A events are not a first-class communication timeline;
- task action feedback is not detailed enough;
- artifact safety rules are not consistently represented in UI.

## Backend Scope

Primary files:

- `aether/services/tasks/contracts.py`
- `aether/services/tasks/service.py`
- `aether/web/routes/tasks.py`
- `aether/runtime/tasks/store.py`
- `aether/subagents/manager.py`
- `aether/tools/builtins/task.py` or the current task tool implementation.

Task detail contract:

- task identity, session ID, run ID, parent ID, child IDs;
- status and status timestamps;
- provider, model, runtime, and duration metadata;
- parent/child relationships and selected child details;
- message streams, including assistant/tool observer messages;
- pending and delivered parent-to-child messages;
- result summary and raw result path metadata;
- artifact manifest;
- stop/cancel capability and current action status;
- follow-up capability and unsupported reasons.

Artifact contracts:

- list artifacts for a task;
- read text/json artifacts safely;
- serve local binary artifacts through safe URLs where supported;
- return image preview metadata;
- expose copy-only local path metadata without leaking unsafe roots;
- include source task/run IDs;
- return not found for deleted-session artifacts instead of raw filesystem
  errors;
- refuse unsafe paths with structured errors.

A2A event shape:

- `event_id`, `task_id`, `source`, `target`, `kind`, `message`, `created_at`,
  `delivered_at`, `status`, `metadata`.

State rules:

- stop/cancel must be idempotent;
- sending follow-up to a completed task returns a clear unsupported result
  unless backend queueing is intentionally supported;
- action responses must distinguish accepted, unsupported, already completed,
  failed, and not found.

## Frontend Scope

Primary files:

- `web/src/api/types.ts`
- `web/src/api/client.ts`
- `web/src/stores/taskStore.ts`
- `web/src/components/chat/SessionTaskBar.tsx`
- `web/src/components/chat/TaskDetailDialog.tsx`
- `web/src/components/chat/blocks/InlineTaskSummary.tsx`
- `web/src/components/chat/blocks/TaskNotificationBlock.tsx`
- `web/src/components/chat/blocks/ToolResultPreview.tsx`
- `web/src/styles.css`

Required UI:

- nested task tree with selected child transcript;
- parent/child message timeline showing pending, delivered, acknowledged/result,
  and failed states when available;
- artifact panel with text preview, JSON preview, image preview, safe URL
  open/download, copy path, and copy JSON;
- task actions: stop, send follow-up, open task transcript, copy result, open
  artifact;
- action feedback: pending, accepted, unsupported, failed;
- result and artifact previews should not overflow the chat/timeline layout;
- deleted or inaccessible task artifacts should render stable "not available"
  states.

## Tests

Python:

- task detail includes hierarchy, streams, capabilities, and artifact manifest;
- artifact read refuses unsafe paths;
- stop is idempotent;
- follow-up to active task queues a message;
- follow-up to completed task returns unsupported or expected queued state;
- deleted-session artifacts are inaccessible.

TypeScript:

- task bar stop calls backend and updates status;
- detail dialog renders nested child streams;
- artifact panel renders text/json/image/binary variants;
- A2A event timeline renders pending and delivered messages;
- unsupported actions show stable readable output.

Browser/manual:

- fixture task hierarchy renders without overflow;
- stop active task from the session bar and detail dialog;
- send follow-up to an active task and see pending/delivered state;
- open text artifact, image artifact, JSON artifact, and copy-only local binary
  artifact;
- verify child task transcript navigation works at narrow widths.

## Acceptance

- Subagent/task work is inspectable without reading task-store files manually.
- Artifacts expose safe actions based on type.
- A2A-style communication is visible when backend event data exists.
- Task action results are visible and not swallowed.

## Explicit Exclusions

- Full multi-agent chat room.
- Cross-host artifact streaming.
- Editing task artifacts in place unless they are workspace files surfaced
  through workspace routes.
