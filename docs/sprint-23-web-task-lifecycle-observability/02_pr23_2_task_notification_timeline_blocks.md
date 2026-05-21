# PR23.2 - Task Notification Timeline Blocks

## Intent

Render async subagent completion notifications as first-class web timeline
blocks instead of raw XML user messages.

Aether's root engine can inject background subagent completion events as user
turns with `metadata.source = "task_notification"` and content shaped as:

```xml
<task-notification>
  <task_id>...</task_id>
  <subagent_type>...</subagent_type>
  <status>completed</status>
  <duration_seconds>1.0</duration_seconds>
  <summary>...</summary>
</task-notification>
```

That wire payload is useful for the model, but it is not a good browser UI.

## Changes

Add a `task_notification` chat block:

- `taskId`
- `subagentType`
- `status`
- `durationSeconds`
- `summary`
- `error`
- `outputFile`

Add parser helpers in `web/src/chat-rendering/content.ts`:

- parse Aether underscore tags,
- tolerate cc-haha/open-claude-code dash tag variants where practical,
- XML-decode text fields,
- require at least task ID and status before rendering.

Update `normalizeTranscript`:

- when a user message has `metadata.source = "task_notification"` or is a pure
  `<task-notification>...</task-notification>` payload, emit a
  `task_notification` block,
- otherwise preserve the message as a normal user message, so pasted XML in a
  real prompt is not accidentally hidden.

Add `TaskNotificationBlock` under `web/src/components/chat/blocks` and wire it
through `ChatTimeline`.

## Tests

- content helper parses XML and decodes entities,
- persisted transcript task notification normalizes to `task_notification`,
- `ChatTimeline` renders the block as readable subagent status,
- block union coverage includes the new block kind.

## Follow-Ups

- Link task notification blocks to `TaskService.get_task` detail.
- Render final output/result summary inside an expandable details drawer.
- Add live WebSocket task-notification frames if the backend starts emitting
  them before transcript persistence.
