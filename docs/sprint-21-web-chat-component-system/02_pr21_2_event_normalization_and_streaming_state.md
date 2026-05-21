# PR21.2 - Event Normalization and Streaming State

## Goal

Convert persisted transcript messages and live run WebSocket frames into the
same `ChatBlock[]` model.

## Problem

The current `chatStore` handles each WebSocket frame directly against UI arrays.
This creates drift between history loading and live streaming:

- persisted assistant tool calls become `ToolBlock[]`,
- live `tool.started` events become another `ToolBlock[]`,
- assistant deltas mutate `ChatMessage[]`,
- permission/approval prompts live outside the timeline,
- reasoning/status frames mostly disappear.

The browser should have one canonical normalization layer that can be tested
without React.

## Required Files

Add:

```text
web/src/chat-rendering/normalizeTranscript.ts
web/src/chat-rendering/normalizeRunFrame.ts
web/src/chat-rendering/blockReducer.ts
web/src/chat-rendering/runState.ts
web/src/chat-rendering/normalizeTranscript.test.ts
web/src/chat-rendering/blockReducer.test.ts
```

Modify:

```text
web/src/stores/chatStore.ts
web/src/api/types.ts
```

## Transcript Normalization

Implement:

```ts
normalizeTranscript(sessionId: string, messages: TranscriptMessage[]): ChatBlock[]
```

Rules:

- `role=user` -> `user_message`.
- `role=assistant` with `text` -> `assistant_message`.
- `role=assistant` with `tool_calls` -> one `tool_call` block per call.
- `role=tool` with `tool_call_id` -> `tool_result`.
- `role=system` -> `system_notice`.
- `is_error` maps to error state on assistant/tool result blocks.
- `metadata.diff` or metadata edit preview forms create a `diff` child block or
  attached diff payload that render model can expose.
- Preserve transcript order. If an assistant message contains both text and tool
  calls, render assistant text first, then tool calls.
- If a tool result arrives without a matching call, still render a standalone
  `tool_result` block.

## Live Frame Reducer

Implement:

```ts
reduceRunFrame(state: ChatRenderState, frame: RunSocketFrame): ChatRenderState
```

Where `ChatRenderState` contains:

- `blocksBySession: Record<string, ChatBlock[]>`
- `activeRunId: string | null`
- `pendingPermission: PermissionPrompt | null`
- `pendingApproval: ApprovalPrompt | null`
- `tokenUsageByRun: Record<string, TokenUsage>`
- `statusByRun: Record<string, RunStatusSnapshot>`

Frame behavior:

- `run.accepted`
  - set active run ID.
- `assistant.delta`
  - append to the active streaming assistant block for the run,
  - create one if none exists,
  - keep `isStreaming: true`.
- `reasoning.delta`
  - append to the active thinking block for the run,
  - mark `isActive: true` while run is active.
- `silent.progress`
  - update or create `streaming_status` without adding noisy transcript blocks.
- `run.status`
  - update `streaming_status.state/detail`.
- `loop.state`
  - update status detail if it is user-meaningful; keep raw value in metadata if
    needed for debugging.
- `iteration.started` / `iteration.finished`
  - update status, do not spam transcript unless later UX needs it.
- `tool.started`
  - add or update a `tool_call`.
- `tool.finished`
  - mark matching `tool_call` finished/failed,
  - add or update matching `tool_result`,
  - extract diff metadata.
- `token.usage`
  - update `tokenUsageByRun`,
  - update visible `streaming_status.tokens`.
- `permission.requested`
  - set pending prompt,
  - add a `permission_request` block in pending state,
  - include preview/diff metadata.
- `approval.requested`
  - set pending approval,
  - add an `approval_request` block in pending state.
- `prompt.resolved`
  - clear pending prompt state,
  - update the matching permission/approval block state when payload identifies
    the prompt ID and result.
- `run.finished`
  - mark assistant/thinking/status blocks inactive,
  - clear active run if matching.
- `run.failed`
  - mark streaming assistant as error if present,
  - append `error` block if there is no visible error,
  - clear active run.
- `run.cancelled`
  - stop streaming,
  - append or update `system_notice` with cancellation summary if partial text
    is not already visible.

## Store Migration

Modify `chatStore` so the public state moves toward:

```ts
blocksBySession: Record<string, ChatBlock[]>
```

Compatibility is allowed during the migration:

- keep `messagesBySession` and `toolsBySession` temporarily if old components
  still read them,
- derive them from blocks rather than maintaining divergent state,
- remove them in PR21.6 when `ChatTimeline` fully replaces `MessageList`.

## Prompt Payload Fix

Check `runSocket.respondApproval(...)`.

The WebSocket protocol expects:

```json
{
  "type": "approval.respond",
  "payload": {
    "prompt_id": "...",
    "result": {
      "confirmed": true,
      "answers": {}
    }
  }
}
```

If the client currently flattens `confirmed` into `payload`, fix it here and
add a test. This is required before approval/question UI can be trusted.

## Tests

Add tests for:

- persisted transcript with assistant text + tool call + tool result becomes
  ordered blocks,
- live assistant deltas merge into one streaming block,
- reasoning deltas merge into thinking block,
- tool started/finished updates call status and creates result,
- permission request creates both pending modal state and timeline block,
- approval request creates both pending modal state and timeline block,
- run finished clears streaming flags,
- run failed creates visible error,
- approval response payload nests `result`.

## Acceptance

- The same `ChatBlock` renderer can consume history and live events.
- No WebSocket frame listed in Sprint 21 overview is silently dropped unless the
  PR explicitly documents why it is UI-internal.
- Prompt state and transcript state no longer diverge.
