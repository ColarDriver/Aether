# PR20.3 - Run Streaming and Approvals

## Goal

Implement the browser run channel: streamed agent turns, run cancellation,
token/status updates, tool events, and web-native approval/permission prompts.

## Current Problem

`AgentRunService` can emit transport-neutral events, but the only production
interactive prompt bridge is the gateway reverse-RPC path. A browser client
needs an independent WebSocket transport that can:

- start a run,
- stream events as they happen,
- ask the browser user for tool/plan/question decisions,
- resume or reject prompts correctly,
- cancel active runs,
- survive ordinary WebSocket disconnects without corrupting sessions.

Hermes offers a JSON-RPC WebSocket bridge and a PTY/event fan-out path. Aether
should not embed the TUI in a browser PTY for this sprint. It should stream
structured `AgentRunService` events directly.

## WebSocket Endpoint

Add:

- `WS /api/runs/ws`

Authentication:

- Query `?token=...` or subprotocol/header path where available.
- Must use the same token validator as REST.
- Tests may create the app with auth disabled.

Server ready frame:

```json
{
  "type": "ready",
  "payload": {
    "protocol": "aether.run.v1"
  }
}
```

## Client Messages

### `run.start`

```json
{
  "type": "run.start",
  "id": "client-request-id",
  "payload": {
    "session_id": "...",
    "user_message": "hello",
    "options": {
      "max_iterations": 10,
      "temperature": null,
      "max_tokens": null,
      "disable_builtin_tools": false,
      "system_override": null
    }
  }
}
```

Behavior:

- Validate `session_id` and `user_message`.
- Start `AgentRunService.start(...)` in a background thread/task so the socket
  receive loop can continue processing prompt responses and cancel messages.
- Attach a web event sink to emit every service event.
- Attach web approval/tool permission prompters to the run request.
- Reply with `run.accepted` or `error`.

### `run.cancel`

```json
{
  "type": "run.cancel",
  "payload": {
    "session_id": "...",
    "run_id": "..."
  }
}
```

Behavior:

- Calls `AgentRunService.cancel(...)`.
- Emits cancel acknowledgement.

### `permission.respond`

```json
{
  "type": "permission.respond",
  "payload": {
    "prompt_id": "...",
    "decision": {
      "type": "allow",
      "updated_arguments": null,
      "feedback": null,
      "rule": null
    }
  }
}
```

### `approval.respond`

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

### `ping`

Server replies with `pong`.

## Server Events

Map `aether/services/runs/events.py` to browser event names:

- `RunStarted` -> `run.started`
- `AssistantDelta` -> `assistant.delta`
- `ReasoningDelta` -> `reasoning.delta`
- `SilentProgress` -> `silent.progress`
- `RunStatusChanged` -> `run.status`
- `LoopStateChanged` -> `loop.state`
- `IterationStarted` -> `iteration.started`
- `IterationFinished` -> `iteration.finished`
- `ToolStarted` -> `tool.started`
- `ToolFinished` -> `tool.finished`
- `TokenUsageUpdated` -> `token.usage`
- `RunFinished` -> `run.finished`
- `RunFailed` -> `run.failed`
- `RunCancelled` -> `run.cancelled`

Prompt events are produced by web prompters:

- tool permission -> `permission.requested`
- plan approval/questions -> `approval.requested`

Each event should include:

- `type`
- `payload`
- monotonic `sequence` per WebSocket where practical
- `run_id` and `session_id` when available

## Prompt Bridge

Create `aether/web/ws/prompts.py`:

- `WebPromptBroker`
  - creates prompt IDs
  - stores pending futures/events
  - resolves prompt responses from WebSocket messages
  - times out with denial or rejection
  - rejects all pending prompts on socket disconnect
- `WebApprovalPrompter`
  - implements `confirm_plan(...)`
  - implements `ask_questions(...)`
- `WebToolPermissionPrompter`
  - implements `request_tool_permission(...)`

The bridge should mirror the gateway prompter semantics but should not import
`aether.gateway.reverse_rpc`.

## Concurrency Rules

- The WebSocket receive loop must remain responsive while a run is active.
- Background run execution must marshal outbound events onto the socket's event
  loop safely.
- Multiple concurrent runs for the same session should be rejected by
  `AgentRunService` and surfaced as a 409-style WebSocket error frame.
- Socket disconnect should not leave prompt futures waiting forever.
- Cancellation should call the shared run registry through `AgentRunService`.

## Tests

Add `aether/tests/web/test_web_run_ws.py`:

- WebSocket accepts and emits `ready`.
- `run.start` with fake `AgentRunService` emits accepted and run events.
- Assistant deltas are forwarded in order.
- Tool events include arguments, result content, error flag, metadata.
- Token usage events update.
- `run.cancel` calls cancel service.
- Permission request waits for `permission.respond`.
- Plan approval request waits for `approval.respond`.
- Disconnect rejects pending prompts.
- Invalid message returns structured `error`.

## Non-Goals

- Do not implement terminal PTY-in-browser.
- Do not reuse gateway reverse-RPC internals.
- Do not implement multi-client run fan-out in the first version.
- Do not persist live event streams to a separate database.

## Acceptance

- Browser WebSocket can run a complete agent turn with streamed events.
- Browser permission and approval prompts unblock the engine.
- Cancelling from the browser stops the active run.
- Existing TUI/gateway prompt behavior is unchanged.
