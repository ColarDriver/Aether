# Sprint 21 - Web Chat Component System

## Goal

Build Aether's own browser-native chat rendering component system so the web
console can render real agent turns with the same semantic coverage as the TUI
and the cc-haha desktop reference.

This sprint is not a visual polish pass. It establishes the foundation that all
web chat features should use:

- a typed TypeScript render contract,
- a deterministic event/transcript normalization layer,
- React DOM components for every chat block kind,
- stable streaming state,
- prompt surfaces for permission, approval, and user questions,
- regression tests for transcript and live WebSocket flows.

## Current State

Aether web currently has a minimal chat surface under `web/src/components/chat`.
It can show user and assistant text, stream assistant deltas, show a basic tool
stack, render simple markdown, render diffs, and open permission/approval
modals. This was enough for Sprint 20 to prove the browser run channel works.

The current structure is not strong enough for production chat rendering:

- `ChatView` renders `messages` and `tools` as separate stacks, so tool calls
  are not reliably positioned in the same timeline as assistant text.
- `ChatMessage` only models `role/text/isStreaming/isError`; it cannot express
  thinking, tool result, diff, permission request, plan approval, or questions
  as first-class timeline items.
- `reasoning.delta`, `silent.progress`, `run.status`, and `loop.state` are not
  converted into visible streaming/thinking state.
- Permission and approval prompts are only modal state; the transcript does not
  retain a stable historical record of the request after the modal resolves.
- `ask_user_question` is not a dedicated web component with answer state.
- Diff rendering exists, but the component contract is too narrow to support
  generated edit previews, final tool result diffs, and standalone diff blocks.
- Markdown rendering is local and lightweight; it should remain dependency-free
  for now, but its partial streaming behavior and table/list/code handling need
  to be treated as core chat behavior.

## Reference Analysis

cc-haha has the useful shape for this work:

- `UIMessage` is a discriminated union for user, assistant, thinking, tool use,
  tool result, permission request, error, system, and task summary blocks.
- `MessageList.buildRenderModel(...)` groups adjacent tool calls, associates
  tool results by `toolUseId`, and avoids rendering duplicate result blocks.
- `MessageBlock` dispatches to typed components instead of branching on loose
  metadata in the top-level chat view.
- `AssistantMessage`, `ThinkingBlock`, `ToolCallBlock`, `ToolResultBlock`,
  `DiffViewer`, `AskUserQuestion`, and `StreamingIndicator` are independently
  testable components.

Aether should use that architecture as a reference, but not copy it wholesale:

- Aether's backend source of truth is `aether/services/runs/events.py` and the
  REST transcript schema exposed by `aether/web/routes/sessions.py`.
- Aether should not depend on cc-haha's i18n, Tailwind, Material Symbols,
  desktop shell, team/task panels, or checkpoint rewind UI.
- Aether should preserve its current Vite/React app, Zustand store, CSS variable
  theme, and service-backed web API.

## Ink Decision

Do not build or reuse Ink for the web chat renderer.

Ink solves terminal rendering. The browser needs DOM semantics, accessibility,
CSS layout, scroll containers, focus management, and pointer/keyboard behavior.
Trying to make a web Ink clone would recreate the same terminal constraints that
the web app is meant to avoid.

If Aether needs a reusable TypeScript package, it should be an Aether-owned web
chat renderer package, not an Ink replacement. The package boundary should look
like this:

```text
web/src/chat-rendering/
  blocks.ts              # discriminated union and helpers
  normalize.ts           # transcript + websocket frame normalization
  renderModel.ts         # grouping and result association
  selectors.ts           # derived state helpers

web/src/components/chat/
  ChatTimeline.tsx       # consumes render model
  blocks/*.tsx           # DOM components for each block kind
```

If another frontend later needs the same renderer, extract
`web/src/chat-rendering` and `web/src/components/chat/blocks` into a package
such as `packages/aether-chat-renderer`. This sprint should first make the
contract correct inside the web app.

## Target Block Coverage

The renderer must support these first-class block kinds:

- `user_message`
- `assistant_message`
- `thinking`
- `tool_call`
- `tool_result`
- `tool_group`
- `diff`
- `permission_request`
- `approval_request`
- `ask_user_question`
- `streaming_status`
- `system_notice`
- `error`

## Event Coverage

The web store must normalize both persisted transcript messages and live frames:

- `assistant.delta`
- `reasoning.delta`
- `silent.progress`
- `run.status`
- `loop.state`
- `iteration.started`
- `iteration.finished`
- `tool.started`
- `tool.finished`
- `token.usage`
- `permission.requested`
- `approval.requested`
- `run.finished`
- `run.failed`
- `run.cancelled`
- `prompt.resolved`

## Non-Goals

- Do not implement a browser terminal emulator.
- Do not introduce Ink or an Ink-compatible abstraction for web rendering.
- Do not copy cc-haha's team mailbox, desktop notification, material icon, or
  checkpoint rewind systems.
- Do not add large markdown/diff/highlight dependencies in the first pass unless
  a later PR proves the local renderer cannot satisfy the acceptance tests.
- Do not route the web backend through gateway handlers.

## Acceptance

- A browser chat turn renders in chronological order across user text,
  assistant text, thinking, tool calls, tool results, diffs, permissions,
  approvals, questions, and streaming status.
- Persisted transcript and live WebSocket events produce the same render model.
- Permission and approval prompts are actionable and leave a durable timeline
  artifact after they resolve.
- `ask_user_question` is answerable in-browser and renders the submitted answer.
- Streaming output is incremental and does not wait for an entire markdown block
  or table before showing useful content.
- The renderer is covered by unit tests at the contract, model, component, and
  store integration layers.
