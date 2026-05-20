# PR20.5 - Chat Transcript, Tools, Diff, and Permissions

## Goal

Implement the core browser chat experience: transcript rendering, streaming
assistant output, markdown, tool blocks, diff/code previews, permission dialogs,
approval dialogs, and composer behavior.

## Current Problem

The web shell alone is not useful unless it can show real agent turns. Aether's
TUI already handles streaming, tool calls, permissions, plan approval, and diffs
in terminal-specific components. The browser app needs equivalent structured
components rather than terminal output emulation.

cc-haha provides the best component reference for chat layout and diff/tool
rendering. Hermes provides useful smaller dashboard components, but its Chat
tab leans on PTY output and sidecar events. Aether should render structured
events directly.

## Required Components

```text
web/src/components/chat/
  ChatView.tsx
  MessageList.tsx
  UserMessage.tsx
  AssistantMessage.tsx
  MarkdownRenderer.tsx
  Composer.tsx
  StreamingIndicator.tsx
  ToolCallBlock.tsx
  ToolResultBlock.tsx
  ToolCallGroup.tsx
  DiffViewer.tsx
  CodeViewer.tsx
  PermissionDialog.tsx
  ApprovalDialog.tsx
  ThinkingBlock.tsx
  TokenUsagePill.tsx
```

## Transcript Rendering

Support:

- user messages
- assistant messages
- tool messages
- assistant tool calls
- error tool results
- metadata badges where present
- empty/partial assistant text during streaming

Transcript state should merge persisted messages from REST with live run events
from the WebSocket.

## Streaming Behavior

- User message appears immediately after send.
- Assistant delta appends to the current streaming assistant block.
- Reasoning delta appears in a collapsible thinking block.
- Silent progress updates a subtle activity indicator without stealing focus.
- Token usage updates in place.
- Run finished commits final text and clears active run state.
- Run failed/cancelled shows a visible terminal state without losing partial
  text.

## Tool Rendering

Render tool blocks from events:

- `tool.started`
  - tool name
  - compact argument preview
  - status `running`
- `tool.finished`
  - content preview
  - error state
  - metadata-driven previews

For file edits and write previews:

- Use metadata/diff fields if available.
- Render unified diff with red/green line backgrounds.
- Include `+`/`-` markers and line numbers.
- Keep diff container aligned with surrounding chat content.
- Avoid overly saturated colors that reduce code readability.

## Markdown and Code

The markdown renderer should support:

- paragraphs
- headings
- lists
- blockquotes
- inline code
- fenced code blocks
- tables
- links

Streaming markdown should render incrementally. Do not buffer an entire table or
large block until the full answer is complete if a reasonable incremental parse
is possible. Prefer stable partial rendering over layout jumps.

## Permissions and Approvals

Permission dialog:

- Triggered by `permission.requested`.
- Shows tool name, arguments, risk/category, reason, and preview/diff.
- Supports allow, deny, allow for session if backend request allows it.
- Sends `permission.respond`.

Approval dialog:

- Triggered by `approval.requested`.
- Supports plan approval markdown.
- Supports questions with options/free text.
- Sends `approval.respond`.
- Rejected plan keeps the session in plan mode; frontend displays a concise
  revision hint from server events when available.

Dialogs should not erase the tool/diff preview from transcript history after the
user decides. The permission panel can close, but the relevant event/result
should remain visible in the chat timeline.

## Composer

Minimum behavior:

- multiline input
- Enter sends, Shift+Enter inserts newline
- disabled state while no session exists
- stop button when a run is active
- preserve input on failed send
- optimistic user message append

## Tests

Add frontend tests:

- `MessageList` renders persisted user/assistant/tool messages.
- Assistant deltas append to streaming message.
- Reasoning delta renders inside thinking block.
- Tool started/finished produces stable tool block.
- DiffViewer renders added/removed lines with markers and line backgrounds.
- MarkdownRenderer renders tables and fenced code.
- PermissionDialog sends expected decision payload.
- ApprovalDialog sends approve/reject/question payloads.
- Composer handles Enter and Shift+Enter.

## Non-Goals

- Do not implement collaborative multi-user chat.
- Do not implement terminal PTY rendering.
- Do not add canvas effects or decorative animations.
- Do not fully clone cc-haha's desktop task/workspace panels.

## Acceptance

- A browser user can run a session and understand what the agent is doing.
- Tool permissions and plan approvals are actionable in-browser.
- Diffs and markdown are readable and aligned.
- Streaming output remains visible and stable during long answers.
