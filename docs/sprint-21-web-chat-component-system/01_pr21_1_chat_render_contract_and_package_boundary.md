# PR21.1 - Chat Render Contract and Package Boundary

## Goal

Introduce Aether's browser chat rendering contract as first-class TypeScript
code. This PR creates the typed block model that all later web chat rendering
must consume.

## Problem

The current web store exposes `ChatMessage` and `ToolBlock` as separate arrays.
That makes the UI easy to bootstrap but hard to make correct:

- tools appear outside the chronological message flow,
- tool results cannot reliably attach to their corresponding calls,
- thinking/permission/approval/question states are side channels,
- streaming state is spread across `activeRunId`, message flags, and token
  usage maps,
- persisted transcript reconstruction and live WebSocket handling do not share a
  single render contract.

## Required Files

Add:

```text
web/src/chat-rendering/blocks.ts
web/src/chat-rendering/blockGuards.ts
web/src/chat-rendering/content.ts
web/src/chat-rendering/index.ts
web/src/chat-rendering/blocks.test.ts
```

Do not create a separate npm workspace package in this PR. Keep the code local
to `web/src/chat-rendering` until the contract proves stable. The folder is the
package boundary and should avoid importing React.

## Block Model

Define a discriminated union named `ChatBlock`.

Minimum shape:

```ts
export type ChatBlockBase = {
  id: string
  sessionId: string
  runId?: string | null
  timestamp: number
  source: 'transcript' | 'live' | 'optimistic'
}
```

Block variants:

- `UserMessageBlock`
  - `kind: 'user_message'`
  - `content: string`
  - `attachments?: ChatAttachment[]`
  - `pending?: boolean`
- `AssistantMessageBlock`
  - `kind: 'assistant_message'`
  - `content: string`
  - `isStreaming?: boolean`
  - `isError?: boolean`
  - `model?: string | null`
- `ThinkingBlock`
  - `kind: 'thinking'`
  - `content: string`
  - `isActive?: boolean`
  - `sequence?: number`
- `ToolCallBlock`
  - `kind: 'tool_call'`
  - `toolCallId: string`
  - `toolName: string`
  - `arguments: Record<string, unknown>`
  - `status: 'pending' | 'running' | 'finished' | 'failed'`
  - `iteration?: number`
  - `parentToolCallId?: string | null`
- `ToolResultBlock`
  - `kind: 'tool_result'`
  - `toolCallId: string`
  - `toolName?: string | null`
  - `content: string`
  - `isError: boolean`
  - `metadata: Record<string, unknown>`
- `DiffBlock`
  - `kind: 'diff'`
  - `path?: string | null`
  - `diff?: string | null`
  - `oldText?: string | null`
  - `newText?: string | null`
  - `language?: string | null`
  - `origin: 'permission_preview' | 'tool_result' | 'transcript'`
- `PermissionRequestBlock`
  - `kind: 'permission_request'`
  - `promptId: string`
  - `toolCallId?: string | null`
  - `toolName: string`
  - `arguments: Record<string, unknown>`
  - `category?: string | null`
  - `risk?: string | null`
  - `reason?: string | null`
  - `preview?: PermissionPreview | null`
  - `state: 'pending' | 'allowed' | 'denied' | 'expired' | 'aborted'`
- `ApprovalRequestBlock`
  - `kind: 'approval_request'`
  - `promptId: string`
  - `approvalKind: 'plan' | 'questions' | string`
  - `planText?: string | null`
  - `planPath?: string | null`
  - `questions: AskUserQuestion[]`
  - `state: 'pending' | 'approved' | 'rejected' | 'answered' | 'expired'`
- `AskUserQuestionBlock`
  - `kind: 'ask_user_question'`
  - `promptId?: string | null`
  - `toolCallId?: string | null`
  - `questions: AskUserQuestion[]`
  - `answers?: Record<string, string>`
  - `state: 'pending' | 'answered' | 'cancelled'`
- `StreamingStatusBlock`
  - `kind: 'streaming_status'`
  - `state: 'thinking' | 'responding' | 'tool_use' | 'idle' | string`
  - `detail?: string | null`
  - `elapsedMs?: number`
  - `tokens?: TokenUsage`
- `SystemNoticeBlock`
  - `kind: 'system_notice'`
  - `content: string`
- `ErrorBlock`
  - `kind: 'error'`
  - `message: string`
  - `code?: string | null`

## Helper Types

Define reusable helper types:

- `ChatAttachment`
- `TokenUsage`
- `PermissionPreview`
- `AskUserQuestion`
- `AskUserQuestionOption`
- `PromptResolution`
- `ToolStatus`

These types should be explicit and narrow. Avoid `any`; use `unknown` for raw
payloads and normalize at the boundary.

## Content Helpers

Add `content.ts` helpers:

- `stringFromUnknown(value: unknown): string`
- `recordFromUnknown(value: unknown): Record<string, unknown>`
- `jsonPreview(value: unknown, options?: { maxChars?: number }): string`
- `firstNonEmptyLine(text: string): string`
- `extractDiffFromMetadata(metadata: Record<string, unknown>): DiffBlock | null`
- `parseAskUserQuestions(value: unknown): AskUserQuestion[]`

These helpers should be dependency-free and unit tested.

## Package Boundary Rules

- `web/src/chat-rendering/*` must not import React.
- `web/src/chat-rendering/*` must not import Zustand stores.
- `web/src/chat-rendering/*` may import TypeScript API types from
  `web/src/api/types.ts` only when needed.
- React components import the render contract, not the reverse.
- Store code can produce `ChatBlock[]`, but should not contain JSX or CSS class
  decisions.

## Migration Scope

This PR may keep the existing UI rendering intact while adding the new contract
and tests. Later PRs will switch `ChatView` and `MessageList` onto the new model.

## Tests

Add unit tests for:

- every block kind can be created with required fields,
- `recordFromUnknown` rejects arrays/null/primitives,
- `stringFromUnknown` handles string, object, array, null, and error-ish values,
- `extractDiffFromMetadata` recognizes `diff`, `path`, `old_string`,
  `new_string`, `oldText`, and `newText` forms,
- `parseAskUserQuestions` supports:
  - `{ questions: [...] }`,
  - `{ question: "...", options: [...] }`,
  - empty/invalid input.

## Acceptance

- `web/src/chat-rendering` is a clear, React-free contract boundary.
- All block kinds required by Sprint 21 exist as TypeScript types.
- The current web app still builds and tests without behavior changes.
