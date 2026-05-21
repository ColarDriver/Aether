# PR21.3 - Message Primitives: User, Assistant, Thinking, Markdown

## Goal

Build the base React DOM components for text-centered chat blocks:

- user messages,
- assistant messages,
- thinking blocks,
- system notices,
- errors,
- markdown and code rendering,
- streaming caret/status affordances.

## Required Files

Add:

```text
web/src/components/chat/ChatTimeline.tsx
web/src/components/chat/blocks/UserMessageBlock.tsx
web/src/components/chat/blocks/AssistantMessageBlock.tsx
web/src/components/chat/blocks/ThinkingBlock.tsx
web/src/components/chat/blocks/SystemNoticeBlock.tsx
web/src/components/chat/blocks/ErrorBlock.tsx
web/src/components/chat/blocks/StreamingStatusBlock.tsx
web/src/components/chat/blocks/index.ts
```

Modify:

```text
web/src/components/chat/MarkdownRenderer.tsx
web/src/components/chat/MessageList.tsx
web/src/styles.css
```

## Component Principles

- Components receive typed `ChatBlock` variants, not raw WebSocket payloads.
- Components do not mutate store state directly.
- Components are readable without visual noise: dense enough for engineering
  workflows, but not terminal-emulation.
- Text must wrap safely at narrow widths.
- Markdown/code/table rendering must be stable while streaming.

## ChatTimeline

`ChatTimeline` should replace `MessageList` as the semantic renderer.

Props:

```ts
type ChatTimelineProps = {
  blocks: ChatBlock[]
  activeRunId?: string | null
  onRespondQuestion?: ...
  onRespondPermission?: ...
  onRespondApproval?: ...
}
```

Responsibilities:

- call `buildRenderModel(blocks)`,
- render blocks in order,
- dispatch variants to components,
- render streaming status at the bottom of the active turn,
- keep empty state local and simple.

It should not fetch session data and should not know about WebSocket clients.

## UserMessageBlock

Render:

- message text with preserved whitespace where useful,
- optional attachments as compact chips,
- pending optimistic state,
- clear alignment distinct from assistant output.

Do not use giant bubbles. Aether is an engineering console; the component should
stay compact and readable.

## AssistantMessageBlock

Render:

- markdown via `MarkdownRenderer`,
- streaming caret when `isStreaming`,
- error state when `isError`,
- optional model label only if useful and not repeated for every message.

Layout mode:

- "bubble" for short prose,
- "document" for long markdown, code blocks, tables, multi-paragraph output.

Implement a helper similar to cc-haha's `shouldUseDocumentLayout`, but keep it
in Aether naming and tests.

## ThinkingBlock

Render:

- collapsed by default,
- first meaningful line as preview,
- active shimmer/dots when `isActive`,
- expanded monospace content area,
- safe max-height with internal scroll.

The block should not force page scroll on every reasoning delta. It is allowed
to keep its own internal scroll at bottom only when expanded and active.

## MarkdownRenderer

Keep the renderer local and dependency-free for this PR.

Required support:

- paragraphs,
- headings,
- unordered lists,
- ordered lists,
- nested-looking indentation without crashing,
- blockquotes,
- inline code,
- bold,
- links,
- fenced code blocks,
- tables,
- partial streaming fences/tables.

Streaming rule:

- render incomplete blocks as best-effort text/code/table fragments,
- do not buffer an entire table until completion,
- avoid replacing previously rendered content with a blank block when the final
  newline arrives.

Syntax highlighting can remain lightweight:

- JSON/JS/TS primitive token colors,
- Python/shell can fall back to plain code until a later dependency decision.

## Styling

Add CSS classes under a clear namespace:

```css
.chat-timeline
.chat-block
.chat-block-user
.chat-block-assistant
.chat-block-thinking
.chat-block-status
.chat-message-shell
.chat-message-document
```

Avoid one-off class names that encode backend concepts.

## Tests

Add component tests:

- user block renders text and pending state,
- assistant block renders markdown and streaming caret,
- assistant document layout activates for code/table/long content,
- thinking block preview and expand/collapse work,
- markdown renders table incrementally,
- markdown renders incomplete fenced code without blanking,
- system/error blocks render with accessible roles where appropriate.

## Acceptance

- User, assistant, thinking, system, error, and streaming status blocks render
  from `ChatBlock` variants.
- Existing chat tests pass after migrating to the new components.
- Long markdown, code, and table output stays readable while streaming.
