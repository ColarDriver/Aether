# PR21.4 - Tool, Result, Diff, and Code Blocks

## Goal

Implement typed tool rendering for Aether web:

- grouped tool calls,
- standalone tool calls,
- tool results,
- edit/write diff previews,
- shell command previews,
- code viewer primitives.

## Required Files

Add:

```text
web/src/chat-rendering/renderModel.ts
web/src/chat-rendering/renderModel.test.ts
web/src/components/chat/blocks/ToolCallBlock.tsx
web/src/components/chat/blocks/ToolCallGroup.tsx
web/src/components/chat/blocks/ToolResultBlock.tsx
web/src/components/chat/blocks/DiffBlock.tsx
web/src/components/chat/blocks/CodeBlock.tsx
```

Modify or replace:

```text
web/src/components/chat/ToolCallBlock.tsx
web/src/components/chat/DiffViewer.tsx
web/src/styles.css
```

## Render Model

Implement:

```ts
buildChatRenderModel(blocks: ChatBlock[]): ChatRenderItem[]
```

Render items:

- plain message block,
- `tool_group` item for adjacent root-level tool calls,
- standalone tool result when no matching call exists,
- diff child item when it should appear inline under a tool or permission.

Rules:

- Adjacent root-level tool calls should group together.
- A `tool_result` with a matching `tool_call_id` should render inside or directly
  under the matching call, not duplicated later.
- Child tool calls with `parentToolCallId` should render nested under their
  parent when present.
- `ask_user_question` should not be hidden inside a generic tool group; it gets
  its own interactive block.
- Failed tool results should keep the result visible and mark the call failed.
- Unknown tools should still render with generic input/result sections.

## ToolCallBlock

Render:

- tool icon from a small local mapping using `lucide-react` where available,
- tool name,
- compact summary from known argument keys,
- status badge,
- expandable details,
- input JSON/code section,
- inline result preview when present,
- child tool calls if present.

Known summaries:

- shell: command/description,
- read: file path,
- write: file path + created line count,
- edit/file_edit: file path + changed line count,
- grep/search: pattern,
- web search/fetch: query/url,
- subagent/task: description/prompt preview,
- ask_user_question: question count.

Do not hard-code only Claude-style tool names. Normalize common Aether names:

- `shell`, `bash`, `run_command`,
- `read_file`, `Read`,
- `write_file`, `Write`,
- `file_edit`, `Edit`,
- `web_search`, `WebSearch`,
- `web_fetch`, `WebFetch`,
- `task`, `Agent`,
- `ask_user_question`, `AskUserQuestion`.

## ToolResultBlock

Render:

- success/error state,
- first-line or line-count summary,
- expandable output,
- JSON pretty-print when the result is structured,
- ANSI-stripped text for summaries,
- monospace output for command/tool logs.

The result should remain visible after permission panels or modals close.

## DiffBlock and DiffViewer

Support two input forms:

- unified diff string,
- old/new text pair.

Render:

- file path header when available,
- additions/deletions count,
- line marker gutter,
- line number gutter where available,
- full-row red/green backgrounds,
- code text with readable syntax colors,
- horizontal scroll for long lines,
- no layout shift when expanded.

If dependency-free old/new diffing is too weak, implement a simple line-based
diff first and document limitations. Do not add a large diff dependency until
tests demonstrate the need.

## CodeBlock

Create a reusable code viewer used by markdown, tool result, shell output, and
diff components.

Support:

- language label,
- copy action when feasible,
- max-height with overflow,
- line wrapping option,
- lightweight token colors for JSON/JS/TS.

## Tests

Add tests for:

- adjacent tool calls group together,
- tool result attaches to matching tool call,
- unmatched result renders standalone,
- ask_user_question escapes generic grouping,
- write/edit arguments produce diff preview,
- metadata diff produces `DiffBlock`,
- error result keeps visible content,
- diff renders full-line add/remove classes and markers,
- code viewer renders JSON token classes.

## Acceptance

- Tool calls and results render in chronological context.
- Diffs from permissions, transcripts, and tool results use the same component.
- The previous separate `tool-stack` no longer controls primary chat rendering.
