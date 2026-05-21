# PR21.7 - Tests, Visual Acceptance, and Regression Hardening

## Goal

Make the web chat renderer safe to evolve. This PR adds coverage across the
normalization layer, render model, components, store integration, and manual
browser acceptance.

## Automated Tests

Run after this sprint:

```text
cd web && npm test
cd web && npm run build
python -m pytest aether/tests/web
python -m pytest aether/tests/services
uv run pyright aether/web aether/services
```

If backend protocol changes touch gateway compatibility, also run:

```text
python -m pytest aether/tests/gateway
```

## Required Test Files

Add or update:

```text
web/src/chat-rendering/blocks.test.ts
web/src/chat-rendering/normalizeTranscript.test.ts
web/src/chat-rendering/blockReducer.test.ts
web/src/chat-rendering/renderModel.test.ts
web/src/components/chat/ChatTimeline.test.tsx
web/src/components/chat/blocks/UserMessageBlock.test.tsx
web/src/components/chat/blocks/AssistantMessageBlock.test.tsx
web/src/components/chat/blocks/ThinkingBlock.test.tsx
web/src/components/chat/blocks/ToolCallBlock.test.tsx
web/src/components/chat/blocks/DiffBlock.test.tsx
web/src/components/chat/blocks/PermissionRequestBlock.test.tsx
web/src/components/chat/blocks/ApprovalRequestBlock.test.tsx
web/src/components/chat/blocks/AskUserQuestionBlock.test.tsx
web/src/components/chat/MarkdownRenderer.test.tsx
```

Existing tests can be renamed or expanded if they cover the same contract. The
current implementation intentionally removed `web/src/stores/chatStore.test.ts`
after `chatStore` stopped owning transcript reconstruction; block reducer and
normalization tests now cover that contract.

Current indirect coverage:

- `ChatTimeline.test.tsx` covers user/assistant/thinking/tool/result/prompt
  integration.
- `PromptContent.test.tsx` covers shared permission preview and question-answer
  rendering.
- `DiffViewer.test.tsx` covers unified diff parsing and line-number rendering.
- `SessionsView.test.tsx` covers persisted transcript rendering through
  `ChatTimeline`.

## Scenario Coverage

### Transcript

- empty transcript,
- user -> assistant,
- assistant text + tool call,
- tool result with matching call,
- standalone tool result,
- assistant error,
- system notice,
- transcript metadata diff.

### Live Streaming

- optimistic user message appears immediately,
- assistant delta streams incrementally,
- reasoning delta creates/updates thinking block,
- token usage updates streaming status,
- status changes from thinking to responding to tool_use to idle,
- run failed preserves partial assistant output,
- run cancelled shows non-destructive terminal state.

### Tools and Diffs

- shell command preview,
- read/write/edit/file_edit summaries,
- write diff from content,
- edit diff from old/new strings,
- unified diff from metadata,
- error result,
- nested/child tool call,
- grouped adjacent tools.

### Prompts

- permission requested -> pending block + modal,
- allow once,
- allow session,
- deny,
- plan approval markdown,
- plan reject,
- questions approval answers,
- ask_user_question options,
- ask_user_question free text,
- prompt resolution leaves durable transcript block.

### Markdown

- heading,
- list,
- ordered list,
- task list,
- table,
- partial table while streaming,
- fenced code,
- incomplete fenced code while streaming,
- inline code,
- bold,
- italic,
- strike,
- links,
- bare HTTP(S) links,
- blockquote,
- horizontal rule,
- streaming caret placement inside the final block.

## Current Markdown Evidence

`web/src/components/chat/MarkdownRenderer.tsx` keeps the renderer Aether-owned
while covering the common GFM surface used in chat transcripts:

- headings through H4,
- fenced code blocks with the shared `CodeBlock` and lightweight
  keyword/string/comment spans,
- inline code, strong, italic, strike,
- explicit safe links and bare HTTP(S) URLs,
- tables, including partial streaming tables,
- blockquotes,
- horizontal rules,
- unordered, ordered, and task-list items,
- a streaming caret rendered inside the final block.

Focused coverage lives in `web/src/components/chat/MarkdownRenderer.test.tsx`.

## Manual Acceptance Script

1. Start the web backend and frontend.
2. Open the browser console.
3. Create or select a session.
4. Send a short message and verify optimistic user message appears immediately.
5. Send a prompt that produces a long markdown answer with a table and fenced
   code; verify streaming renders incrementally.
6. Trigger a read/search tool; verify tool call and result appear in timeline.
7. Trigger a file edit permission; verify diff appears in permission surface and
   remains after approval/denial.
8. Trigger plan mode and `exit_plan_mode`; verify plan approval markdown renders
   and approve/reject payloads unblock the run correctly.
9. Trigger `ask_user_question`; verify option/free-text answer flow.
10. Scroll up during streaming; verify the page does not force-scroll until
    "jump to latest" is used.
11. Reload the page and verify persisted transcript reconstructs the same tool
    and diff blocks.

## Visual Acceptance

The chat renderer should be judged against these expectations:

- timeline blocks align to one readable content width,
- user and assistant messages are visually distinct but not oversized,
- thinking blocks are compact by default,
- tool blocks are scannable when collapsed and useful when expanded,
- diff add/remove rows use full-line color backgrounds,
- permission/approval prompts are prominent without erasing context,
- streaming status is visible but not distracting,
- narrow widths do not crush buttons or overflow long paths.

## Regression Guardrails

- Existing TUI behavior must not change.
- Existing gateway tests must not require web renderer imports.
- WebSocket protocol changes must be reflected in `web/src/api/types.ts` tests.
- No new frontend dependency should be added without a short note explaining why
  local code is insufficient.
- No prompt action should send secrets or raw API keys into visible transcript
  blocks.

## Acceptance

- Every Sprint 21 block kind has either a direct component test or a render model
  test.
- The full web build passes.
- Backend web tests still pass.
- Manual script confirms the browser chat chain works end to end.
