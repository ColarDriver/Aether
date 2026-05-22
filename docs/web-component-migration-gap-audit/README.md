# Web Component Migration Gap Audit

Status date: 2026-05-22
Branch observed: `web-console-migration`

## Bottom Line

Aether web has a real Aether-owned React/TypeScript chat-rendering foundation. It can normalize persisted transcript messages and live WebSocket frames into typed blocks, render the main chat timeline, show user/assistant/thinking/tool/diff/prompt/status blocks, and run through basic web-console workflows.

It is not complete cc-haha/Hermes-style developer-console parity. The remaining gap is mostly workflow depth: composer command ergonomics, richer per-tool lifecycle rendering, task/subagent drill-down, checkpoint/edit/retry message actions, repository/worktree launch controls, MCP management, and browser-level visual acceptance.

A practical completion estimate is:

- Foundation chat/runtime path: mostly complete.
- Coding-oriented tool UI: partial.
- Composer/workspace workflow parity: partial.
- Advanced developer-console controls: mostly missing or backend-dependent.
- Visual/browser acceptance: still weak.

## Current Implementation Evidence

Implemented in Aether web:

- `web/src/chat-rendering/*` defines a React-free block contract, transcript normalization, live frame reduction, render model grouping, and run-state snapshots.
- `ChatTimeline.tsx` renders a single typed timeline for user, assistant, thinking, tool groups, diffs, permission, approval, ask-user-question, streaming status, task notifications, system notices, and errors.
- `UserMessageBlock.tsx`, `AssistantMessageBlock.tsx`, `ThinkingBlock.tsx`, `StreamingStatusBlock.tsx`, and `SystemNoticeBlock.tsx` cover the base message states, with local copy/quote/edit/retry actions where those actions can be executed without new backend contracts.
- `MarkdownRenderer.tsx`, `MermaidRenderer.tsx`, and `MathRenderer.tsx` cover common Markdown, tables during streaming, fenced code, safe links, inline images/lightbox, Mermaid diagrams, and KaTeX math with lazy-loaded Mermaid/KaTeX bundles.
- `ToolCallBlock.tsx`, `ToolCallGroup.tsx`, `ToolResultBlock.tsx`, `TerminalChrome.tsx`, `ToolResultPreview.tsx`, `TodoListPreview.tsx`, `InlineTaskSummary.tsx`, `SessionTaskBar.tsx`, and `TaskDetailDialog.tsx` cover shell chrome, todo previews, file-read previews, file-change previews, notebook edit previews with cell source/diff rendering, grep/search previews, LSP previews, browser-result previews, nested provider web-search previews, web-fetch document cards, image artifacts, non-image artifact bundles with copy/open actions, spill notices, session task history, related task drill-down, and first-class task/subagent summaries.
- `DiffViewer.tsx`, `DiffBlock.tsx`, `DiagnosticsBlock.tsx`, and `CurrentTurnChangeCard.tsx` cover unified diff rows, diagnostics attachment blocks, plus a turn-level changed-file summary from existing diff blocks.
- `PermissionDialog.tsx`, `ApprovalDialog.tsx`, `PromptContent.tsx`, `PermissionRequestBlock.tsx`, `ApprovalRequestBlock.tsx`, and `AskUserQuestionBlock.tsx` cover prompt modal and historical prompt rendering, including selected option, multi-select, free-text, and unmatched-answer summaries.
- `Composer.tsx`, `SlashPopover.tsx`, `WorkspaceReferencePopover.tsx`, `ComposerInspectorPanel.tsx`, `AttachmentGallery.tsx`, `composerAttachments.ts`, and `workspaceReferences.ts` cover send/stop, attachments, slash completion, @workspace references, per-session drafts, local inspector panels, model chip, token ring, and project context chip.
- `WorkspaceRail.tsx`, `WorkspaceView.tsx`, settings/catalog views, API client, and stores provide the web shell around the chat path.

Latest focused verification after this audit update:

- `cd web && npm run typecheck`
- `cd web && npm test -- ToolResultPreview.test.tsx blocks.test.ts normalizeTranscript.test.ts EdgeBlocks.test.tsx RuntimeStatusBlocks.test.tsx ToolCallBlock.test.tsx ChatTimeline.test.tsx`

## Component Completion Matrix

| Area | Status | Evidence | Main remaining gaps |
| --- | --- | --- | --- |
| Aether-owned render contract | Mostly complete | `chat-rendering/blocks.ts`, reducer/normalizer tests | Needs browser acceptance across real persisted and live sessions. |
| User messages | Mostly complete | `UserMessageBlock`, `AttachmentGallery` | Editing/quoting/rewind actions are missing. |
| Assistant messages | Partial to mostly complete | `AssistantMessageBlock`, `MarkdownRenderer`, `MathRenderer`, `MermaidRenderer` | Visual acceptance for large diagrams, formula-heavy output, partial streaming edge cases. |
| Thinking/reasoning | Foundation complete | `ThinkingBlock`, `reasoning.delta` reducer | Needs richer collapse/history behavior and browser scroll tests. |
| Streaming status | Foundation complete | `StreamingStatusBlock`, token usage reducer | Token estimates are active-run oriented, not full prompt-budget reconstruction. |
| Tool call grouping | Partial | `ToolCallGroup`, `ToolCallBlock`, render model grouping | Nested/parallel lifecycle depth and richer durations/status metadata remain incomplete. |
| Shell/terminal results | Partial to mostly complete | `TerminalChrome` | Needs browser visual tests and more structured backend metadata. |
| File/search/web/tool previews | Partial, improved | `ToolResultPreview` | File edit, notebook edit, LSP, browser, file-read, search, nested web-search/web-fetch, task, todo, image artifacts, non-image artifact bundles, and spill notices now have first-pass previews. Remaining gaps are browser visual acceptance, richer notebook output/state visualization, and richer arbitrary artifact bundle actions. |
| Todo preview | Mostly complete | `TodoListPreview` | Needs parity validation against live tool payloads. |
| Subagent/task inline summary | Partial, improved | `InlineTaskSummary`, `TaskNotificationBlock`, `ToolResultPreview`, `SessionTaskBar`, `TaskDetailDialog` | Inline summaries, session task hierarchy, and parent/child drill-down are wired from current task APIs. A2A event timelines and richer nested message streams still need deeper backend integration. |
| Diff/code/diagnostics rendering | Partial | `DiffViewer`, `DiffBlock`, `DiagnosticsBlock`, `CurrentTurnChangeCard` | Diagnostics attachments no longer render as raw user prompts. No checkpoint/undo/revert/fork workflow yet; changed-file summary is derived from current diff blocks only. |
| Permission/approval | Foundation complete | `PermissionDialog`, `ApprovalDialog`, prompt blocks | Needs prompt queue stress tests, richer diff previews, and browser focus/accessibility acceptance. |
| Ask user question | Mostly complete | `AskUserQuestionBlock`, `PromptContent` | Historical blocks now show selected options, multi-select/free-text answers, and unmatched answer metadata. Remaining gap is live provider/browser acceptance. |
| Composer plus menu | Partial | `ComposerControlMenu` inside `Composer.tsx` | Not full cc-haha command surface yet; no repository launch controls. |
| Slash/local inspector panels | Partial to mostly complete | `ComposerInspectorPanel`, slash tests | MCP is explicit unavailable state until backend MCP routes exist. |
| Workspace references | Partial | `WorkspaceReferencePopover`, `ProjectContextChip` | Needs fuller file-menu navigation, directory traversal, selected context management, and visual acceptance. |
| Project/workspace shell | Partial | `WorkspaceRail`, `Sidebar`, `ChatWorkbenchHeader` | Needs repo/worktree launch, branch controls, and tighter Hermes-like layout polish. |
| Message action bar | Partial | `MessageActionBar`, `ChatTimeline`, `Composer` | Copy, quote, user edit-to-composer, and user retry are wired locally. Backend-backed rewind/fork/checkpoint and assistant retry are still missing. |
| Browser acceptance | Missing/weak | Unit tests and build only | Need Playwright/browser smoke for streaming, prompts, tool outputs, narrow width, scrolling, and workspace flows. |

## Important Missing Components Or Behaviors

1. Message lifecycle actions: local quote, user edit-to-composer, and user retry are now available. Assistant retry, fork, rewind, and checkpoint still require backend contracts before UI should pretend they work.
2. Full composer command surface: richer local panels, keyboard-first file menu, selected context chips, clear individual references, repository/worktree launch controls.
3. Rich edit/workflow cards: current changed-file card exists, but no checkpoint, undo, accept/reject per file, diagnostics follow-up, or post-edit verification bundle view.
4. Deep task/subagent UI: inline summaries, task detail drawer, session task hierarchy, and parent/child drill-down are now connected to current task APIs. A2A event timelines, nested child message streams, and richer result artifacts remain open.
5. MCP management: current web panel intentionally says unavailable. Complete MCP UI needs backend routes for configured servers, connection state, tools/resources, and credentials.
6. Tool-specific preview breadth: shell/read/search/web/todo/task/images plus file edit, notebook edit, LSP, browser-result, diagnostics, artifact bundle, and spill-notice previews are started; browser visual acceptance, provider-specific web-search edge cases, richer notebook output/state visualization, and richer artifact actions still need dedicated renderers.
7. True context accounting: current token ring and context panel use active-run/session metadata. They do not reconstruct full prompt budget, cache tokens, model window pressure, or compaction state with high fidelity.
8. Browser visual acceptance: most confidence is unit-test/build based. We still need screenshot/interaction tests for real chat turns and layout states.

## Recommended Next PR Order

### PR A - Browser Acceptance Baseline

Add Playwright or equivalent browser smoke for first load, session create/select, one streaming turn, permission modal, approval modal, workspace rail, composer wrapping, and representative chat blocks. This should happen early because many remaining gaps are visual regressions waiting to happen.

### PR B - Composer Workflow Parity

Finish file search/menu behavior, selected context chip management, per-reference clear actions, keyboard navigation polish, and responsive wrapping. Defer repo/worktree launch until backend routes are explicit.

### PR C - Rich Tool Preview Expansion

Continue beyond the first-pass file edit/write, notebook edit, LSP, browser/screenshot, nested web-search/web-fetch, image, diagnostics, artifact, spill-notice, and task previews. Add richer notebook output/state visualization, browser visual artifact routing, provider-specific edge cases, and richer artifact bundle actions. Keep generic JSON/code fallback for unknown tools.

### PR D - Task/Subagent Drill-Down

Connect inline task summaries to task detail data, nested child tasks, output tail/result files, model/provider, duration, token usage, and A2A-style task notifications where backend events exist.

### PR E - Current Turn Change Management

Promote `CurrentTurnChangeCard` from derived summary to a real workflow card once backend can expose changed files, checkpoints, diagnostics, undo/revert, and verification state.

### PR F - Message Actions

Add retry/edit/quote/rewind/fork only after session/run backend operations are defined. Avoid UI-only buttons that cannot execute real behavior.

## Short Answer

The migration is at a usable foundation stage, not completion. The core chat-rendering architecture is in place and several important components have been rebuilt in Aether-native TypeScript. However, many higher-level developer-console capabilities are still partial or missing, especially composer workflow parity, task/subagent drill-down, message actions, rich edit cards, MCP management, and browser-level acceptance.
