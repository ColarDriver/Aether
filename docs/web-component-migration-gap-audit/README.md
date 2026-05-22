# Web Component Migration Gap Audit

Status date: 2026-05-21
Branch observed: `web-console-migration`

## Bottom Line

Aether web has migrated the core chat/runtime path: web shell, REST and WebSocket clients, typed chat rendering blocks, live transcript rendering, permission and approval prompts, basic diff rendering, workspace browsing, and settings/catalog views.

It has not completed full cc-haha or Hermes-style developer console parity. The missing work is mostly higher-level workflow components around the composer, local inspector panels, current-turn change management, richer tool rendering, media/markdown rendering, and browser-level acceptance coverage.


## Implementation Progress After Initial Audit

Additional work completed on this branch:

- Added Aether-native composer inspector panels for `/status`, `/context`, `/cost`, `/skills`, and `/mcp`. The first four use existing backend APIs; `/mcp` is an explicit unavailable-state panel until backend MCP routes exist.
- Local inspector commands are now included in slash completion and open in the composer without sending an agent run or appending transcript notices.
- Added a composer `+` control menu for attach files, slash command insertion, workspace reference insertion, and local inspector panels.
- Added per-session composer drafts for text and attachments while switching sessions in the mounted chat workbench.
- Added `TerminalChrome` for shell/exec-style tool results, including command header, exit status, duration, copy, and expandable truncation. Shell calls no longer duplicate raw JSON input when the terminal chrome already carries the command.
- Added `ToolResultPreview` for common non-shell tools: file reads, grep/search output, web search/fetch JSON results, and subagent/task summaries. Unknown tools still fall back to the generic code block.
- Added `CurrentTurnChangeCard` at the chat-turn level. It summarizes changed files, addition/removal totals, per-file stats, and expandable per-file diffs derived from existing diff blocks.
- Added Markdown image rendering for `![alt](url)` and bare image URLs, with safe-source filtering and a keyboard/navigable lightbox.
- Added Aether-native Mermaid rendering for `mermaid` fenced blocks and unlabeled diagram fences, with sanitized SVG output, loading/error fallback, copy, and zoomable preview.
- Added KaTeX-backed math rendering for inline `$...# Web Component Migration Gap Audit

Status date: 2026-05-21
Branch observed: `web-console-migration`

## Bottom Line

Aether web has migrated the core chat/runtime path: web shell, REST and WebSocket clients, typed chat rendering blocks, live transcript rendering, permission and approval prompts, basic diff rendering, workspace browsing, and settings/catalog views.

It has not completed full cc-haha or Hermes-style developer console parity. The missing work is mostly higher-level workflow components around the composer, local inspector panels, current-turn change management, richer tool rendering, media/markdown rendering, and browser-level acceptance coverage.


## Implementation Progress After Initial Audit

Additional work completed on this branch:

- Added Aether-native composer inspector panels for `/status`, `/context`, `/cost`, `/skills`, and `/mcp`. The first four use existing backend APIs; `/mcp` is an explicit unavailable-state panel until backend MCP routes exist.
- Local inspector commands are now included in slash completion and open in the composer without sending an agent run or appending transcript notices.
- Added a composer `+` control menu for attach files, slash command insertion, workspace reference insertion, and local inspector panels.
- Added per-session composer drafts for text and attachments while switching sessions in the mounted chat workbench.
- Added `TerminalChrome` for shell/exec-style tool results, including command header, exit status, duration, copy, and expandable truncation. Shell calls no longer duplicate raw JSON input when the terminal chrome already carries the command.
- Added `ToolResultPreview` for common non-shell tools: file reads, grep/search output, web search/fetch JSON results, and subagent/task summaries. Unknown tools still fall back to the generic code block.
- Added `CurrentTurnChangeCard` at the chat-turn level. It summarizes changed files, addition/removal totals, per-file stats, and expandable per-file diffs derived from existing diff blocks.
- Added Markdown image rendering for `![alt](url)` and bare image URLs, with safe-source filtering and a keyboard/navigable lightbox.
, inline `\(...\)`, and display `$...$` / `\[...\]` math blocks, with sanitized output and raw fallback.
- Verified the updated composer path with `npm test -- Composer.test.tsx` and `npm run typecheck`.
- Verified terminal and preview tool rendering with `npm test -- ToolResultPreview.test.tsx ToolCallBlock.test.tsx TerminalChrome.test.tsx` and `npm run build`.
- Verified changed-file summaries with `npm test -- CurrentTurnChangeCard.test.tsx ChatTimeline.test.tsx` and `npm run build`.
- Verified Markdown image rendering with `npm test -- MarkdownRenderer.test.tsx` and `npm run build`.
- Verified Mermaid and math rendering with `npm test -- MathRenderer.test.tsx MarkdownRenderer.test.tsx MarkdownRenderer.mermaid.test.tsx MermaidRenderer.test.tsx`, `npm run typecheck`, and `npm run build`.
- Verified structured tool image previews with `npm test -- ToolResultPreview.test.tsx ToolResultBlock.test.tsx MathRenderer.test.tsx MarkdownRenderer.test.tsx` and `npm run typecheck`.
- Verified lazy Mermaid/KaTeX loading with `npm test -- MathRenderer.test.tsx MermaidRenderer.test.tsx MarkdownRenderer.test.tsx MarkdownRenderer.mermaid.test.tsx ToolResultPreview.test.tsx`, `npm run typecheck`, and `npm run build`.

These changes move PR A to implemented, start PR B, and land the first slice of PR D. PR B is not complete yet: fuller file search navigation, project context chips, and repository/worktree launch controls remain separate work. PR C now has the frontend-only summary layer implemented from existing diff blocks; undo/checkpoint remains backend-dependent. PR D is still not complete yet: the first search/web/subagent/file-read previews exist, but richer per-tool lifecycle metadata, nested task output, and backend-shaped structured payloads remain separate work.

## Evidence Inspected

Aether current files:

- `web/src/chat-rendering/*`
- `web/src/components/chat/*`
- `web/src/components/chat/blocks/*`
- `web/src/components/layout/*`
- `web/src/components/settings/*`
- `web/src/api/*`
- `web/src/stores/*`
- `docs/sprint-20-web-console-migration/99_acceptance_matrix.md`
- `docs/sprint-21-web-chat-component-system/99_acceptance_matrix.md`
- `docs/sprint-24-webui-core-shell-parity/99_acceptance_matrix.md`

Reference files:

- `/workspace/cc-haha/desktop/src/components/chat/*`
- `/workspace/cc-haha/desktop/src/components/markdown/MarkdownRenderer.tsx`
- `/workspace/cc-haha/desktop/src/components/shared/*`
- `/workspace/cc-haha/desktop/src/components/workspace/*`
- `/workspace/hermes-webui/static/*`

## Migrated Or Rebuilt In Aether

| Area | Status | Aether evidence | Notes |
| --- | --- | --- | --- |
| Web shell | Mostly migrated | `App.tsx`, `Sidebar.tsx`, `ChatWorkbenchHeader.tsx`, `WorkspaceRail.tsx`, `styles.css` | Three-panel workbench exists; details continue to move. |
| API client | Mostly migrated | `api/client.ts`, `api/runSocket.ts`, `api/types.ts` | Covers sessions, runs, providers, tools, skills, tasks, plan, logs, analytics, docs, workspace, env/config. |
| Chat render contract | Migrated foundation | `chat-rendering/blocks.ts`, `blockReducer.ts`, `normalizeTranscript.ts`, `renderModel.ts` | Good Aether-owned TS boundary; no Ink dependency. |
| Timeline integration | Migrated foundation | `ChatTimeline.tsx`, `ChatView.tsx`, `chatStore.ts` | User/assistant/thinking/tool/diff/prompt/status blocks render through one timeline. |
| User and assistant messages | Migrated foundation | `UserMessageBlock.tsx`, `AssistantMessageBlock.tsx`, `MarkdownRenderer.tsx` | Copy action exists; richer actions are missing. |
| Thinking and streaming status | Migrated foundation | `ThinkingBlock.tsx`, `StreamingStatusBlock.tsx`, `blockReducer.ts` | Basic DOM rendering exists. |
| Tool calls and results | Partial | `ToolCallBlock.tsx`, `ToolCallGroup.tsx`, `ToolResultBlock.tsx`, `TodoListPreview.tsx`, `TerminalChrome.tsx`, `ToolResultPreview.tsx` | Basic cards, todo preview, shell terminal chrome, and first non-shell previews exist; richer nested lifecycle remains missing. |
| Diff rendering | Partial | `DiffViewer.tsx`, `DiffBlock.tsx` | Unified diff rows, markers, line numbers exist; change-summary workflow is missing. |
| Permission and approval | Mostly migrated foundation | `PermissionDialog.tsx`, `ApprovalDialog.tsx`, `PromptContent.tsx`, prompt blocks | Modal plus timeline history exist; advanced prompt queue and computer-use-specific UI are not present. |
| Ask user question | Migrated foundation | `AskUserQuestionBlock.tsx`, approval question support | Basic question display exists. |
| Composer | Partial | `Composer.tsx`, `SlashPopover.tsx`, `WorkspaceReferencePopover.tsx` | Send/stop, attachments, slash, model chip, token ring, @path references exist. |
| Workspace rail | Partial to mostly migrated | `WorkspaceRail.tsx`, `WorkspaceView.tsx`, workspace API | Tree/search/preview exists; file actions and worktree management are missing. |
| Settings/catalog views | Mostly migrated foundation | `ProviderSettings.tsx`, `ToolsView.tsx`, `SkillsView.tsx`, `EnvironmentView.tsx`, `LogsView.tsx`, `AnalyticsView.tsx` | Useful service-backed views exist, but not full cc-haha/Hermes domain surface. |

## Important Missing Or Incomplete Components

| Reference capability | Reference evidence | Aether state | Required next work |
| --- | --- | --- | --- |
| Local slash inspector panels | `cc-haha/.../LocalSlashCommandPanel.tsx` | Missing | Add Aether panels for `/status`, `/context`, `/cost`, `/skills`, and `/mcp` where backend data exists. |
| Context usage indicator | `ContextUsageIndicator.tsx` | Partial token ring only | Replace simple active-run percentage with model context window, input/output/cache usage, and live session estimate. |
| Composer plus menu | `ChatInput.tsx` plus menu state | Missing | Add attach, slash, workspace/file search, and local inspector commands as first-class footer controls. |
| Per-session composer drafts | `ChatInput.tsx` draft refs | Missing | Preserve draft text and attachments when switching sessions. |
| Full file search menu | `FileSearchMenu.tsx` | Partial via `WorkspaceReferencePopover.tsx` | Add navigable search results, keyboard support, directory traversal, and selected context chips. |
| Project context chip | `ProjectContextChip.tsx` | Partial workspace chip only | Show active root/workspace, missing workspace state, and context clear/remove actions. |
| Repository launch controls | `RepositoryLaunchControls.tsx` | Missing | Support empty-session workdir/branch/worktree selection only if Aether backend exposes compatible session creation. |
| Current turn change card | `CurrentTurnChangeCard.tsx` | Partial | Turn-level changed-file summary, insertion/deletion counts, and expandable per-file diffs exist; undo/checkpoint hooks need backend support. |
| Message action bar expansion | `MessageActionBar.tsx`, `MessageList.tsx` | Copy only | Add retry, edit, quote, rewind/fork/checkpoint only after backend contracts are real. |
| Terminal chrome | `TerminalChrome.tsx`, `ToolCallBlock.tsx` | Partial | Shell/exec output now has terminal chrome with command/status/duration/copy/truncation; structured payload and nested lifecycle depth still need work. |
| Inline task summary | `InlineTaskSummary.tsx` | Missing | Render task/subagent summaries inline with status, output tail, model, duration. |
| Inline image gallery | `MarkdownRenderer.tsx`, `AttachmentGallery.tsx`, `ToolResultPreview.tsx` | Partial | User attachments, Markdown assistant images, and structured tool-output image references now have preview/lightbox behavior; remaining work is backend-shaped artifact metadata and browser visual coverage. |
| Mermaid rendering | `MermaidRenderer.tsx`, markdown renderer | Partial | Aether now renders Mermaid fenced blocks with sanitized SVG, loading/error fallback, copy, zoomable preview, and async Mermaid loading; remaining work is visual polish, CSP review, and browser acceptance. |
| Math rendering | Hermes/markdown parity expectation | Partial | KaTeX-backed inline and display math now renders with sanitized output, fallback, and async KaTeX/CSS loading. Remaining work is broader browser visual coverage. |
| MCP management UI | cc-haha local panel, Hermes MCP CSS/API | Missing | Needs backend MCP service/routes first or explicit no-MCP product boundary. |
| Browser E2E screenshots | Sprint 21/24 acceptance notes | Missing | Add Playwright or browser-level smoke for chat turn, streaming, permission, approval, workspace, narrow viewport. |

## Completion Assessment

### What is complete enough to rely on

- Starting and serving the web console.
- Listing and creating sessions through the web APIs.
- Sending a basic run through WebSocket and rendering streamed output.
- Rendering persisted transcript messages into a typed timeline.
- Rendering basic tool calls/results, diffs, permissions, approvals, and questions.
- Browsing workspace files and viewing settings/catalog pages.

### What is not complete

- The composer is not cc-haha-equivalent. It is usable, but not yet a full command/control surface.
- The chat timeline is structurally right, but tool-specific rendering is not rich enough for coding workflows.
- Turn-level code change management is missing.
- Message lifecycle actions are minimal.
- Markdown/media support now covers common text Markdown, inline image/lightbox behavior, structured tool-output image references, Mermaid diagram rendering, and KaTeX math rendering. Mermaid and KaTeX are now lazy-loaded. Remaining media gaps are mostly backend-shaped artifact metadata and browser visual acceptance.
- MCP and advanced session inspection need backend/frontend contracts before meaningful UI parity.
- Visual acceptance is still weak because current verification is mostly unit tests and build checks.

## Recommended PR Sequence

### PR A - Composer Local Inspector Panels

Goal: bring over the most important `LocalSlashCommandPanel` behavior in Aether-native TS/React.

Scope:

- Add `ComposerInspectorPanel.tsx` for `/status`, `/context`, `/cost`, `/skills`, and a clearly bounded `/mcp` placeholder or endpoint-backed view.
- Extend slash execution so local panel commands open a panel instead of creating transcript notices.
- Use existing APIs first: health/status/current provider, analytics, skills, tools, session metadata, active run token usage.
- Do not invent fake MCP data. If no MCP route exists, show an explicit unavailable state and document backend requirement.
- Add component tests and composer interaction tests.

Acceptance:

- Typing `/status`, `/context`, `/cost`, `/skills` opens a panel above the composer.
- Panels are keyboard dismissible and do not send a model run.
- Existing slash commands such as `/plan` continue to work.

### PR B - Composer Workflow Parity

Goal: close the largest composer UX gaps from cc-haha while staying compatible with Aether backend.

Scope:

- Add plus menu for attach, slash, workspace/file search, and local panels.
- Preserve per-session drafts and attachments.
- Upgrade `WorkspaceReferencePopover` into a fuller file search menu with keyboard navigation and context chips.
- Add project/workspace context chip with clear/remove affordances.
- Defer repository branch/worktree launch controls unless backend routes are confirmed.

Acceptance:

- Switching sessions preserves draft text and attachments.
- `@` file search can be navigated with keyboard and selected into composer context.
- Footer wraps safely on narrow widths.

### PR C - Current Turn Change Management

Goal: add cc-haha-style changed-file summary after coding turns.

Scope:

- Define Aether backend contract for turn change summaries if it does not exist.
- Add `CurrentTurnChangeCard.tsx` with changed files, insertions/deletions, expandable diff preview, loading/error states.
- Wire it into `ChatTimeline` or `ChatView` near the relevant turn.
- Add tests for path relativization, diff loading, and display states.

Acceptance:

- After a turn with file edits, web shows a compact changed-files card.
- Expanding a file displays its diff without leaving chat.

### PR D - Tool Rendering Upgrade

Goal: make tool output readable as coding UI rather than generic JSON cards.

Scope:

- Add `TerminalChrome.tsx` for shell/exec output.
- Add per-tool previews for shell, file read, search, web search/fetch, subagent/task, todo, and file edit.
- Add output truncation with expand/copy controls.
- Show durations/status where event metadata provides it.

Acceptance:

- Shell output appears in terminal chrome.
- Subagent/task events render as inline task summaries with status and output tail.
- Generic JSON remains available as fallback.

### PR E - Markdown And Media Upgrade

Goal: close rich assistant-output rendering gaps.

Scope:

- Decide whether to keep custom parser or add a markdown pipeline.
- Add Mermaid support with safe fallback.
- Harden inline image/gallery behavior for backend-shaped assistant and tool artifact references.
- Add browser-level visual acceptance for formula-heavy assistant output and large diagrams.

Acceptance:

- Mermaid fenced blocks render as diagrams or explicit safe fallback.
- Image references render as inspectable gallery items.

### PR F - Browser Acceptance Suite

Goal: prove full workflows visually and interactively.

Scope:

- Add browser smoke for first load, session creation, streaming turn, permission, approval, workspace rail, local inspector panel, and responsive composer.
- Include screenshots or DOM assertions for representative chat blocks.
- Keep unit tests for renderer reduction behavior.

Acceptance:

- `npm test`, `npm run build`, and browser smoke pass locally.
- Visual regressions around composer/timeline are caught before merge.

## Current Risk Notes

- The current worktree is dirty; some files include user or previous-agent edits. Future implementation should split docs-only commits from runtime commits and avoid rewriting unrelated UI changes.
- Some cc-haha features depend on backend contracts Aether may not have yet, especially checkpoint diffs, MCP inspection, repository launch/worktree flows, and message rewind/fork. These need backend-first or explicit product-boundary decisions.
- Sprint 21 documents describe the renderer foundation as implemented, but their remaining-hardening section is still accurate: browser-level evidence and individual block coverage are incomplete.

## Short Answer

Migration is roughly at the usable foundation stage, not complete parity. The web console can render and run Aether chat workflows, but the advanced cc-haha developer-console components are still mostly unimplemented or only lightly represented. The next concrete step should be PR A, Composer Local Inspector Panels, followed by composer workflow parity and current-turn change management.
