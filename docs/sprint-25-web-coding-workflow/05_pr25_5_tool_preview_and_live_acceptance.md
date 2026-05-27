# PR25.5 - Tool Preview And Live Acceptance

## Objective

Finish the renderer and verification loop for live coding output. Known
Aether/Codex/Anthropic/OpenAI-compatible/Brave/Tavily/Bocha/browser/notebook
payloads should render as useful previews instead of raw JSON, and browser/live
acceptance should prove the web console works under real streaming, permission,
approval, provider-error, task, context, and workspace workflows.

## Current State

Already present or partially implemented:

- shell/terminal previews;
- file read/edit/write previews;
- search/grep previews;
- web search/fetch previews;
- provider web-search metadata;
- Codex/Anthropic hosted search metadata;
- Brave/Tavily/Bocha/Aether result shapes;
- notebook source/diff/output/lifecycle previews;
- browser screenshots and structured artifacts;
- LSP/diagnostics;
- todo and task/subagent previews;
- image artifacts and non-image artifact bundles;
- spill notices;
- generic JSON/code fallback;
- browser smoke for App shell, persisted transcript blocks, rich Markdown,
  notebook output, provider web-search rendering, long-history internal
  scrolling, mobile overflow containment, narrow workspace references, long
  terminal containment, primary navigation, task drill-down, session delete,
  resizable panels, non-streamed `run.result`, live streaming, permission,
  approval, queued prompt sequencing, workspace rail, and screenshot baselines.

Main gaps:

- provider-specific edge cases are only covered by representative fixtures;
- many artifact and notebook variants lack browser visual acceptance;
- arbitrary artifact bundles need richer type-specific actions;
- real-provider/manual acceptance is incomplete;
- accessibility and keyboard review for prompt/tool surfaces is incomplete.

## Scope

Primary files:

- `web/src/chat-rendering/blocks.ts`
- `web/src/chat-rendering/blockReducer.ts`
- `web/src/chat-rendering/normalizeTranscript.ts`
- `web/src/components/chat/ChatTimeline.tsx`
- `web/src/components/chat/blocks/ToolResultPreview.tsx`
- `web/src/components/chat/blocks/ToolResultBlock.tsx`
- `web/src/components/chat/blocks/ToolCallBlock.tsx`
- `web/src/components/chat/TerminalChrome.tsx`
- `web/src/components/chat/MarkdownRenderer.tsx`
- `web/src/components/chat/TaskDetailDialog.tsx`
- `web/src/api/types.ts`
- `web/e2e/chat-console.spec.ts`
- `web/playwright.config.ts`
- `docs/web-component-migration-gap-audit/README.md`

Preview variants to harden:

- web search:
  - OpenAI-compatible hosted web-search payloads;
  - Codex citations and source metadata;
  - Anthropic server-tool citations;
  - Brave raw results;
  - Tavily raw and summarized results;
  - Bocha raw result envelopes;
  - failed/partial search with query metadata.
- browser:
  - screenshot-only result;
  - URL plus screenshot;
  - page title plus markdown extraction;
  - multiple image artifacts;
  - failed navigation with diagnostics.
- notebook:
  - queued/running/completed/failed execution;
  - stdout/stderr streams;
  - rich output image;
  - traceback;
  - hidden/truncated output;
  - cell deletion/insert/update diff.
- artifacts:
  - inline text;
  - inline JSON;
  - image;
  - binary local path copy;
  - remote URL open;
  - multiple artifact bundle;
  - unsafe path blocked.
- LSP/diagnostics:
  - grouped severity counts;
  - code-action hints if present;
  - file/line links where workspace path is valid.

Renderer rules:

- known variants get concise structured previews;
- unknown variants still fall back to stable JSON/code;
- never trust URLs or local paths blindly;
- previews should be collapsed or summarized enough to keep the timeline usable;
- copy/open actions must be type-specific and safe;
- streaming partial results must not flicker, duplicate, or jump alignment.

## Browser And Live Acceptance

Automated browser scenarios:

- workspace root switch and run CWD propagation;
- stale prompt after simulated backend restart;
- accept/reject changed file with conflict;
- checkpoint-backed undo/retry;
- nested task/artifact drill-down;
- context pressure and compression;
- MCP server validation/resource read;
- provider web-search payload variants;
- notebook failed execution;
- artifact bundle;
- mobile and narrow-width prompt/composer behavior;
- long live stream without scroll hijacking inside the web scroll container.

Screenshot matrix:

- desktop chat shell;
- narrow chat shell;
- permission modal with diff;
- approval modal;
- ask-user-question prompt;
- changed-file card;
- task detail dialog;
- workspace file preview/edit;
- context inspector;
- MCP inspector;
- artifact preview;
- provider error block.

Live-provider scripts:

- OpenAI-compatible successful short answer;
- OpenAI-compatible provider HTTP error;
- Codex provider with web search if configured;
- Anthropic provider with hosted/web-search metadata if configured;
- plan mode approval;
- shell permission prompt;
- file edit permission and diff rendering;
- subagent/task run;
- workspace attachment run.

Each manual run records command, provider/model, prompt, screenshots or notes,
pass/fail, and linked issue if failed. Do not store secrets in screenshots or
logs.

## Tests

TypeScript:

- fixture tests for each known provider payload shape;
- malformed payloads do not crash previews;
- unsafe href/path is not clickable;
- raw JSON fallback remains available;
- streaming partial tool results do not flicker or duplicate.

Browser:

- screenshot fixtures for web search cards;
- browser result with screenshot;
- notebook failed execution;
- artifact bundle;
- narrow-width terminal/tool containment;
- long live stream stability.

Repository verification:

```bash
cd web && npm run typecheck
cd web && npm test
cd web && npm run test:e2e
python -m pytest aether/tests/web aether/tests/services -q
git diff --check
```

Optional live script once the provider harness exists:

```bash
python scripts/web_live_acceptance.py --provider openai-compatible --model "$AETHER_MODEL"
```

## Acceptance

- Known provider/tool outputs render as useful summaries.
- Unknown outputs remain readable and inspectable.
- Tool previews stay aligned with surrounding timeline cards.
- Rendering does not regress streaming timeline stability.
- Every PR in this sprint has at least one automated test and one browser/manual
  acceptance note.
- Live-provider failures produce structured, actionable UI blocks.
- No visual acceptance path exposes raw credentials.

## Explicit Exclusions

- Implementing new backend tools.
- Provider SDK changes unrelated to rendering/metadata.
- Full notebook editor replacement.
- Full cross-browser matrix beyond Chromium unless a regression justifies it.
- Hosted CI infrastructure.
- Performance benchmarking beyond functional UI stability.
