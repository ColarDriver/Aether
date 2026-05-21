# Sprint 22 - Web Composer Command Input

## Goal

Bring the Aether web composer closer to the web/desktop references in
`/workspace/hermes-agent` and `/workspace/cc-haha` while keeping the
implementation browser-native and Aether-owned.

Sprint 21 made the timeline capable of rendering agent output. Sprint 22 focuses
on the input side:

- slash command discovery and completion,
- keyboard-safe composer behavior,
- slash execution into timeline-visible notices or agent runs,
- file/image attachment input,
- workspace file references,
- paste and drag/drop handling,
- focused tests for command and attachment input flows.

## Current State

Aether web currently has a minimal composer:

- one textarea,
- Enter to send,
- Shift+Enter for newline,
- send/stop button,
- no slash completion,
- no command execution pipeline,
- no composer attachments,
- no file search or workspace reference selection.

The backend already exposes slash metadata through gateway
`commands.catalog`, but the web backend does not expose a browser REST endpoint
for the same catalog.

## Reference Analysis

`/workspace/hermes-agent` has a compact web-specific shape:

- `SlashPopover.tsx` fetches command completions and owns keyboard selection.
- `slashExec.ts` separates slash parsing/execution from the composer view.
- The web chat page keeps slash handling explicit instead of treating slash
  input as normal user messages.

`/workspace/cc-haha` has a broader desktop composer:

- `composerUtils.ts` contains pure slash-token parsing and replacement helpers.
- `ChatInput.tsx` manages slash menus, file attachments, workspace references,
  paste handling, draft persistence, and local command panels.
- `AttachmentGallery.tsx` renders composer and message attachments.

Aether should take the architecture, not the shell:

- no Tauri-only desktop APIs in the browser console,
- no Material Symbols or Tailwind dependency,
- no gateway JSON-RPC dependency in web routes,
- no huge markdown/highlight packages for composer UI,
- keep the current REST/WebSocket web service boundary.

## PR Boundaries

1. Add a REST command catalog and TypeScript slash completion primitives.
2. Add the composer popover and keyboard integration.
3. Add slash execution for local/session commands and timeline notices.
4. Add composer attachments and workspace file references.
5. Add automated and manual acceptance coverage.

## Acceptance

- Typing `/` in the web composer shows available Aether slash commands.
- Filtering is incremental, stable, and backed by the Python command catalog.
- Arrow keys, Tab/Enter, Escape, and click behave predictably.
- Command completion does not submit accidental normal agent messages.
- Later PRs execute supported commands through web-native services and render
  durable timeline feedback.
- File and image attachments can be added before sending and later render as
  Sprint 21 user attachments.
