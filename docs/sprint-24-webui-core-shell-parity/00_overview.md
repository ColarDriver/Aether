# Sprint 24 - WebUI Core Shell Parity

## Goal

Align Aether web with the important core interface shape from
`/workspace/hermes-webui`, while keeping Aether's frontend implementation in
React and TypeScript.

Hermes WebUI is a useful reference for product shape:

- left navigation/sidebar for sessions and control surfaces,
- center chat as the primary artifact,
- right workspace/file rail for project context,
- composer footer as the command surface for model, workspace, attachments,
  context usage, stop, and send controls,
- quiet transcript metadata instead of making every internal event a large card.

Aether should adopt this interaction structure, not the implementation stack.
Hermes WebUI is Python plus vanilla JavaScript/CSS; Aether web remains
Vite/React/TypeScript with typed API contracts, Zustand stores, CSS variables,
and reusable TS components.

## Current State

Aether web already has:

- a React app shell with sidebar, top bar, status bar, and content pane,
- typed chat rendering blocks under `web/src/chat-rendering`,
- chat timeline components for user, assistant, thinking, tools, diffs,
  permissions, approvals, questions, tasks, and streaming state,
- a workspace browser page backed by `/api/workspace/*`,
- a composer with slash completion, attachments, file paste, and workspace
  reference insertion.

The remaining UI gap is not the chat event contract. The gap is the workbench
shape around it:

- chat is still a two-panel page instead of a three-panel workbench,
- workspace context is a separate full page instead of an adjacent rail,
- composer controls are too action-only and do not expose model/workspace/context
  state where users naturally compose,
- sidebar controls are not organized like a compact control center,
- responsive behavior needs to preserve the recently fixed chat scroll/input
  layout while adding the right rail.

## TypeScript Boundary

Do not copy Hermes WebUI's `static/*.js` into Aether.

The implementation boundary for this sprint is:

```text
web/src/components/layout/
  Sidebar.tsx
  TopBar.tsx
  StatusBar.tsx

web/src/components/chat/
  ChatView.tsx
  Composer.tsx
  WorkspaceRail.tsx

web/src/components/shared/
  reusable buttons, appearance controls, and future shell primitives

web/src/styles.css
  shell tokens, responsive layout, and component styling
```

If the shell and chat components later need to serve another frontend, extract a
TypeScript package after the contracts settle:

```text
packages/aether-web-shell/
packages/aether-chat-renderer/
```

This sprint should first stabilize the components in `web/src`.

## Non-Goals

- No Ink clone for the browser.
- No vanilla JS port of Hermes WebUI.
- No wholesale copy of Hermes themes, i18n, service worker, terminal, mobile
  drawer, kanban, cron, profile, or dashboard code.
- No backend workspace mutation or editor support in the right rail.
- No large dependency addition for layout or rendering.

## Acceptance

- Chat route has a three-panel desktop shape: sidebar, chat, workspace rail.
- Chat timeline and composer remain internally scroll-stable and visible.
- Right rail can browse/search/preview workspace files using existing Aether API.
- Composer footer displays current provider/model, workspace context affordance,
  context usage indicator, attachments, Stop, and Send.
- Narrow layouts collapse the right rail without hiding the composer.
- Existing chat, composer, workspace, and sidebar tests continue to pass.
