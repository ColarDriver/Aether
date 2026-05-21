# PR21.6 - Chat Timeline Integration and Scroll Behavior

## Goal

Wire the new block renderer into `ChatView` and remove the old split
message/tool rendering path.

## Required Files

Modify:

```text
web/src/components/chat/ChatView.tsx
web/src/components/chat/MessageList.tsx
web/src/stores/chatStore.ts
web/src/styles.css
```

Delete or turn into compatibility wrappers when no longer needed:

```text
web/src/components/chat/ToolCallBlock.tsx
web/src/components/chat/DiffViewer.tsx
```

Only delete old files if imports have moved cleanly to `blocks/*`.

## Integration Steps

1. `ChatView` reads `blocksBySession[sessionId]`.
2. `ChatView` renders `ChatTimeline` instead of separate `MessageList` and
   `tool-stack`.
3. `ChatTimeline` receives prompt response callbacks from the store.
4. `pendingPermission` and `pendingApproval` overlays, if still used, are
   derived from the same prompt blocks.
5. Remove direct top-level rendering of `tools.map(...)`.
6. Remove old `messagesBySession` and `toolsBySession` once tests no longer
   depend on them.

## Scroll Behavior

Browser chat should not repeat the TUI scrollback problems.

Rules:

- The chat scroll container should auto-scroll only when the user is already
  near the bottom.
- If the user scrolls up, streaming deltas should not yank the scroll position.
- A "jump to latest" button should appear when user is away from bottom and new
  output arrives.
- Prompt modals should trap focus but must not reset chat scroll when opened or
  closed.
- Expanding a thinking/tool/diff block should not force the outer scroll unless
  the user explicitly opens content at the bottom.
- Store per-session scroll snapshot if switching sessions/tabs is introduced in
  the web app.

Use cc-haha's scroll model as reference:

- `isNearScrollBottom(...)`,
- `shouldAutoScrollRef`,
- `isProgrammaticScrollingRef`,
- session scroll snapshots,
- `scrollToBottom('smooth' | 'auto')`.

Implement it in Aether naming and without importing unrelated cc-haha stores.

## Streaming Status Placement

Render streaming status as part of the timeline:

- below the active thinking block while reasoning,
- below the active assistant block while responding,
- below the relevant tool group while executing tools,
- with token usage and elapsed time when available.

Do not display duplicate token/status pills in multiple places unless they serve
different purposes. The chat header can keep a compact run summary; the timeline
status should show immediate activity.

## Empty and Loading States

States:

- no selected session -> existing `EmptyState`,
- selected session with no blocks -> "No messages in this session yet",
- transcript loading -> subtle loading state,
- transcript load error -> visible retry affordance if store exposes retry.

## Tests

Add tests for:

- `ChatView` renders `ChatTimeline`,
- tool blocks appear in chronological order between assistant messages,
- no separate `tool-stack` duplicate exists,
- active streaming status renders once,
- prompt overlay does not remove timeline prompt block,
- auto-scroll only runs when near bottom,
- jump-to-latest appears when user scrolls away,
- switching sessions restores or resets scroll intentionally.

## Acceptance

- The browser chat is driven by one timeline model.
- User, assistant, thinking, tool, result, diff, permission, approval, question,
  and streaming blocks coexist in order.
- Long streaming turns do not take scroll control away from the user.
