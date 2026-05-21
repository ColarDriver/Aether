# PR24-2 - Three-Panel Chat Workbench

## Objective

Make the chat route a Hermes-style workbench: left sidebar, center chat, right
workspace rail.

## Implementation

- Add `web/src/components/chat/WorkspaceRail.tsx`.
- Use existing Aether workspace endpoints:
  - `api.workspaceTree`
  - `api.workspaceFile`
  - `api.workspaceSearch`
- Keep the rail read-only in this PR.
- In `App.tsx`, render the chat route as:

```tsx
<div className="chat-workbench">
  <ChatView session={activeSession} />
  <WorkspaceRail onOpenWorkspace={() => setActiveView('workspace')} />
</div>
```

- Do not render the rail for non-chat settings pages.
- Preserve `content-pane-chat` as an overflow-hidden container.
- On narrower viewports, collapse or hide the right rail so the composer remains
  visible and usable.

## Workspace Rail Behavior

- Header shows `Workspace`, root path, refresh, and open-full-workspace action.
- Browser list supports directories and files.
- Parent navigation is available when not at root.
- Search shows path matches and can be cleared.
- File preview shows markdown through the existing `MarkdownRenderer`, code in a
  scrollable `<pre>`, and a clear binary/truncated state.
- API failures render a compact error row instead of breaking the chat page.

## Tests

- Add a `WorkspaceRail` component test for:
  - initial tree load,
  - file preview,
  - directory navigation,
  - search.
- Existing `WorkspaceView` tests remain unchanged.

## Acceptance

- The chat page is usable with and without workspace files.
- Right rail failures do not hide chat.
- Resizing below desktop width hides the rail before the composer becomes
  unusable.
