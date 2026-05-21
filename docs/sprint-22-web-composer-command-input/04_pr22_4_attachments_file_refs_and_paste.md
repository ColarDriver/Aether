# PR22.4 - Attachments, File References, And Paste

## Scope

Add composer-side attachment input so Sprint 21's user attachment renderer has a
complete source.

## Frontend Work

- Add composer attachment state.
- Add file input for images and text/files.
- Add paste handling for images.
- Add drag/drop handling.
- Reuse `AttachmentGallery` for composer previews.
- Convert selected attachments to run-start payload metadata or structured
  user content once the backend run channel supports attachments.

## Backend Work

- Extend the run WebSocket `run.start` payload to carry attachments.
- Store display attachments in the persisted user message metadata so resume
  can render the same message.
- Ensure image data URLs are compatible with provider transport image handling.

## Workspace References

- Add file search or workspace browser insertion in a follow-up within this PR
  if the basic attachment payload is stable.
- Prefer a small API-backed search surface over copying cc-haha desktop-only
  file search behavior.

## Tests

- File attachment preview appears before send.
- Image paste produces an image attachment preview.
- Sent attachments are present in optimistic user block.
- Resumed transcript renders the same attachments.

## Current Implementation Evidence

Implemented on branch `web-console-migration`:

- Composer attachment input:
  - `web/src/components/chat/Composer.tsx`
  - `web/src/components/chat/composerAttachments.ts`
  - `web/src/components/chat/AttachmentGallery.tsx`
  - `web/src/styles.css`
- Run payload and optimistic timeline attachments:
  - `web/src/api/runSocket.ts`
  - `web/src/api/types.ts`
  - `web/src/stores/chatStore.ts`
  - `web/src/components/chat/ChatView.tsx`
- Backend display persistence:
  - `aether/web/ws/runs.py`
  - `aether/services/runs/contracts.py`
  - `aether/services/runs/builder.py`
  - `aether/runtime/core/contracts.py`
  - `aether/agents/core/agent.py`

Notes:

- Attachments are persisted as user-message `metadata.displayAttachments`.
- The first implementation keeps provider-bound prompt content textual and
  stores attachments for display/resume. Provider-native multimodal dispatch is
  intentionally left to the provider transport layer so web metadata does not
  silently change model payload semantics.
