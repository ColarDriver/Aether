# PR24-3 - Composer Footer Command Surface

## Objective

Move the visible command state toward Hermes WebUI's composer footer model while
preserving Aether's current slash, attachment, paste, and send behavior.

## Implementation

- Extend `Composer` props with optional command-state metadata:
  - `provider`
  - `model`
  - `mode`
  - `inputTokens`
  - `outputTokens`
- Render a footer under the textarea with:
  - provider/model chip,
  - workspace chip,
  - plan/agent mode chip when applicable,
  - context usage ring,
  - attach button,
  - Stop or Send button.
- Keep hidden file input and attachment gallery behavior unchanged.
- Keep slash popover and workspace reference popover anchored to the composer.
- Avoid making footer controls stateful unless a backed API already exists.

## Context Ring

First pass uses available token usage from the active run:

- display total active-run input/output token count,
- use a bounded percentage for the ring visualization,
- show a tooltip/title that makes clear this is active run usage, not full model
  context accounting.

Future PRs can wire full context length and compression thresholds from backend
metadata.

## Tests

- Existing composer send/slash/attachment tests must pass.
- Add a test that provider/model and context ring render when props are supplied.

## Acceptance

- Enter sends the same way as before.
- `/plan`, `/help`, and absolute paths keep their current routing.
- Attach, paste, and workspace references keep working.
- Footer wraps or compacts without overlapping the textarea or send button.
