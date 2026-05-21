# PR24-1 - Reference Analysis and Shell Tokens

## Objective

Translate Hermes WebUI's interface principles into Aether-owned React/CSS tokens
without copying its vanilla implementation.

## Reference Findings

Hermes WebUI's durable ideas:

- `body` is a fixed-height workbench with internal scroll containers.
- `.layout` owns the horizontal panels.
- `.messages` owns chat scrolling, not the entire page.
- `.composer-wrap` is fixed at the bottom of the chat column.
- `.rightpanel` is a workspace rail that can collapse independently.
- token/context usage lives in the composer footer as a compact circular
  affordance.
- session list and control center live in the left sidebar.

Aether already has the first scroll principle in `ChatView`; preserve it.

## Implementation

- Add semantic shell variables to `web/src/styles.css`:
  - `--surface`
  - `--sidebar-bg`
  - `--hover-bg`
  - `--focus-ring`
  - `--chat-column-max`
- Keep existing `--bg`, `--panel`, `--panel-2`, `--line`, `--text`, `--muted`,
  and semantic colors as the compatibility layer for current components.
- Update shell and chat CSS to use the semantic tokens where appropriate.
- Do not introduce one-off component colors when a token can carry the intent.

## Tests

- Existing component tests should continue to pass.
- `git diff --check` must be clean.

## Acceptance

- Token additions do not change API or store contracts.
- Dark and terminal themes still render with readable text and borders.
- The chat scroll container remains `.chat-scroll`, not `body`.
