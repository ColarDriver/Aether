# PR24-4 - Sidebar Control Center and Session Density

## Objective

Bring the left side closer to Hermes WebUI's dense operational sidebar without
removing existing Aether routes.

## Implementation

- Keep all current Aether nav destinations.
- Add a bottom control-center area for:
  - appearance controls,
  - settings shortcut,
  - compact runtime/session metadata.
- Keep session search and new-session affordance near the session list.
- Preserve session title truncation and active-session highlighting.
- Do not add hover behaviors that reflow session rows on touch devices.

## Tests

- Update `Sidebar.test.tsx` to assert:
  - control center renders,
  - settings shortcut is available,
  - session filtering still works,
  - new session callback still fires.

## Acceptance

- Sidebar remains usable at desktop width.
- Session list scrolls independently of the page.
- Control center does not crowd out sessions on short screens.
