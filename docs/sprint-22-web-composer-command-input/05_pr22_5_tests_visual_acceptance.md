# PR22.5 - Tests And Visual Acceptance

## Automated Tests

- Python:
  - `/api/commands`
  - command catalog category parity
  - run start attachment persistence
- TypeScript:
  - slash helper functions
  - slash popover keyboard behavior
  - composer send vs completion behavior
  - local slash execution
  - attachment preview and send payload

## Manual Acceptance

1. Start the web server and open the browser console.
2. Select or create a session.
3. Type `/` and confirm commands appear.
4. Type `/pl`, select `/plan`, and confirm the token is replaced.
5. Confirm ArrowUp/ArrowDown, Tab, Enter, and Escape behavior.
6. Submit a normal prompt and confirm it still starts a run.
7. Submit `/help` after PR22.3 and confirm no agent run starts.
8. Add an image/file after PR22.4 and confirm it previews before send.
9. Resume the session and confirm attachments render in the timeline.

## Regression Checks

- Normal Enter/Shift+Enter behavior is unchanged.
- Send/stop buttons still work.
- Timeline auto-scroll is not affected by opening the popover.
- Prompt approval modals still retain focus behavior.
- Mobile/narrow composer width does not overflow.

## Current Implementation Evidence

Implemented on branch `web-console-migration`:

- PR22.1 shared catalog source:
  - `aether/cli/commands.py`
  - `aether/gateway/handlers/commands_methods.py`
- PR22.1 web REST endpoint:
  - `aether/web/routes/commands.py`
  - `aether/web/app.py`
  - `web/src/api/client.ts`
  - `web/src/api/types.ts`
- PR22.1 TypeScript slash helpers:
  - `web/src/components/chat/slashCompletion.ts`
- PR22.2 popover and composer integration:
  - `web/src/components/chat/SlashPopover.tsx`
  - `web/src/components/chat/Composer.tsx`
  - `web/src/styles.css`

Latest verification performed during implementation:

- `cd web && npm test` (36 files / 86 tests)
- `cd web && npm run build`
- `python -m pytest aether/tests/web/test_web_rest_services.py aether/tests/gateway/test_commands_methods.py`
- `uv run pyright aether/web aether/gateway/handlers/commands_methods.py`

Remaining before Sprint 22 completion:

- PR22.3 slash execution and timeline notices.
- PR22.4 composer attachments, file references, paste, and drag/drop.
- Full manual browser acceptance after PR22.3 and PR22.4.
