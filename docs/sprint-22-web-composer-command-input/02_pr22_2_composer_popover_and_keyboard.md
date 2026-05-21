# PR22.2 - Composer Popover And Keyboard

## Scope

Render slash completions above the composer and wire keyboard handling into the
textarea without breaking normal send behavior.

## Component Work

- Add `SlashPopover.tsx` under `web/src/components/chat`.
- The popover consumes:
  - `commands`,
  - `value`,
  - `cursorPosition`,
  - `onApply(nextValue, nextCursorPosition)`.
- It renders a `listbox` of command options with:
  - command name,
  - description,
  - category badge.
- It owns selection state and exposes `handleKey(event)`.

## Keyboard Behavior

- ArrowDown/ArrowUp moves selected command.
- Tab applies the selected command.
- Enter applies the selected command while the popover is open.
- Escape closes the popover.
- Enter sends the message only when the popover did not consume the key.
- Shift+Enter still inserts a newline.

## Composer Integration

- Load command catalog through `api.commands()`.
- Show the popover only for valid slash tokens.
- Restore textarea focus and selection after applying a command.
- Disable completion when the composer is disabled.

## Tests

- Component test for rendering and click apply.
- Keyboard test for ArrowDown + Enter/Tab.
- Composer test proving slash selection edits the textarea instead of sending.
- Composer test proving normal Enter still sends.
