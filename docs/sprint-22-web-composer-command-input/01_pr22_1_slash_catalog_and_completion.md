# PR22.1 - Slash Catalog And Completion

## Scope

Expose the slash command catalog to the web app through REST and add pure
TypeScript helpers for browser-side completion.

## Backend Work

- Add a shared Python catalog helper rather than duplicating command metadata in
  web and gateway code.
- Keep `aether/cli/commands.py` as the source of command names/descriptions.
- Preserve the existing category semantics:
  - `session`: `/new`, `/session`, `/sessions`, `/resume`, `/system`,
    `/model`, `/plan`
  - `control`: `/interrupt`
  - `remote`: `/tools`
  - `local`: everything else
- Add `GET /api/commands` returning:

```json
{
  "commands": [
    { "name": "/help", "description": "Show this help table", "category": "local" }
  ]
}
```

- The route must not call the gateway dispatcher or require a running gateway.

## Frontend Work

- Add `SlashCommandInfo` / `CommandCatalog` API types.
- Add `api.commands()`.
- Add pure slash helper functions:
  - find an active slash token at the textarea cursor,
  - filter command catalog entries by token,
  - replace the token with the selected command,
  - preserve cursor position.

## Tests

- Python web route test proves `/api/commands` includes `/plan`.
- Gateway catalog tests continue to pass from the same source of truth.
- TS helper tests cover:
  - beginning-of-input slash,
  - whitespace-delimited slash,
  - rejecting file paths like `/workspace/Aether`,
  - filtering,
  - replacement with trailing space.
