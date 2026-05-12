# Aether Test Layout

Tests are grouped by the subsystem they primarily exercise:

- `cli/` - command parsing, UI, preferences, and tool group presentation.
- `engine/` - agent loop behavior, streaming, phantom tool recovery, and max-iteration handling.
- `models/` - provider adapters and provider error surfaces.
- `runtime/` - shared runtime primitives, recovery, budgeting, state, and result storage.
- `skills/` - skill catalog discovery and skill tool behavior.
- `subagents/` - subagent manager behavior.
- `tools/` - built-in tool executors and tool-adjacent managers.

Run all tests from the repository root with:

```bash
uv run pytest aether/tests -q
```
