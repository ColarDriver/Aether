# PR 18.5 - Tests, Observability, and Acceptance

## Goal

Close Sprint 18 with deterministic regression coverage.

## Verification

Run:

- `python -m pytest aether/tests/runtime/tools`
- `python -m pytest aether/tests/agents/runtime/test_tool_dispatch_controller.py`
- `python -m pytest aether/tests/engine/test_tool_permission_gate.py`
- `python -m pytest aether/tests/engine/test_interrupt_message_injection.py`
- `python -m pytest aether/tests/tools`
- targeted `uv run pyright` for tool runtime and dispatch files

## Manual Checks

- Ask the model to read multiple files in one turn.
- Confirm read-only tools run as one parallel batch.
- Ask for one read and one file edit.
- Confirm sequential fallback.
- Trigger interrupt during a slow web fetch/search.
- Confirm prompt returns cleanly.

## Acceptance

- Parallel execution improves safe read-only batches.
- Mutating and permission tools remain conservative.
- Transcript order is stable.
- No plan-mode blocker regression.

## Detailed Acceptance Procedure

### Static Review

Confirm:

- Unknown tools default to sequential.
- Shell remains sequential.
- Permission-required tools are sequential.
- Plan-mode blocker runs before scheduling.
- Scheduler does not append transcript messages directly.
- ToolDispatchController remains the only owner of tool result message shape.

### Automated Tests

Focused:

```bash
python -m pytest aether/tests/runtime/tools/test_capabilities.py
python -m pytest aether/tests/runtime/tools/test_parallel_scheduler.py
python -m pytest aether/tests/agents/runtime/test_tool_dispatch_controller.py
python -m pytest aether/tests/engine/test_tool_permission_gate.py
python -m pytest aether/tests/engine/test_interrupt_message_injection.py
```

Broader:

```bash
python -m pytest aether/tests/tools
python -m pytest aether/tests/engine
python -m pytest aether/tests/agents
```

Targeted type check:

```bash
uv run pyright aether/runtime/tools aether/agents/runtime/tool_dispatch.py aether/tools
```

### Manual Acceptance Script

1. Start Aether normally.
2. Ask for three independent file reads.
3. Confirm metadata/logs show one parallel batch.
4. Ask for one read and one edit.
5. Confirm sequential fallback.
6. Enter plan mode and ask for a write.
7. Confirm plan blocker triggers before scheduling.
8. Trigger interrupt during a slow fetch/search batch.
9. Confirm the turn returns cleanly and transcript order remains coherent.

### Observability Requirements

Expose enough metadata to answer:

- Was parallel scheduling enabled?
- Did this batch run in parallel?
- Why did it fall back?
- How many calls were in the batch?
- How long did the batch take?
- Did any worker error?

Do not expose:

- thread objects
- full tool output duplicates
- raw exceptions with sensitive local paths beyond existing behavior

### Default-On Decision

At the end of PR18.5, decide whether to set
`parallel_tool_execution_enabled=True` by default. Criteria:

- full tests pass
- manual checks pass
- no permission regressions
- no interrupt hanging
- no known thread-safety issue with any tool marked parallel safe

If any criterion fails, keep default off and document the blocker.
