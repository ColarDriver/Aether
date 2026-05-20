# PR 18.3 - Permission, Interrupt, and Cancellation

## Goal

Define how parallel execution interacts with permission prompts and interrupts.

## Permission Rules

First version:

- If any tool in the batch needs an interactive permission prompt, run the batch
  sequentially.
- Do not show multiple permission prompts concurrently.
- Plan mode blocker still runs before permission.

Future optional version:

- Group permission previews into one modal only after TUI support exists.

## Interrupt Rules

- If interrupt is already set before dispatch, do not start new workers.
- If interrupt fires during parallel execution, ask running tools to stop through
  the existing interrupt signal.
- Collect completed results.
- Mark interrupted/incomplete tools with structured error results.

## Cancellation Rules

- The scheduler must shut down worker threads promptly.
- No background orphan worker should keep writing after a turn is finalized.

## Tests

Add/update:

- `aether/tests/runtime/tools/test_parallel_scheduler_interrupt.py`
- `aether/tests/engine/test_interrupt_message_injection.py`

Cover:

- Interrupt before scheduling.
- Interrupt during running tools.
- Permission-required batch falls back to sequential.
- Plan-mode blocked write does not enter worker pool.

## Acceptance

- No permission modal regression.
- Interrupt behavior is deterministic.

## Detailed Implementation Notes

### Permission Sequencing

The safe first version is:

1. prepare/repair/dedup tool calls
2. apply plan-mode blocker per call
3. if any call would require permission prompt, run whole batch sequentially
4. if no prompts are needed and scheduler plan is parallel, execute parallel

This avoids concurrent prompts and keeps preview generation unchanged.

### Permission Detection

Do not duplicate permission policy in scheduler. Add a method on the dispatch
adapter if needed:

- `would_require_permission(tool_call, context) -> bool`

If current permission gate can only decide by prompting, do not add parallelism
for permission-requiring tools yet.

### Interrupt Before Start

If `context.interrupt_signal.is_aborted()` is true before scheduling:

- do not start workers
- return interrupted dispatch result
- append no partial successful results unless existing sequential behavior would
  have done so

### Interrupt During Execution

Each worker uses the same `TurnContext` and interrupt signal. This means tools
that already observe `context.interrupt_signal` can stop themselves.

For tools that do not observe interrupts:

- wait for bounded completion if already running
- mark incomplete results only if future cancellation support can prove they
  did not finish
- do not kill threads unsafely

### Cleanup

Always shut down the executor. If using `ThreadPoolExecutor`, use a context
manager or explicit `shutdown(wait=True)` unless tests prove non-waiting
shutdown is required for responsiveness.

### Tests

Add fake tools:

- slow interrupt-aware tool
- slow interrupt-unaware tool
- permission-required fake tool

Cover:

- interrupted before start
- interrupted after one tool completes
- permission batch sequential fallback
- plan mode blocked write never starts a worker
- executor shutdown runs after errors

### Review Checklist

- No parallel permission prompts.
- No orphan executor.
- Interrupt metadata matches existing dispatch behavior.
