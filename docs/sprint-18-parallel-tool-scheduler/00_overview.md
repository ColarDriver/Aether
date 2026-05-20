# Sprint 18 - Parallel Tool Scheduler

## Decision

This must be its own sprint.

Parallel tool execution changes tool ordering, permission prompts, interrupts,
path mutation safety, and transcript message order. It should not be bundled
with provider transport or runner extraction.

## Motivation

Aether currently has solid tool dispatch policy but executes ordinary tool
calls conservatively. Hermes has a safety classifier that parallelizes read-only
and independent path-scoped calls while keeping interactive or mutating calls
sequential.

Aether should add that capability only after tool dispatch is already isolated.

## Goals

- Add tool capability metadata for parallel safety.
- Add a scheduler that can run safe tool batches concurrently.
- Preserve transcript order.
- Preserve permission semantics.
- Preserve interrupt behavior.
- Keep mutating tools sequential unless explicitly proven independent.

## Non-Goals

- Do not parallelize shell commands in the first version.
- Do not parallelize interactive tools.
- Do not bypass permission prompts.
- Do not change tool descriptor wire schema unless a backward-compatible
  metadata field is added.
- Do not change subagent async lifecycle.

## PR Roadmap

| PR | File | Boundary |
|---|---|---|
| 18.1 | `01_pr18_1_tool_capability_metadata.md` | Tool safety metadata and classifier |
| 18.2 | `02_pr18_2_parallel_scheduler_core.md` | Scheduler with ordering and path overlap |
| 18.3 | `03_pr18_3_permission_interrupt_and_cancellation.md` | Permission batching, interrupt, cancellation |
| 18.4 | `04_pr18_4_tool_dispatch_integration.md` | Integrate scheduler into ToolDispatchController |
| 18.5 | `05_pr18_5_tests_observability_and_acceptance.md` | Tests, metrics, regression |

## Completion Criteria

- Safe read-only tool batches can run concurrently.
- Results are appended in original tool-call order.
- Unsafe batches fall back to sequential execution.
- Existing permission and plan-mode behavior remains stable.

## Current Aether Anchors

Review:

- `aether/agents/runtime/tool_dispatch.py`
- `aether/tools/registry.py`
- `aether/tools/base.py`
- `aether/runtime/tools/tool_permissions.py`
- `aether/config/schema.py`
- built-in tools under `aether/tools/builtins/`

Tool dispatch already owns schema repair, plan-mode blocking, permission gates,
dedup/delegate cap, post hooks, diagnostics, cheap-tool refund metadata, and
tool result message appending. Sprint 18 should add scheduling under that
controller, not bypass it.

## Hermes Reference Points

Read:

- `/workspace/hermes-agent/agent/tool_dispatch_helpers.py`
- `/workspace/hermes-agent/agent/tool_executor.py`

Important ideas to borrow:

- parallelize only safe batches
- path-scoped tools require overlap checks
- interactive tools force sequential execution
- result order must be stable even when execution order is not

Do not copy Hermes' exact tool names; Aether has its own tool registry and plan
mode policy.

## Safety Philosophy

Default unsafe. A tool runs in parallel only if Aether can prove it is safe.
This means:

- unknown external tools are sequential
- shell is sequential
- permission-required tools are sequential in first version
- mutating tools are sequential unless a future PR proves independent path
  isolation
- read-only file/search/list/fetch tools may be parallel

Performance is secondary to correctness. The first version should be easy to
reason about.

## Transcript Invariant

The model must see tool results in the same order as the original assistant
`tool_calls`. Even if workers finish out of order, the final messages appended
to `messages` must be ordered by original call index.

## Permission Invariant

Parallel scheduling must never cause multiple simultaneous permission prompts.
Plan-mode blockers and permission gates run before execution decisions. If any
call requires interactive approval, fallback to sequential dispatch for the
batch.
