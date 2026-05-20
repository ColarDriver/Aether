# PR 15.5 - Facade Cleanup and Acceptance

## Goal

Remove dead wrappers left by PR15.1-15.4 and document the new engine shape.

## Cleanup Rules

- Remove private `AgentEngine` methods only when no tests or adapters call them.
- Keep public methods stable.
- Do not collapse new controllers back into the runner.
- Do not mix unrelated style cleanup.

## Required Documentation

Update:

- `docs/sprint-15-turn-runner-response-stream/99_acceptance_matrix.md`
- Any agent engine architecture docs that mention `run_loop()` internals.

## Verification

Run:

- `python -m pytest aether/tests/agents aether/tests/engine aether/tests/runtime`
- `python -m pytest aether/tests/gateway`
- targeted `uv run pyright` for `aether/agents/core/agent.py` and
  `aether/agents/runtime/`

## Acceptance

- `AgentEngine` is visibly smaller.
- `run_loop()` is a facade into `TurnRunner`.
- Response finalization and stream behavior have focused tests.
- No TUI/gateway wire schema changes.

## Detailed Cleanup Plan

### Remove Temporary Adapter Methods

Review `LegacyTurnRunnerAdapter` after PR15.2-PR15.4. Any method that simply
calls a controller should be deleted from the adapter and replaced with direct
runner dependency injection.

Expected adapter removals:

- length continuation helpers after PR15.3
- invalid JSON/truncated tool helpers after PR15.3
- stream callback helpers after PR15.4
- empty/phantom finalization helpers after PR15.2

Keep adapter methods only for genuinely engine-owned concerns that are planned
for a later sprint.

### Line Count Target

Record line counts before and after:

```bash
wc -l aether/agents/core/agent.py aether/agents/runtime/*.py
```

The target is not an arbitrary minimum. The useful signal is that
`AgentEngine.run_loop()` is short and new behavior no longer lands in
`agent.py`.

### Public API Audit

Verify these remain stable:

- `AgentEngine.__init__`
- `run_loop`
- `run_turn`
- `resume`
- `interrupt`
- `clear_interrupt`
- `send_steer`
- `run_subagents`
- `delegate_depth`
- `subagent_id`
- `parent_subagent_id`
- `subagent_manager`

### Regression Suite

Run:

```bash
python -m pytest aether/tests/agents
python -m pytest aether/tests/engine
python -m pytest aether/tests/runtime
python -m pytest aether/tests/gateway
python -m pytest aether/tests/subagents
```

Then targeted type check:

```bash
uv run pyright aether/agents/core/agent.py aether/agents/runtime
```

### Documentation Update

Update the sprint acceptance matrix with:

- old/new `AgentEngine` line count
- list of controllers now responsible for loop behavior
- any deferred private methods still in `agent.py`
- reason each deferred method stayed

### Review Checklist

- New controllers are not circularly importing `AgentEngine`.
- New tests cover extracted behavior directly, not only through full engine
  tests.
- No unrelated TUI/scroll/markdown code changed.
