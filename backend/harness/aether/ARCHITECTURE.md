# Aether Harness Architecture (v1)

This document captures the first modular split of the harness and the migration sequence.

## Module Map

- `agents/core/agent.py`
  - Main `AgentEngine` orchestrator.
  - Exposes `run_loop()` as the primary execution method.
  - Owns loop transitions and invokes provider/tool/middleware services.
- `runtime/state_machine.py`
  - Explicit finite state machine and transition validation.
- `runtime/contracts.py`
  - Shared DTOs: request/result/context/tool calls/responses.
- `runtime/interrupts.py`
  - Session-scoped interrupt controller.
- `runtime/services.py`
  - Dependency container for engine collaborators.
- `models/provider/base.py`
  - Provider abstraction (`ModelProvider.generate`).
- `tools/base.py`, `tools/registry.py`
  - Tool contract and dispatch registry.
- `agents/middlewares/base.py`, `agents/middlewares/pipeline.py`
  - Synchronous middleware hooks and chain execution.
- `agents/memory/base.py`, `skills/base.py`, `agents/entry/subagent_base.py`
  - Placeholder extension interfaces for future modules.
- `subagents/manager.py`, `subagents/default_builder.py`
  - Parent-child delegation orchestration with depth/concurrency limits.

## Loop States

`INIT -> PREPARE -> PRE_LLM -> LLM_CALL -> POST_LLM -> TOOL_DISPATCH -> TOOL_EXECUTE -> CHECK_EXIT -> FINALIZE -> DONE/FAILED/INTERRUPTED`

This is the only allowed orchestration path in v1.

## High-Risk Coupling Points

1. **State logic vs side effects**
   - Risk: provider/tool calls leak into transition policy.
   - Rule: state machine only validates transitions; engine handlers perform side effects.

2. **Provider response shape vs engine contracts**
   - Risk: provider-specific fields pollute engine internals.
   - Rule: providers return `NormalizedResponse` and `ToolCall` only.

3. **Tool dispatch vs interrupt semantics**
   - Risk: long tool execution ignores interrupts.
   - Rule: interrupt checked at every loop boundary and before each tool dispatch.

4. **Middleware hooks vs loop determinism**
   - Risk: unbounded middleware behavior causes hidden transitions.
   - Rule: middleware can mutate payloads but cannot mutate loop state directly.

5. **Future subagent delegation vs parent loop ownership**
   - Risk: embedding child orchestration in `AgentEngine` makes replacement hard.
   - Rule: keep delegation behind `SubagentDispatcher` interface.

## Next Split Sequence

1. Add context manager module (prepare-stage integration only).
2. Add memory provider implementation behind `MemoryProvider`.
3. Add skills provider implementation behind `SkillProvider`.
4. Add subagent dispatcher implementation behind `SubagentDispatcher`.
5. Add streaming provider path and retry/fallback policy module.
6. Add richer error taxonomy and recovery planner module.

## Test Coverage in v1

- State machine transition validation tests.
- Engine integration tests:
  - text response completion,
  - tool round trip,
  - pre-turn interrupt,
  - provider failure path.
