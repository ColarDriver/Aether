# PR 13.6 — Session Lifecycle and Engine Facade

## 目标

把 session lifecycle 和 turn lifecycle setup / finalize 从 `AgentEngine` 中抽出，让 `AgentEngine` 成为稳定 public facade 和 orchestrator。该 PR 是 Sprint 13 的收敛 PR。

## 当前问题

`AgentEngine` 当前同时负责：

- 构造默认 tool registry。
- 构造默认 memory provider / project memory store。
- 构造 task store 并 recovery orphaned tasks。
- session cwd seeding。
- `_prepare_turn_entry`。
- system prompt selection / persistence。
- session start hook。
- turn metadata initialization。
- resource cleanup。
- interrupt clear / child interrupt。
- result building。
- subagent root registration。

这些逻辑混在 constructor 和 run loop 中，导致 public facade 与 internal lifecycle 无法区分。

## 实现改动

### 新增 lifecycle 模块

新增 `aether/agents/runtime/session_lifecycle.py`。

核心类型：

- `TurnPreparationResult`
  - `state_machine: EngineStateMachine`
  - `messages: list[dict[str, Any]]`
  - `context: TurnContext`
  - `active_system_prompt: str | None`

- `TurnFinalizationInput`
  - `request: EngineRequest`
  - `context: TurnContext | None`
  - `final_response: str | None`
  - `error_text: str | None`
  - `exit_reason: ExitReason`
  - `iterations: int`
  - `budget: IterationBudget | None`

- `SessionLifecycleController`
  - `prepare_turn(...)`
  - `prepare_session_and_system_prompt(...)`
  - `finalize_turn(...)`
  - `cleanup_task_resources(...)`
  - `emit_session_hooks(...)`

### 抽出职责

Lifecycle controller 负责：

- 从 `EngineRequest` 构造 initial messages / `TurnContext`。
- 初始化 metadata 和 runtime refs。
- session store 读取 / 写入 system prompt。
- system prompt augmentation 调用，但不改变 augmentation 内容。
- skill listing system prompt 拼接。
- plan mode active metadata 标记。
- cwd seeding。
- session start / end hooks。
- task resource cleanup。
- result building 入口调用。

### AgentEngine facade 保留

`AgentEngine` public API 保持：

- `__init__(provider, *, tool_registry=None, middleware_pipeline=None, config=None, ...)`
- `run_loop(request)`
- `run_turn(request)`
- `resume(request)`
- `interrupt(session_id=None, reason=None)`
- `clear_interrupt(session_id=None)`
- `send_steer(session_id, text)`
- `run_subagents(...)`
- `delegate_depth` / `subagent_id` / `parent_subagent_id` / `subagent_manager` properties

### Constructor 收敛

不改变 constructor signature，但内部可以拆成私有 setup helpers：

- `_build_default_tool_registry(...)`
- `_build_memory_provider(...)`
- `_build_task_store(...)`
- `_build_runtime_controllers(...)`

避免 constructor 继续膨胀。

### 行数目标

PR 完成后：

- `aether/agents/core/agent.py` 行数目标减少至少 25%。
- 不能通过删除注释或压缩格式来达标；必须是职责移动。
- 如果为了兼容保留 wrapper，允许短期没有达到 25%，但 PR 文档和 follow-up issue 必须写明剩余 blockers。

## 测试

新增：

- `aether/tests/agents/runtime/test_session_lifecycle.py`
- `aether/tests/agents/test_agent_engine_facade_compat.py`

覆盖：

- new session 触发 `on_session_start`。
- stored system prompt 被复用。
- request system prompt 覆盖 stored prompt 并持久化。
- plan mode session 标记 `plan_mode_active`。
- cwd seeding 行为不变。
- task resource cleanup 在 completed / interrupted / failed 时都执行。
- `run_turn` / `resume` 仍调用 `run_loop` 等价路径。
- interrupt active children 行为不变。

回归：

- gateway `agent.run` tests。
- session resume / clear tests。
- subagent tests。
- diagnostics tests。
- plan mode tests。

## 验收

- `AgentEngine` 已明显收敛为 orchestrator。
- public API 兼容。
- session lifecycle 有独立测试。
- 新 controller 不从全局重新构造 provider / registry / hooks。
- `AgentEngine` 不再是新增 runtime 能力的默认堆放点。
