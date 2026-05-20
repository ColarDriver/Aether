# PR 13.3 — Context Assembly Pipeline

## 目标

把 PRE_LLM 阶段的上下文装配逻辑从 `AgentEngine.run_loop` 中抽出，形成一个顺序明确、可测试的 `ContextAssemblyPipeline`。该 PR 不引入新的 context engine，不改变 memory / skill / diagnostics / plan mode 行为。

## 当前问题

PRE_LLM 前的 provider-bound messages 由很多步骤依次构造：

1. preflight compaction。
2. skill nudge。
3. pending task messages / `<task-notification>`。
4. diagnostics attachment。
5. verifier reminder。
6. plan mode attachment。
7. pre_llm hooks。
8. memory context merge。
9. middleware `before_llm`。
10. collapse projection view。

这些步骤顺序非常敏感。当前它们分散在 `run_loop` 中，后续任何改动都容易改变注入顺序，导致 plan mode、diagnostics 或 memory 出现隐性回归。

## 实现改动

### 新增 pipeline 模块

新增 `aether/agents/runtime/context_assembly.py`。

核心类型：

- `ContextAssemblyInput`
  - `request: EngineRequest`
  - `messages: list[dict[str, Any]]`
  - `context: TurnContext`
  - `iteration: int`

- `ContextAssemblyResult`
  - `canonical_messages: list[dict[str, Any]]`
  - `prepared_messages: list[dict[str, Any]]`
  - `hook_outcome: HookOutcome | None`
  - `preflight_compaction: CompactionResult | None`

- `ContextAssemblyPipeline`
  - constructed with `services`, `hooks`, and a small adapter exposing legacy helper methods.
  - method `assemble(input: ContextAssemblyInput) -> ContextAssemblyResult`

### Legacy helper adapter

为了避免一次性重写所有功能，PR 13.3 可先引入 internal adapter：

- `maybe_compact_messages(...)`
- `register_skill_nudge(...)`
- `maybe_inject_skill_nudge(...)`
- `drain_pending_messages(...)`
- `maybe_inject_diagnostic_attachment(...)`
- `maybe_inject_verifier_reminder(...)`
- `maybe_inject_plan_mode_attachment(...)`
- `collect_pre_llm_hook_outcome(...)`
- `merge_memory_context_into_hook_outcome(...)`
- `apply_hook_outcome_to_messages(...)`
- `apply_collapse_view(...)`

第一版可以让 adapter 调用 `AgentEngine` 的现有私有方法；后续 PR 再逐步下沉实现。这样可以让拆分风险可控。

### 必须保持的顺序

Pipeline 必须保持以下顺序：

```text
preflight compaction
→ skill nudge
→ pending messages / task notifications
→ diagnostics attachment
→ verifier reminder
→ plan mode attachment
→ pre_llm hook
→ memory context merge
→ hook outcome application
→ middleware before_llm
→ collapse projection view
```

注意：

- collapse view 只影响 `prepared_messages`，不能污染 `canonical_messages`。
- memory context 仍通过 hook outcome / provider-bound copy 注入，不能写回 persisted transcript。
- plan mode attachment 必须仍邻近 latest user turn。
- diagnostics attachment 必须仍在下一轮 PRE_LLM 注入，而不是 edit 同轮注入。

### AgentEngine 迁移

`run_loop` 中 PRE_LLM 片段改为：

- 调用 `context_pipeline.assemble(...)`。
- 用 `result.canonical_messages` 更新本轮 canonical messages。
- 用 `result.prepared_messages` 进入 provider invocation。
- 保留 state machine transition 和 error handling。

## 测试

新增：

- `aether/tests/agents/runtime/test_context_assembly_pipeline.py`

覆盖：

- 使用 fake adapter 记录调用顺序，断言顺序完全匹配。
- plan mode 下 attachment 仍出现。
- diagnostics pending 时，prepared messages 末尾包含 `<diagnostics>`。
- memory injection 不改变 canonical messages。
- collapse view 不改变 canonical messages，只改变 prepared messages。
- middleware `before_llm` 抛错时仍转为 `MIDDLEWARE_ERROR`。

回归：

- plan mode prompt tests。
- diagnostics attachment tests。
- skill nudge tests。
- compaction projection tests。

## 验收

- `run_loop` 中 PRE_LLM 上下文装配代码显著减少。
- 上下文装配顺序有独立单测保护。
- 不改变 provider 请求最终内容，除非测试暴露当前行为不稳定。
- 不改变任何用户可见 TUI 文案。
