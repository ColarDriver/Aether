# Sprint 13 — AgentEngine Decomposition (Overview)

## 背景 / Motivation

Aether 的 Agent runtime 已经具备相当多的能力：state machine、iteration budget、middleware、hooks、permission、memory、compaction、provider fallback、plan mode、diagnostics、subagent task store、interrupt / steer 等都已经接入主循环。

当前主要问题不是缺少某个单点能力，而是这些能力高度集中在 `aether/agents/core/agent.py` 的 `AgentEngine` 中。该文件已经超过 6600 行，`AgentEngine.__init__` 注入大量依赖，`run_loop` 同时承担 turn setup、context assembly、provider invocation、recovery、tool dispatch、session lifecycle、resource cleanup 和 result building。每个新增 runtime 能力都会继续扩大这个类，使后续迭代越来越难安全修改。

Sprint 13 的目标是进行 **refactor-only decomposition**：把当前 `AgentEngine` 内部已经存在的职责拆到明确的小模块中，保持外部行为和 wire schema 基本不变，为后续 Sprint 14+ 引入 provider transport、context curator、credential runtime 等更大架构打底。

## 当前状态 / Current State

Aether 已有能力：

- `AgentEngine` 使用显式 `LoopState` / `ExitReason` / `EngineResult` contract。
- `EngineServices` 作为轻量 DI 容器持有 provider、tool registry、middleware、interrupt、recovery、fallback、steer、memory。
- `TurnContext.metadata` 承载 per-turn 状态，包括 retry counters、usage、memory、compaction、permission、diagnostics、plan mode、resource cleanup。
- `SessionRuntimeRegistry` 承载 per-session runtime state，包括 memory / skill nudge、permission rules、task memory snapshot。
- Provider 层已有 `ModelProvider.generate()`、`validate_response()`、`list_models()`、`set_model()` 等基础 contract。
- Tool 层已有 registry dispatch、permission gate、plan-mode blocker、dedup、delegate cap、post-tool diagnostics 等能力。
- Context 层已有五层 compaction pipeline、memory injection、skill listing / nudge、diagnostics attachment、verifier reminder、plan-mode reminder、collapse projection view。
- Subagent 层已有 sync / async task、TaskStore、send_message / task_output / notification 基础。

结构性问题：

- `AgentEngine` 同时是 orchestrator、provider invoker、context assembler、tool dispatcher、recovery coordinator、session lifecycle manager。
- `TurnContext.metadata` 的 internal keys 和 public metadata keys 分散在多个模块，新增 key 容易泄露到 `EngineResult.metadata["turn"]`。
- Provider 调用相关逻辑和 recovery 逻辑耦合紧密，后续引入 transport 层风险较高。
- PRE_LLM 上下文注入顺序很关键，但目前顺序隐藏在长 `run_loop` 中，回归风险高。
- Tool dispatch 包含 permission、plan-mode blocker、dedup、delegate cap、middleware、hooks、metadata normalization，职责过重。
- Session lifecycle 和 turn lifecycle 没有独立边界，`/resume`、plan mode、interrupt、resource cleanup、system prompt persistence 都依赖主类细节。

## Sprint Goals

1. 将 `AgentEngine` 收敛为 facade / orchestrator，而不是所有 runtime 行为的承载者。
2. 保持 public API 兼容：`run_loop`、`run_turn`、`resume`、`interrupt`、`send_steer`、subagent helper 不变。
3. 保持 provider contract 不变：不在本 sprint 改 `ModelProvider.generate()`。
4. 保持 tool contract 不变：不改 `ToolRegistry.dispatch`、tool descriptor、permission wire schema。
5. 保持 TUI / gateway wire schema 不变；除非测试暴露 bug，否则不碰 TUI rendering。
6. 保持现有 `EngineResult.metadata` 公开字段不变。
7. 将未来可替换边界显式化：provider invocation、context assembly、tool dispatch、recovery、session lifecycle。
8. 通过测试确认普通 agent mode、plan mode、tool permission、diagnostics、subagent、compaction / recovery 都无行为回归。

## Non-Goals

以下能力明确不进入 Sprint 13：

- 不实现 Hermes 风格完整 `ProviderTransport`。
- 不实现 pluggable `ContextEngine` / `ContextCurator`。
- 不实现 credential pool、多账号轮换、token refresh runtime。
- 不重做 subagent scheduler，不引入跨进程 durable queue。
- 不修改 TUI scroll / shimmer / markdown rendering。
- 不修改 plan mode 产品语义。
- 不修改 provider payload 或 web search provider 行为。
- 不做大规模命名清理、格式化、无关文件重排。

这些能力可以在 Sprint 14+ 基于 Sprint 13 的边界继续推进。

## Roadmap

| # | 文档 | 内容 |
|---|---|---|
| 1 | [`01_pr13_1_runtime_context_and_metadata_contract.md`](./01_pr13_1_runtime_context_and_metadata_contract.md) | 集中 runtime metadata key、internal snapshot、turn metadata helper |
| 2 | [`02_pr13_2_provider_invocation_controller.md`](./02_pr13_2_provider_invocation_controller.md) | 抽出 provider 调用、stream wrapper、API hooks、usage / validation |
| 3 | [`03_pr13_3_context_assembly_pipeline.md`](./03_pr13_3_context_assembly_pipeline.md) | 抽出 PRE_LLM 上下文装配顺序 |
| 4 | [`04_pr13_4_tool_dispatch_controller.md`](./04_pr13_4_tool_dispatch_controller.md) | 抽出 tool dispatch、permission、plan blocker、hooks |
| 5 | [`05_pr13_5_recovery_and_compaction_boundaries.md`](./05_pr13_5_recovery_and_compaction_boundaries.md) | 抽出 recovery facade 与 compaction 边界 |
| 6 | [`06_pr13_6_session_lifecycle_and_engine_facade.md`](./06_pr13_6_session_lifecycle_and_engine_facade.md) | 抽出 session lifecycle，收敛 AgentEngine facade |
| 7 | [`07_pr13_7_tests_observability_and_acceptance.md`](./07_pr13_7_tests_observability_and_acceptance.md) | 总体验收、观测字段、回归清单 |
| 8 | [`99_acceptance_matrix.md`](./99_acceptance_matrix.md) | 端到端行为矩阵 |

## Dependency Graph

```text
PR 13.1 (metadata contract)
  ├─→ PR 13.2 (provider invocation)
  ├─→ PR 13.3 (context assembly)
  └─→ PR 13.4 (tool dispatch)

PR 13.2 + PR 13.3
  └─→ PR 13.5 (recovery + compaction boundaries)

PR 13.1 + 13.2 + 13.3 + 13.4 + 13.5
  └─→ PR 13.6 (session lifecycle + facade)

PR 13.7 validates the full sprint.
```

推荐合入顺序：

1. PR 13.1 先落地 metadata contract，降低后续 extraction 的泄露风险。
2. PR 13.2 / 13.3 / 13.4 分别拆 provider、context、tool 三条主线。
3. PR 13.5 整理 recovery 与 compaction 边界。
4. PR 13.6 收敛 `AgentEngine` public facade 和 session lifecycle。
5. PR 13.7 做回归矩阵和 acceptance。

## Acceptance Summary

- `AgentEngine.run_loop()` 外部行为不变。
- `EngineResult.status`、`final_response`、`error`、`exit_reason`、`metadata` 公开字段不变。
- `ModelProvider.generate()` contract 不变。
- Tool registry、permission prompt、approval prompt、plan mode blocker 行为不变。
- PRE_LLM 注入顺序不变：compaction → skill / pending / diagnostics / verifier / plan → hooks / memory → middleware → collapse view → provider。
- Provider API hooks 调用顺序不变。
- Tool post hooks 调用顺序不变。
- Subagent sync / async task 行为不变。
- Existing Python tests 全绿；Sprint 13 新增 controller-level tests 覆盖拆分边界。

## Migration Principle

每个 PR 都必须遵守：

- 先抽 helper / controller，再移动调用点。
- 一次只移动一个职责边界，避免 mixed refactor。
- 不做行为修正，除非测试暴露当前行为与文档 contract 矛盾。
- 旧私有方法可以临时保留为 delegating wrapper，下一 PR 再清理。
- 所有新模块必须从 `AgentEngine` 传入依赖，不从全局重新构造 provider / registry / hooks。
