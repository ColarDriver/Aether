# PR 13.7 — Tests, Observability, and Acceptance

## 目标

完成 Sprint 13 的总体验收，补齐 controller-level tests、integration regression tests、observability consistency checks，确认 AgentEngine decomposition 没有改变用户可见行为。

## 测试范围

### Python unit tests

新增或扩展：

- `aether/tests/runtime/core/test_turn_metadata.py`
- `aether/tests/agents/runtime/test_provider_invocation_controller.py`
- `aether/tests/agents/runtime/test_context_assembly_pipeline.py`
- `aether/tests/agents/runtime/test_tool_dispatch_controller.py`
- `aether/tests/agents/runtime/test_recovery_controller.py`
- `aether/tests/agents/runtime/test_session_lifecycle.py`
- `aether/tests/agents/test_agent_engine_facade_compat.py`

每类 controller tests 都必须使用 fake provider / fake registry / fake hooks / fake adapter 做窄单测，避免只能靠 full engine integration 发现顺序错误。

### Integration regression tests

至少覆盖：

- 普通文本响应。
- read-only tool call。
- dangerous tool permission approve / reject。
- plan mode blocker。
- `exit_plan_mode` approval。
- provider invalid response recovery。
- empty response recovery。
- length continuation。
- context overflow compaction。
- subagent sync task。
- subagent async task notification。
- diagnostics after edit。
- verifier reminder。
- memory injection。
- interrupt during LLM call。
- interrupt during tool call。

### Type checks

运行：

```bash
python -m pytest aether/tests
uv run pyright
```

如本 sprint 没有修改 TUI / gateway wire schema，不要求每个 PR 都跑 TS tests。若任何 PR 修改 gateway protocol 或 TUI-facing event，则追加：

```bash
npm --prefix tui run type-check
npm --prefix tui test
```

## Observability Requirements

Sprint 13 是 refactor-only，因此观测字段要保持稳定：

- `EngineResult.metadata["usage"]` 不变。
- `EngineResult.metadata["api_calls"]` 不变。
- `EngineResult.metadata["memory"]` 不变。
- `EngineResult.metadata["compaction"]` 不变。
- `EngineResult.metadata["resource_cleanup"]` 不变。
- `EngineResult.metadata["iteration_budget"]` 不变。
- `EngineResult.metadata["exit"]` 不变。
- provider recovery decision trail 字段不变。
- tool result metadata 中 permission、edited_paths、plan_mode_blocked 等字段不变。

允许新增 debug log，但不能改变 gateway event schema。新增 log 应使用明确 namespace：

- `runtime.provider_invocation.*`
- `runtime.context_assembly.*`
- `runtime.tool_dispatch.*`
- `runtime.recovery.*`
- `runtime.session_lifecycle.*`

## Manual Acceptance

手工验收脚本：

```bash
cd /workspace/Aether
uv run aether
```

场景：

1. 输入普通问题，确认能流式回复。
2. 让模型 `read_file` 当前仓库文件，确认 read-only tool 正常。
3. 让模型执行需要 permission 的 shell / edit，确认 permission modal 行为不变。
4. 输入 `/plan add auth flow`，确认 plan mode reminder 和 blocker 正常。
5. 让模型编辑一个文件后引入明显诊断，确认下一轮 diagnostics attachment 正常。
6. 让模型 spawn 一个 subagent，确认 sync / async task output 正常。
7. 长上下文或 mock context overflow 下确认 compaction / recovery 不回归。

## Acceptance Gates

该 PR 合入前必须满足：

- 新增 controller tests 全绿。
- 现有 engine / runtime / tools / subagents tests 全绿。
- `AgentEngine` public API 未破坏。
- TUI / gateway 无 wire schema 回归。
- `agent.py` 行数下降来自职责拆分，而不是删注释或格式压缩。
- docs `99_acceptance_matrix.md` 中全部场景有对应测试或手工验收说明。

## Rollback Plan

如果任何 controller extraction 导致高风险回归：

- 保留新增模块，但让 `AgentEngine` wrapper 走旧私有方法。
- 单独 revert 问题 PR，不影响已合入的 metadata contract。
- 不在 Sprint 13 内混入行为修复；行为修复单独开 bugfix PR。
