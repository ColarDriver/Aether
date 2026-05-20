# PR 13.1 — Runtime Context and Metadata Contract

## 目标

把 `TurnContext.metadata` 的内部 key、公开 metadata、runtime live object 引用和 snapshot 逻辑集中到一个可维护的 contract 中。该 PR 不改变业务行为，只为后续拆分 provider / context / tool / recovery controller 提供稳定基础。

## 当前问题

`AgentEngine` 目前在多个位置读写 `context.metadata`：

- retry counters：empty response、provider error、invalid JSON、phantom tool、length continuation。
- live runtime refs：provider config、parent agent、subagent manager、approval prompter、tool permission prompter、skill catalog、task store、LSP / browser / diagnostics manager。
- public observability：usage、api_calls、memory、trajectory、resource_cleanup、iteration_budget、exit、reasoning、compaction。
- feature flags：plan_mode_active、should_review_memory、should_review_skills、verifier state。

这些 key 一部分定义在 `aether/runtime/session/session_runtime.py`，一部分定义在 `AgentEngine._METADATA_INTERNAL_KEYS`，还有一部分散落在 run loop 和 helper 内。新增功能很容易把 live object 泄露进 `EngineResult.metadata["turn"]`，或者因为 typo 导致某个 recovery path 静默失效。

## 实现改动

### 新增 metadata contract 模块

新增 `aether/runtime/core/turn_metadata.py`，集中定义：

- `TURN_RETRY_COUNTER_KEYS`：继续复用或 re-export `session_runtime.py` 的 retry key。
- `RUNTIME_REF_KEYS`：所有 live object key，例如 `_engine_config`、`_parent_agent`、`_subagent_manager`、`_approval_prompter`、`_tool_permission_prompter`、`_skill_catalog`、`_task_store`、`_lsp_manager`、`_diagnostic_tracker`、`_browser_manager`。
- `INTERNAL_METADATA_KEYS`：所有不能进入 public snapshot 的 key，包含 runtime refs、internal counters、temporary loop keys、resource handles。
- `PUBLIC_STABLE_KEYS`：文档化当前承诺稳定的 top-level metadata keys，例如 `request`、`turn`、`runtime`、`usage`、`api_calls`、`memory`、`trajectory`、`resource_cleanup`、`iteration_budget`、`exit`、`reasoning`、`compaction`。

### 新增 helper

提供最小 helper，不引入复杂 abstraction：

- `init_turn_retry_counters(metadata: dict[str, Any]) -> None`
- `set_runtime_ref(context: TurnContext, key: str, value: Any) -> None`
- `get_runtime_ref(context: TurnContext, key: str, expected_type: type[T] | None = None) -> T | None`
- `public_turn_metadata(metadata: dict[str, Any]) -> dict[str, Any]`
- `sanitize_metadata_value(value: Any) -> Any`

`public_turn_metadata()` 必须保留当前 `_build_result` 的行为：deep-copy / dataclass / list / dict / primitive normalization 与 internal key filtering 语义不变。

### 迁移 AgentEngine

在 `aether/agents/core/agent.py` 中：

- 删除或改为 import `AgentEngine._METADATA_INTERNAL_KEYS` 的定义。
- `_prepare_turn_entry` 中初始化 retry counters 时改用 `init_turn_retry_counters()`。
- 向 `context.metadata` 写 live object 时改用 `set_runtime_ref()`。
- `_build_result` 中构建 `metadata["turn"]` 时改用 `public_turn_metadata()`。
- 保留现有 metadata 字段名，不重命名公开字段。

### 兼容策略

- `session_runtime.py` 中已有常量不要直接删除；可以 re-export 或在 `turn_metadata.py` import，避免大量旧测试路径失效。
- 该 PR 允许 `AgentEngine` 仍然包含主要业务逻辑，只清理 metadata contract。
- 不改变 `EngineResult.metadata` shape。

## 测试

新增或扩展：

- `aether/tests/runtime/core/test_turn_metadata.py`
- `aether/tests/agents/test_engine_metadata_contract.py`

覆盖：

- internal live refs 不出现在 `EngineResult.metadata["turn"]`。
- dataclass / enum / list / dict metadata 能正常 JSON-friendly snapshot。
- retry counters 初始化为当前默认值。
- unknown public primitive metadata 仍保留，保持现有 ad-hoc observability 兼容。
- `usage`、`iteration_budget`、`memory`、`compaction`、`resource_cleanup` top-level 字段不回归。
- plan mode 下 `plan_mode_active` 仍可观察。

## 验收

- PR 合入后，所有现有 engine tests 行为不变。
- `AgentEngine` 中 metadata key 的集中定义明显减少。
- 后续 PR 可以通过 helper 访问 runtime refs，不需要直接拼写 `_task_store` / `_diagnostic_tracker` 等字符串。
- 没有任何 TUI / gateway schema 改动。
