# PR 13.5 — Recovery and Compaction Boundaries

## 目标

明确 provider recovery、empty response recovery、length continuation、tool-call truncation、invalid JSON retry、context overflow compaction 的边界，让 `AgentEngine` 不再直接承载大量 recovery 分支。该 PR 仍然是 facade-style extraction，不重写策略。

## 当前问题

Aether 的 recovery 能力已经很丰富，但分支分散在 `AgentEngine` 内：

- provider invocation error retry / backoff。
- classified recovery decision。
- rate limit fallback。
- context overflow / payload too large compaction。
- response invalid retry。
- empty response degradation。
- partial stream recovery。
- length continuation。
- truncated tool call retry。
- invalid JSON repair。
- fallback exhausted / compression exhausted exit reason。

这些分支和 provider call、message mutation、metadata counters 交织在一起。后续拆 provider transport 或 context curator 前，必须先把 recovery coordination 抽出。

## 实现改动

### 新增 recovery controller

新增 `aether/agents/runtime/recovery_controller.py`。

核心类型：

- `RecoveryAttemptInput`
  - `request: EngineRequest`
  - `messages: list[dict[str, Any]]`
  - `prepared_messages: list[dict[str, Any]]`
  - `context: TurnContext`
  - `invoke_provider: Callable[..., ProviderInvocationResult]`

- `RecoveryAttemptResult`
  - `response: NormalizedResponse | None`
  - `messages: list[dict[str, Any]]`
  - `interrupted: bool`
  - `exit_reason: ExitReason | None`
  - `error_text: str | None`
  - `continue_loop: bool`

- `RecoveryController`
  - owns coordination, not strategies。
  - constructed with `services`, `config`, `logger`, compaction adapter。

### Controller responsibilities

Controller 负责调用现有策略，而不是重写策略：

- 使用 `services.recovery_strategy` 解释 `ProviderInvocationError` / `ResponseInvalidError`。
- 在 decision 要求 fallback 时调用 existing `FallbackChain`。
- 在 decision 要求 compaction 时调用 existing compaction pipeline。
- 管理 per-turn retry counters。
- 处理 empty response recovery outcome。
- 处理 length continuation message override。
- 处理 truncated tool call retry。
- 处理 invalid JSON retry / tool-error injection。
- 产出现有 `ExitReason`。

### Compaction 边界

不改 `aether/services/compact` 的 tier 实现，只调整调用边界：

- `RecoveryController` 可以调用 `maybe_compact_messages(...)` adapter。
- preflight compaction 仍由 context assembly 负责。
- context overflow / payload too large 的 reactive compaction 由 recovery controller 负责。
- `CompactionResult` metadata 字段保持原样。

### AgentEngine 迁移

- `_invoke_provider_with_recovery` 可改为 delegating wrapper，内部调用 `RecoveryController.invoke_with_recovery(...)`。
- `_handle_empty_response`、length continuation、truncated tool call helper 可逐步迁入 controller。
- 主 loop 只关心 result：拿到 response、退出、继续、或失败。

## 测试

新增：

- `aether/tests/agents/runtime/test_recovery_controller.py`

覆盖：

- provider transient error 后 retry 成功。
- provider invalid response 后按现有 retry budget 行为。
- 429 rate limit 可触发 fallback 或 RATE_LIMITED。
- context overflow 触发 compaction adapter。
- compaction exhausted 产生 `COMPRESSION_EXHAUSTED`。
- empty response recovery 路径不变。
- length continuation 成功拼接。
- truncated tool call retry 不把 broken assistant message 写进 history。
- invalid JSON retry 超限后注入 tool error。

回归：

- existing recovery tests 全绿。
- compaction pipeline tests 全绿。
- provider fallback tests 全绿。

## 验收

- `AgentEngine` 不再直接承载主要 recovery decision 分支。
- Recovery controller 可用 fake provider invoker / fake compaction adapter 单测。
- `ExitReason` 值不变。
- recovery metadata 字段不变。
- 不改变 provider / middleware / tool public contract。
