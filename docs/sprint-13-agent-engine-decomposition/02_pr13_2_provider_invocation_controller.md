# PR 13.2 — Provider Invocation Controller

## 目标

把 provider 调用链从 `AgentEngine.run_loop` 和相关私有方法中抽出，形成一个轻量 `ProviderInvocationController`。该 PR 不改变 provider public contract，不引入 transport 层，只移动现有职责边界。

## 当前问题

Provider 调用目前和主 loop 耦合过深：

- active provider 读取。
- model / api_mode / provider metadata 采集。
- stream callback / silent callback 包装。
- API request debug dump。
- pre / post API hooks。
- provider.generate 调用。
- provider.validate_response。
- usage normalization。
- reasoning extraction。
- provider error 包装。
- api call count 和 elapsed metadata。

这些逻辑夹在 PRE_LLM、recovery、middleware、empty response 处理之间。后续如果要引入 Hermes 风格 `ProviderTransport`，现在的边界过于模糊。

## 实现改动

### 新增 controller

新增 `aether/agents/runtime/provider_invocation.py`。

核心类型：

- `ProviderInvocationRequest`
  - `request: EngineRequest`
  - `canonical_messages: list[dict[str, Any]]`
  - `prepared_messages: list[dict[str, Any]]`
  - `tools: list[ToolDescriptor]`
  - `context: TurnContext`
  - `stream_callback: StreamDeltaCallback | None`
  - `stream_silent_callback: StreamSilentCallback | None`

- `ProviderInvocationResult`
  - `response: NormalizedResponse | None`
  - `interrupted: bool`
  - `error: ProviderInvocationError | ResponseInvalidError | None`
  - `elapsed_ms: float`
  - `provider_name: str`
  - `api_mode: str`
  - `model: str | None`

- `ProviderInvocationController`
  - constructed with `services: EngineServices` and `hooks: EngineHooks`
  - method `invoke(invocation: ProviderInvocationRequest) -> ProviderInvocationResult`

### Controller responsibilities

Controller 负责现有 provider call 的“非策略”部分：

- 从 `services.provider` 获取当前 provider。
- 解析 model 名称，保持当前 `ModelCallConfig.extra["model"]` 优先级。
- 计算 message count、tool count、approx input tokens、request char count。
- 触发 `hooks.pre_api_request(...)`。
- 调用 `provider.generate(...)`。
- 调用 `provider.validate_response(...)`。
- 触发 `hooks.post_api_request(...)`。
- 标准化 usage，并写入 `context.metadata["usage"]` / accumulator。
- 提取 reasoning metadata。
- 增加 `context.metadata["api_calls"]` 或现有等价字段。
- 捕获 provider SDK 异常并包装为现有 `ProviderInvocationError`。

### AgentEngine 迁移

在 `AgentEngine` 中：

- 保留 `_invoke_provider_with_recovery` 作为 recovery 边界入口。
- 让 `_invoke_provider_with_recovery` 内部调用 `ProviderInvocationController.invoke(...)`。
- 暂时保留旧 helper 作为 delegating wrapper，避免单 PR 过大。
- 不改变 `_build_stream_callback` / `_build_stream_silent_callback` 的外部行为；若移动到 controller，只移动实现，不改 callback 语义。

### 不做的事

- 不拆 provider payload conversion。
- 不新增 `ProviderTransport`。
- 不改变 OpenAI / Claude / Codex provider 文件的 public behavior。
- 不改变 fallback chain 策略，只确保 controller 每次读取 `services.provider`。

## 测试

新增：

- `aether/tests/agents/runtime/test_provider_invocation_controller.py`

覆盖：

- scripted provider 返回 text，controller 返回 `NormalizedResponse`。
- scripted provider 返回 tool calls，tool calls 原样保留。
- `provider.validate_response()` 返回 false 时产生 `ResponseInvalidError`。
- provider 抛异常时产生 `ProviderInvocationError`。
- `pre_api_request` / `post_api_request` hooks 调用一次，字段完整。
- `stream_silent_callback` 收到 non-visible chunk 后 usage / token counter 行为不变。
- fallback chain 切换 active provider 后 controller 使用新 provider。

回归：

- existing provider tests 全绿。
- empty response recovery tests 全绿。
- activity token usage tests 全绿。

## 验收

- `AgentEngine.run_loop` 中 provider call 相关代码减少。
- Controller 可单独用 fake provider / fake hooks 测试。
- `ModelProvider.generate()` signature 不变。
- `EngineResult.metadata["api_calls"]`、`metadata["usage"]`、reasoning 字段不变。
- provider error / response invalid exit reason 不变。
