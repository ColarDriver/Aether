# AgentEngine 增强设计总览

## 背景
本章节记录 Aether `AgentEngine` 在本次改造中新增的两段前置流程与配套能力：

- 入口与初始化（Entry & Initialization）
- 会话与系统提示准备（Session & System Prompt Preparation）

同时补齐以下能力：

- 流式回调（stream callback）
- todo 状态回填（hydration）
- memory / skill nudge 计数器
- 生命周期 hooks
- system prompt 会话级持久化

## 文档目录

- [01_入口与初始化](./01_entry_and_initialization.md)
- [02_会话与系统提示准备](./02_session_and_system_prompt.md)
- [03_流式回调与Provider事件](./03_streaming_callback_and_provider_events.md)
- [04_hooks与session_store](./04_hooks_and_session_store.md)
- [05_nudge机制](./05_nudge_mechanism.md)
- [06_测试矩阵](./06_test_matrix.md)
- [07_配置建议_env_vs_hermes_style](./07_configuration_strategy.md)

## 主要代码入口

- `aether/agents/core/agent.py`
- `aether/agents/runtime/session_lifecycle.py`
- `aether/agents/runtime/context_assembly.py`
- `aether/agents/runtime/provider_invocation.py`
- `aether/agents/runtime/recovery_controller.py`
- `aether/agents/runtime/tool_dispatch.py`
- `aether/agents/runtime/turn_runner.py`
- `aether/agents/runtime/response_finalization.py`
- `aether/agents/runtime/response_repair.py`
- `aether/agents/runtime/stream_controller.py`
- `aether/runtime/core/contracts.py`
- `aether/runtime/core/hooks.py`
- `aether/runtime/session/session_store.py`

## 当前运行时边界

Sprint 15 之后，`AgentEngine.run_loop()` 只保留 public facade 语义：
设置当前 session id，调用 `TurnRunner.run(...)`，最后清理当前 session id。
具体 turn 执行职责由 runtime 层组件承担：

- `TurnRunner`：loop 状态、iteration budget、provider/tool/finalization 调度。
- `ResponseFinalizationController`：无 tool response、empty response、phantom tool 的 finalization 决策。
- `ResponseRepairController`：length continuation、truncated tool call、invalid JSON argument 的 repair 决策。
- `StreamController`：visible/silent stream callback 构造和 interrupt 检查。
