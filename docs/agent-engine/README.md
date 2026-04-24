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

- `backend/harness/aether/agents/core/agent.py`
- `backend/harness/aether/runtime/contracts.py`
- `backend/harness/aether/runtime/hooks.py`
- `backend/harness/aether/runtime/session_store.py`
