# 04 Hooks 与 Session Store

## 生命周期 Hooks

新增 `EngineHooks`（默认 no-op）：

- `on_session_start`
- `pre_llm_call`
- `post_llm_call`
- `on_session_end`

### 调用时机

- `on_session_start`：新会话准备阶段
- `pre_llm_call`：每次 LLM 调用前
- `post_llm_call`：每次 LLM 返回后
- `on_session_end`：`run_loop` 结束后

## Session Store

新增 `SessionStore` 抽象及 `InMemorySessionStore` 实现。

最小接口：

- `get_session(session_id)`
- `update_system_prompt(session_id, prompt_snapshot)`

默认情况下 `AgentEngine` 会自动使用内存 store，避免未注入 store 时功能失效。
