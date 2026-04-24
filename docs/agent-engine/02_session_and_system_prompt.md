# 02 会话与系统提示准备

## 功能点

`AgentEngine` 新增 `_prepare_session_and_system_prompt()`。

### 做了什么

- 支持 `EngineRequest.system_message`
- 从 `SessionStore` 读取历史 `system_prompt`（续跑）
- 注入 system 消息到首位并去重
- 写入 metadata：
  - `system_prompt_applied`
  - `system_prompt_source`
  - `system_prompt_preview`
- 可选执行 todo hydration（`enable_todo_hydration`）
- 在合适时机把 system prompt 回写 `SessionStore`

## 语义约束

- 若请求没有 `system_message` 且 store 也无快照，则行为与旧版本一致
- 同内容 system prompt 不重复插入
