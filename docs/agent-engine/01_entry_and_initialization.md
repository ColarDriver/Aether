# 01 入口与初始化

## 功能点

`AgentEngine` 新增 `_prepare_turn_entry()`，把每轮启动阶段结构化。

### 做了什么

- 深拷贝并清洗 `request.messages`
- 追加当前 `user_message`（若存在）
- 生成并写入 `task_id`、`turn_id`
- 重置本轮计数器：
  - `empty_response_retries`
  - `provider_error_retries`
- 标准化 metadata：
  - `entry_prepared`
  - `request_has_stream_callback`

## 设计目的

- 避免请求对象在主循环里被直接污染
- 保证每轮有独立可追踪 ID
- 将“防炸初始化”从主逻辑拆出，降低 `run_loop` 认知复杂度
