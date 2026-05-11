# PR 5.5 — Trajectory Persistence

## 目标

把每个 turn 的用户、助手、工具交互以 JSONL 形式保存，用于调试、训练数据采集和回放分析。该能力默认关闭。

## 配置

```python
save_trajectories: bool = False
trajectory_dir: Path = Path("./trajectories")
```

目录结构：

```text
trajectories/
  sessions/
    {session_id}.jsonl
  failed/
    {session_id}.jsonl
```

每行 JSON：

```json
{
  "ts": "2026-05-11T00:00:00Z",
  "session_id": "...",
  "turn_id": "...",
  "task_id": "...",
  "model": "...",
  "provider": "...",
  "completed": true,
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."},
    {"from": "tool", "value": "..."}
  ]
}
```

## Conversion 规则

- `role="user"` -> `from="human"`。
- `role="assistant"` -> `from="gpt"`。
- `role="tool"` -> `from="tool"`。
- assistant reasoning 存在时，放在 assistant value 前部的 `<think>...</think>` block。
- assistant tool calls 用 `<tool_call>...</tool_call>` 包裹，每个 call 内部是 JSON。
- 连续 tool results 可以合并为一条 `from="tool"` value，但必须保留 `tool_call_id` 和 tool name。
- 只保存当前 run_loop 可见消息，不保存 API key、headers、provider raw kwargs。

## 文件改动

- 新增 `runtime/trajectory.py`。
- `runtime/contracts.py`：扩展 config。
- `agents/core/agent.py`：run loop 正常结束和失败收尾时调用 `_save_trajectory`。
- `EngineResult.metadata["trajectory"]`：记录 `saved: bool`、`path: str | None`、`error: str | None`。

## 参考实现

Hermes 的 trajectory converter 会把 reasoning 写入 `<think>`，tool call 写入 `<tool_call>`，tool result 写入 `<tool_response>`。Aether 应保留这个训练数据友好的格式，但字段外壳用更明确的 JSONL envelope。

## 隐私与安全

- 默认关闭。
- 不保存 request headers。
- 不保存 API credentials。
- 如果 tool result 已经包含用户敏感内容，trajectory 不做内容级自动判断；这是调用方开启该功能时的显式选择。
- 写文件失败只记录 metadata，不让 run loop 失败。

## 测试

- `tests/runtime/test_trajectory.py`
- `tests/engine/test_trajectory_persistence.py`
- 完成一个普通 user -> assistant turn，断言 `sessions/{session_id}.jsonl` 追加一行。
- provider 失败后保存到 `failed/{session_id}.jsonl`。
- assistant reasoning 被包进 `<think>`。
- assistant tool calls 和 tool responses 被稳定转换。
- `save_trajectories=False` 时不创建目录。
- 写文件异常被吞掉并写入 metadata error。

## 验收门

- 不改变 `EngineResult.final_response`。
- 不改变 session history。
- JSONL 每行可独立解析。
