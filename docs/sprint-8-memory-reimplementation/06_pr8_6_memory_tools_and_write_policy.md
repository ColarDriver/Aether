# PR 8.6 - Memory Tools + Write Policy

## 目标

提供模型可调用的 memory 工具，同时保证写入是可控、可审计、可拒绝的。

## 工具列表

| 工具 | 默认权限 | 说明 |
|---|---|---|
| `memory_read` | allow | 查询 memory，返回短摘要 |
| `memory_list` | allow | 列出 topic / entry metadata |
| `memory_write` | ask | 新增 project memory entry |
| `memory_update` | ask | 更新已有 entry |
| `memory_forget` | ask | 删除或 tombstone entry |

`memory_read` 和 `memory_list` 属于 cheap tools，可继续放在
`cheap_tool_names` 中。写类 memory tool 不应 refund，除非后续证明无副作用。

## 新增/修改文件

| 文件 | 内容 |
|---|---|
| `backend/harness/aether/tools/builtins/memory.py` | memory tools |
| `backend/harness/aether/memory/write_policy.py` | 写入策略 |
| `backend/harness/aether/tools/registry.py` | 注册 memory tools |
| `backend/harness/aether/tests/tools/test_memory_tools.py` | 工具测试 |

## Tool Schema

`memory_write` 参数：

```json
{
  "scope": "project",
  "kind": "decision",
  "topic": "decisions",
  "text": "...",
  "tags": ["memory"],
  "paths": ["backend/harness/aether/agents/core/agent.py"],
  "confidence": "high",
  "reason": "User explicitly asked to remember this project decision."
}
```

限制：

- 默认只允许 `scope="project"`。
- `scope="user"` 只有 personal assistant 模式允许。
- `text` 必须小于配置上限，例如 2000 chars。
- `reason` 必填，便于权限确认和审计。

## 写入策略

| 场景 | 默认结果 |
|---|---|
| 用户明确说“记住这个项目约定” | 允许请求 `memory_write`，仍可确认 |
| 模型自行总结用户偏好 | 拒绝 user memory |
| shell 输出包含 token | 拒绝或 redacted |
| retrieved memory 再写回 | 拒绝 |
| task decision 提升到 project memory | 需要明确用户指令或工具确认 |

## Permission 集成

如果 Sprint 7 tool permission 已启用：

- `memory_write/update/forget` 走同一 permission gate。
- preview 显示 scope、topic、kind、paths、正文摘要。
- `accept once` 只允许本次写入。
- `accept session` 只能对 project scope 生效，不能放开 user scope。

非交互模式：

- read/list allow。
- write/update/forget 默认 deny，除非 `memory_auto_write_enabled=True` 且 scope 是 project。

## Tool Result 格式

工具结果必须短：

```text
Memory written:
- id: project-decision-20260512-001
- scope: project
- topic: decisions
- source: .aether/memory/topics/decisions.md
```

禁止把完整 topic 文件回填给模型。

## Forget 语义

默认使用 tombstone 而非物理删除：

```yaml
deleted: true
deleted_at: ...
delete_reason: ...
```

物理删除可以后续加 `memory_prune`，不在 Sprint 8 默认工具中提供。

## 测试

```python
def test_memory_read_returns_short_relevant_summary(): ...
def test_memory_write_project_scope_requires_reason(): ...
def test_memory_write_user_scope_denied_in_project_mode(): ...
def test_memory_write_rejects_secret_like_text(): ...
def test_memory_update_preserves_entry_id_and_source(): ...
def test_memory_forget_writes_tombstone(): ...
def test_non_interactive_memory_write_denied_by_default(): ...
def test_memory_tool_results_do_not_dump_full_topic_file(): ...
```

## 验收门

- 模型可以显式读写项目 memory。
- 默认不会写个人长期记忆。
- 写类操作可被权限系统拦截。
- 工具结果不会显著增加上下文负担。
