# PR 8.4 - Project Memory Store

## 目标

实现可审计的项目级 memory store。Sprint 8 不使用 vector DB，而是采用
markdown topic files + index cache。

## 存储位置

默认优先项目内存储：

```text
.aether/memory/
  MEMORY.md
  index.json
  topics/
    architecture.md
    workflows.md
    decisions.md
    pitfalls.md
```

如果项目目录不可写，或用户配置了外部路径，则使用：

```text
~/.aether/memory/projects/<project_hash>/
```

`project_hash` 使用 canonical working directory 计算，避免不同项目互相污染。

## 新增/修改文件

| 文件 | 内容 |
|---|---|
| `backend/harness/aether/memory/project_store.py` | markdown store、index、读写 |
| `backend/harness/aether/memory/frontmatter.py` | frontmatter parser/renderer |
| `backend/harness/aether/memory/sanitize.py` | secret sanitizer |
| `backend/harness/aether/tests/memory/test_project_store.py` | store 测试 |

## MEMORY.md

`MEMORY.md` 是人工可读 index：

```markdown
# Aether Project Memory

This file indexes durable project memory used by Aether.

## Topics

- [Architecture](./topics/architecture.md)
- [Workflows](./topics/workflows.md)
- [Decisions](./topics/decisions.md)
- [Pitfalls](./topics/pitfalls.md)
```

## Topic Entry 格式

每条 memory 使用 frontmatter：

```markdown
---
id: project-decision-20260512-001
scope: project
kind: decision
created_at: 2026-05-12T00:00:00Z
updated_at: 2026-05-12T00:00:00Z
source_session: abc123
confidence: high
tags:
  - memory
  - compaction
paths:
  - backend/harness/aether/agents/core/agent.py
---

Memory injection is outbound-only. Retrieved memory must not be persisted
into canonical messages or compaction summaries.
```

## index.json

`index.json` 是缓存，不是权威数据。损坏时从 markdown 重建。

```json
{
  "version": 1,
  "updated_at": "2026-05-12T00:00:00Z",
  "entries": [
    {
      "id": "project-decision-20260512-001",
      "topic": "decisions",
      "path": "topics/decisions.md",
      "scope": "project",
      "kind": "decision",
      "tags": ["memory", "compaction"],
      "paths": ["backend/harness/aether/agents/core/agent.py"],
      "updated_at": "2026-05-12T00:00:00Z",
      "token_estimate": 42
    }
  ]
}
```

## 写入策略

默认只允许显式写入：

- 用户明确说“记住/写入项目记忆”。
- 模型调用 `memory_write` 且通过权限策略。
- 后续 PR 的 review flow 明确批准。

禁止默认自动写入：

- 用户普通偏好。
- 临时调试路径。
- shell 输出里的 secret。
- `.env`、private key、API token。
- retrieved memory 本身。

## 原子写和锁

写入流程：

1. 获取 store lock。
2. 读取 topic 当前内容。
3. 追加或更新 entry。
4. 写临时文件。
5. atomic replace。
6. rebuild index。

如果 lock 获取失败：

- read path 返回 stale index 或空结果。
- write path 返回结构化错误，不部分写入。

## Secret Sanitizer

首版覆盖：

- `sk-...`、`ghp_...`、`xoxb-...` 等常见 token。
- PEM private key block。
- `.env` 样式 `KEY=VALUE` 且 key 包含 `TOKEN/SECRET/API_KEY/PASSWORD`。
- AWS access key pattern。

命中策略：

- 默认拒绝写入。
- 对 read/injection 已存在内容做 redaction。
- metadata 记录 `redacted=True`，不记录原文。

## 测试

```python
def test_project_store_initializes_memory_md_and_topics(): ...
def test_project_store_appends_frontmatter_entry(): ...
def test_index_rebuilds_from_markdown_when_corrupted(): ...
def test_project_hash_is_stable_for_same_workdir(): ...
def test_store_falls_back_to_home_when_project_not_writable(): ...
def test_atomic_write_does_not_leave_partial_topic_file(): ...
def test_secret_like_content_is_rejected_or_redacted(): ...
```

## 验收门

- `MEMORY.md` 和 topic 文件可人工 review。
- `index.json` 删除或损坏后能恢复。
- 文件写入不会产生半截 memory。
- 默认不会写 user profile。
