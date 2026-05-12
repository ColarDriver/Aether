# Sprint 8 Acceptance Matrix

## 总体验收

| ID | 场景 | 配置 | 期望 |
|---|---|---|---|
| M-001 | 默认启动 engine | 默认配置 | `memory_enabled=True`, `memory_mode="project"`，user memory 不启用 |
| M-002 | `memory_mode="off"` | memory off | 不检索、不注入，metadata `skipped_reason="mode_off"` |
| M-003 | provider 抛异常 | project mode | 主 LLM call 继续，metadata 记录 memory error |
| M-004 | 检索超时 | timeout 1ms | 跳过注入，turn 不失败 |
| M-005 | 无相关候选 | 空 store | 不注入 `<memory_context>` |

## 注入与 Compaction

| ID | 场景 | 期望 |
|---|---|---|
| I-001 | 有相关 task memory | 注入最近 user message 的 hook context |
| I-002 | 注入后检查原 messages | canonical messages 未被修改 |
| I-003 | preflight compaction 开启 | compaction input 不包含 retrieved memory |
| I-004 | provider overflow 后 recovery compaction | retry 前重新计算 memory budget |
| I-005 | prompt 接近阈值 | memory budget 降低或 skipped |
| I-006 | trajectory persistence 开启 | 默认不记录 memory block text |

## Task / Session Memory

| ID | 场景 | 期望 |
|---|---|---|
| T-001 | 新 session | 创建独立 task snapshot |
| T-002 | 两个 session 并发 | task memory 不串线 |
| T-003 | 用户明确给约束 | constraints 被记录 |
| T-004 | 预算很小 | goal 和 constraints 优先保留 |
| T-005 | before_compaction | snapshot 可更新但 messages 不变 |
| T-006 | retrieved project memory 存在 | 不自动提升到 task snapshot |

## Project Store

| ID | 场景 | 期望 |
|---|---|---|
| P-001 | 首次启用 | 创建 `MEMORY.md` 和 `topics/` |
| P-002 | 写入 decision | topic entry 有 frontmatter 和 source |
| P-003 | 删除 `index.json` | 下次读取可重建 |
| P-004 | 损坏 entry | 跳过坏 entry，保留好 entry |
| P-005 | 项目目录不可写 | fallback 到 home project hash store |
| P-006 | 写入中断 | 不留下半截 topic 文件 |

## Retrieval / Budget

| ID | 场景 | 期望 |
|---|---|---|
| R-001 | task 与 project 同时命中 | task memory 排名更高 |
| R-002 | 用户提到文件路径 | path matched project memory boost |
| R-003 | 候选低相关 | 不注入 |
| R-004 | block 太大 | 裁剪后注入或丢弃 |
| R-005 | project mode | user memory 不参与 |
| R-006 | personal mode 但 user flag 关 | user memory 仍不参与 |
| R-007 | token budget 2500 | 渲染后不超过预算加 estimator reserve |

## Tools / Write Policy

| ID | 场景 | 期望 |
|---|---|---|
| W-001 | `memory_read` | 返回短摘要，不 dump topic 全文 |
| W-002 | `memory_write` project | 需要 reason，写入可审计 |
| W-003 | `memory_write` user in project mode | 拒绝 |
| W-004 | secret-like text | 拒绝或 redacted |
| W-005 | `memory_update` | 保持 entry id 和 provenance |
| W-006 | `memory_forget` | 写 tombstone |
| W-007 | 非交互写入 | 默认 deny |

## Observability / Safety

| ID | 场景 | 期望 |
|---|---|---|
| O-001 | memory disabled | metadata 仍有 `memory` key |
| O-002 | memory content logging default | 不记录 block text |
| O-003 | `memory_debug_log_content=True` | 内容日志经过 redaction |
| O-004 | stale memory conflict | 注入 policy 提醒以当前文件和用户消息为准 |
| O-005 | subagent | 默认可读 project memory，不读 user memory |
| O-006 | file lock timeout | 不无限阻塞 |
| O-007 | sanitizer failure | 写入失败而非绕过 sanitizer |

## Release Gate

Sprint 8 合入前必须满足：

- 所有 PR 的单元测试通过。
- 至少一个集成测试证明 memory 注入不改变 canonical messages。
- 至少一个集成测试证明 compaction summary 不包含 retrieved memory。
- 至少一个端到端测试证明 project memory 能跨 session 召回。
- 至少一个失败注入测试证明 provider error 不影响主 LLM call。
- 文档和默认配置明确 personal/user memory 默认关闭。

