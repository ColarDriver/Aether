# Sprint 5 — P2 UX / Observability Hardening

> 设计版本：v1.0。
> 来源：[`docs/run-loop-roadmap/04_p2_ux_observability_gaps.md`](../run-loop-roadmap/04_p2_ux_observability_gaps.md)。
> 参考实现：`/workspace/hermes-agent` 与 `/workspace/open-claude-code`。

本目录把 P2 UX / 可观测性缺口拆成可独立 review、可独立测试、可回滚的小 PR。拆分原则是：

- 先补 run loop 扩展点，再补依赖这些扩展点的用户纠偏能力。
- 先补观测数据，再补依赖观测数据的 trajectory。
- Provider / 本地后端 / rate guard 这类恢复能力各自独立，不混进主循环大 PR。
- 所有行为必须有专项测试；不能只靠端到端手测。

## 文件导航

| 文档 | 内容 | 角色 |
|---|---|---|
| [`00_overview.md`](./00_overview.md) | 整体拆分、依赖关系、Aether 当前状态、参考实现索引 | 入口 |
| [`01_hook_outcome_and_api_hooks.md`](./01_hook_outcome_and_api_hooks.md) | PR 5.1 — HookOutcome + pre/post API hooks | 地基 |
| [`02_async_steer_inbox.md`](./02_async_steer_inbox.md) | PR 5.2 — `/steer` 异步用户引导 | 用户纠偏 |
| [`03_unicode_payload_recovery.md`](./03_unicode_payload_recovery.md) | PR 5.3 — surrogate / ASCII codec payload recovery | 环境适配 |
| [`04_reasoning_and_failed_request_observability.md`](./04_reasoning_and_failed_request_observability.md) | PR 5.4 — reasoning 抽取 + failed request dump | 可观测性 |
| [`05_trajectory_persistence.md`](./05_trajectory_persistence.md) | PR 5.5 — trajectory JSONL 持久化 | 数据采集 |
| [`06_task_resource_cleanup.md`](./06_task_resource_cleanup.md) | PR 5.6 — task-scoped resource cleanup | 资源治理 |
| [`07_schema_sanitizer_for_local_backends.md`](./07_schema_sanitizer_for_local_backends.md) | PR 5.7 — llama.cpp grammar schema sanitizer | 本地后端兼容 |
| [`08_image_too_large_shrink_recovery.md`](./08_image_too_large_shrink_recovery.md) | PR 5.8 — `image_too_large` shrink retry | 多模态兼容 |
| [`09_cross_session_rate_guard.md`](./09_cross_session_rate_guard.md) | PR 5.9 — 跨会话 rate guard | 并发安全 |
| [`10_provider_network_resilience.md`](./10_provider_network_resilience.md) | PR 5.10 — Anthropic OAuth 1M beta disable + dead connection cleanup | Provider 韧性 |
| [`99_acceptance_matrix.md`](./99_acceptance_matrix.md) | 全 sprint 验收矩阵、测试命令、回归风险 | 验收 |

## 推荐实施顺序

1. PR 5.1：先让 hook 能返回结构化 outcome，并暴露 API-call 观测点。
2. PR 5.2：在 hook 地基上实现 `/steer`，确保异步纠偏不破坏 tool-call role 序列。
3. PR 5.3：补 Unicode payload recovery，降低环境编码导致的 provider 前失败。
4. PR 5.4：补 reasoning 和 request dump，为后续 trajectory 提供可信字段。
5. PR 5.5：补 trajectory persistence。
6. PR 5.6：补 task resource cleanup。
7. PR 5.7、5.8、5.9、5.10 可并行，分别处理本地后端、多模态、跨会话配额、provider 网络韧性。

## 关键约束

- `EngineResult.metadata` 只能 additive 扩展，不能改名或收窄 Sprint 3 已稳定的字段形状。
- Hook 异常必须被隔离，不能让插件或上层观测逻辑破坏 run loop。
- Recovery 行为必须有 retry 上限，并在 metadata 中留下是否触发过的痕迹。
- 默认配置必须保持现有用户行为不破坏：trajectory 和 failed request dump 默认关闭。
- 涉及敏感信息的 dump 必须脱敏，尤其是 API key、Authorization header、cookie、token。

下一步：阅读 [`00_overview.md`](./00_overview.md)。
