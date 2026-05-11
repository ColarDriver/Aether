# Sprint 5 P2 UX / Observability — Acceptance Matrix

## PR 验收矩阵

| PR | 必须通过的专项测试 | 必须确认的回归点 |
|---|---|---|
| 5.1 HookOutcome + API hooks | hook outcome 注入、short-circuit、pre/post API metadata、hook 异常隔离 | 旧 `EngineHooks` subclass 不需要修改 |
| 5.2 Async steer | 多线程 append、tool result 注入、无 tool result pending、interrupt cleanup | role 序列不新增 user message |
| 5.3 Unicode recovery | surrogate retry、ASCII codec retry、credential sanitize、retry 上限 | UTF-8 正常非 ASCII 内容不被清洗 |
| 5.4 Reasoning + request dump | current-turn reasoning、跨 turn 隔离、dump 脱敏、默认关闭 | metadata shape 不变 |
| 5.5 Trajectory | success/failed JSONL、reasoning `<think>`、tool call/result conversion | 默认不写盘 |
| 5.6 Task cleanup | success/failure/interrupt finally cleanup、cleanup exception 隔离 | 原始 run_loop error 不被覆盖 |
| 5.7 Schema sanitizer | recursive strip、classifier、retry once、原 schema 不变 | 非 grammar error 不触发 |
| 5.8 Image shrink | 3 种 image shape、oversize shrink、Pillow 缺失降级、retry once | 不主动压缩正常图片 |
| 5.9 Rate guard | lock write/read/expiry/clear、fallback preflight、多线程 atomicity | guard 文件异常降级为 no-op |
| 5.10 Provider resilience | OAuth beta retry、dead connection rebuild/retry | provider-specific 逻辑不泄漏到通用接口 |

## 推荐测试命令

从 `backend/harness` 执行：

```bash
python -m pytest aether/tests/runtime aether/tests/engine aether/tests/models
```

如果 PR 只改单一模块，可先跑对应专项文件，再跑完整 runtime/engine/models 三组测试。

## 端到端验收场景

- 插件 hook 注入 `Memory: foo`，模型下一次请求能看到该上下文，session history 不持久污染。
- 用户在工具执行中发送 `/steer`，当前工具不中断，下一轮 LLM 能看到 `User guidance`。
- 含 lone surrogate 的消息第一次 provider call 失败后被清洗并 retry 成功。
- provider 返回失败后，开启 dump 配置会生成脱敏 JSON，关闭配置不会生成文件。
- 一个成功 turn 开启 trajectory 后写入 `sessions/{session_id}.jsonl`。
- provider 失败 turn 开启 trajectory 后写入 `failed/{session_id}.jsonl`。
- task_id 存在时，无论 success/failure/interrupt 都触发 release。
- llama.cpp grammar pattern error 后 schema 被剥离并 retry 一次。
- 6MB base64 image 遇到 image-too-large 后 shrink 并 retry。
- session A 写入 rate lock，session B 在 lockout 内不打同一 provider，优先 fallback。
- Anthropic OAuth long-context beta 未开通时，移除 1M beta 后 retry 一次。

## Sprint 完成定义

- 10 个 PR 文档对应能力全部实现。
- 每个 PR 至少有一个专项测试文件或明确扩展既有专项测试。
- 全量 runtime/engine/models 测试通过。
- `docs/run-loop-roadmap/04_p2_ux_observability_gaps.md` 的 P2-1 到 P2-12 状态可逐项更新为已完成或明确 deferred。
