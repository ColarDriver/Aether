# PR 5.10 — Provider Network Resilience

## 目标

补两个 provider/network 层韧性能力：

- Anthropic OAuth 用户遇到 1M context beta 未开通时，自动移除该 beta 并 retry 一次。
- OpenAI-compatible provider 在跨 turn 复用 client 前清理死连接，降低 keep-alive stale socket 导致的首请求 hang。

## Anthropic OAuth 1M Beta Disable

当前 Aether Claude provider 已检测 OAuth token 并设置 `anthropic-beta` header。新增 provider 内部状态：

```python
_oauth_1m_beta_disabled: bool = False
```

行为：

- 仅 OAuth token 路径生效。
- provider 收到 400/401/403，body 包含 `long context beta` 和 `not yet available` 时触发。
- 从 `anthropic-beta` header 移除 1M context beta，设置 `_oauth_1m_beta_disabled=True`，重建 client，retry 一次。
- 该状态只在内存中保存，不写配置，不跨进程持久化。
- 如果用户后续重启进程且权限已开通，beta 会恢复正常尝试。

参考：Hermes 在 model list 请求中遇到 long-context beta 拒绝后，移除 `_CONTEXT_1M_BETA` 再 retry。

## Dead Connection Cleanup

新增 provider optional method：

```python
def cleanup_dead_connections(self) -> bool:
    ...
```

OpenAI-compatible provider 行为：

- 在新 turn 入口或 provider.generate 前轻量检查 client / transport 状态。
- 如果确认 connection pool 有 stale/dead connection，关闭 client 并重建。
- 如果无法可靠检测，则提供 conservative fallback：捕获特定 connection reset / stale keep-alive error 后重建并 retry 一次。
- 返回值表示是否重建过 client。

注意：不要做昂贵的主动网络探测；cleanup 应优先是本地 client/pool 状态管理。

## 文件改动

- `models/provider/claude.py`：新增 OAuth beta disable/retry。
- `models/provider/openai_compatible.py`：新增 `cleanup_dead_connections`。
- `models/provider/base.py`：可选 protocol / default no-op。
- `agents/core/agent.py`：在跨 turn provider call 前调用 provider cleanup；或者 provider.generate 内部自处理。
- metadata：记录 `oauth_1m_beta_disabled`、`dead_connections_cleaned`。

## 测试

- `tests/models/test_claude_oauth_beta_disable.py`
- OAuth token + long-context beta error，断言第一次请求带 beta，第二次请求移除 1M beta，且只 retry 一次。
- 非 OAuth token 不触发。
- 其他 401/403 不触发该 recovery。
- `tests/models/test_openai_compatible_dead_connections.py`
- fake client 标记 stale，cleanup 后 rebuild 被调用且返回 True。
- cleanup 抛异常时记录日志，不影响后续 generate 正常路径。
- stale connection error 后 rebuild + retry 一次。

## 验收门

- OAuth beta disable 不持久化。
- dead connection cleanup 不引入每次请求的额外网络 round trip。
- provider-specific 逻辑留在 provider 层，不污染通用 recovery strategy。
