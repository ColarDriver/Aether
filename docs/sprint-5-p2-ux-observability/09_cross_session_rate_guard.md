# PR 5.9 — Cross-Session Rate Guard

## 目标

当一个 session 已经确认某 provider/base_url 处于 rate limit 时，其他 session 在 lockout 期间不要继续请求同一 provider，避免多会话并发把配额打穿。

## 设计

新增配置：

```python
rate_guard_dir: Path | None = None
rate_guard_enabled: bool = True
```

默认路径：

- `$XDG_RUNTIME_DIR/aether/rate_guard/` 存在时优先使用。
- 否则使用平台临时目录下的 `aether/rate_guard/`。

lock 文件：

```json
{
  "provider": "openai-compatible",
  "base_url_hash": "...",
  "until_unix": 1770000000.0,
  "reason": "rate_limit",
  "source_session_id": "...",
  "created_unix": 1769999900.0
}
```

key：

- `{provider}_{sha256(base_url or model namespace)[:16]}.json`。
- 不把完整 base_url 写入文件名。

运行规则：

- 每次 LLM call 前检查 guard。
- 如果 lockout 未过期，直接请求 fallback chain activate next；不打当前 provider。
- 如果没有 fallback，返回明确 rate-limit blocked error。
- 收到 429 或 classifier reason `rate_limit` 且 headers 有 reset/retry-after 时写 lock。
- 成功响应后清除该 provider key 的 lock。
- 文件读写使用 file lock 或 atomic replace，保证多进程安全。

## 文件改动

- 新增 `runtime/rate_guard.py`。
- `runtime/contracts.py`：新增 config。
- `runtime/recovery.py` 或 run loop provider-call 前：接入 preflight check。
- `provider_errors.py`：确保 headers / retry-after metadata 可被 guard 读取。
- metadata：记录 `rate_guard.checked`、`blocked`、`until_unix`、`fallback_activated`。

## 参考实现

- Hermes 对 Nous Portal 做跨 session lockout，避免多个 session 在 RPH 限制下重复消耗失败请求。
- open-claude-code 在 fast mode 上有 retry-after/cooldown 设计；Aether 这里不做 fast/slow mode，只做 provider-level guard。

## 测试

- `tests/runtime/test_rate_guard.py`
- 写入 60s lockout 后，另一个 session preflight 返回 blocked。
- 过期 lock 被忽略并清理。
- 成功响应清除 lock。
- 两个进程/线程同时写 lock，不产生破损 JSON。
- run loop 遇到 active lock 且有 fallback，断言 provider.generate 未调用当前 provider，fallback 被调用。
- active lock 且无 fallback，返回明确错误并不调用 provider.generate。

## 验收门

- lockout 是 provider/base_url 粒度，不是全局禁用所有模型。
- guard failure 不能让 run loop 崩溃；文件系统异常时降级为不使用 guard。
- 不持久保存 API key 或完整 request。
