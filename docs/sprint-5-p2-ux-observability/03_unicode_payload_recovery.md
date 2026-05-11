# PR 5.3 — Unicode Payload Recovery

## 目标

让 provider 调用在遇到 lone surrogate 或 ASCII codec 环境问题时自动清洗 payload 并有限 retry，避免用户因为复制粘贴或 `LANG=C` 环境直接失败。

## 当前问题

- Aether 只有 turn 入口的 `_sanitize_text` 类处理，无法覆盖 provider payload 的所有派生字段。
- retry 时不会重新清理 `api_kwargs`、tool schema、prefill/system/header/API key 等路径。
- 没有 ASCII codec 专门路径。

## 设计

新增文本 helper：

```python
def strip_surrogates(text: str) -> str: ...
def strip_non_ascii(text: str) -> str: ...
def sanitize_structure_surrogates(value: Any) -> bool: ...
def sanitize_structure_non_ascii(value: Any) -> bool: ...
```

错误识别：

- `UnicodeEncodeError` 且错误字符串包含 `surrogate`，或 UTF-8 encode failure 指向 surrogate：全 payload strip surrogates。
- `UnicodeEncodeError` 且错误字符串包含 `ascii`，或运行环境表现为 ASCII codec：启用 `_force_ascii_payload`，全 payload strip non-ascii。
- `context.metadata["unicode_recovery_passes"]` 最多为 2。

清洗范围：

- canonical `messages`。
- API-call copy / `api_kwargs`。
- tool schema / tool descriptors。
- system message / transient system addendum。
- prefill messages。
- provider headers。
- API key / auth token，仅 ASCII codec 路径处理，并写 warning 日志提示用户重新复制 credential。

## 文件改动

- 新增或扩展 `runtime/unicode_sanitizer.py`。
- `agents/core/agent.py`：在 provider.generate exception 分支接入 recovery。
- provider payload 构建处：如果 `_force_ascii_payload` 为 True，在发请求前 sanitize `api_kwargs`。
- metadata：记录 `unicode_recovery_passes`、`unicode_recovery_reason`、`force_ascii_payload`。

## 参考实现

Hermes 对 `UnicodeEncodeError` 做两阶段处理：

- surrogate 路径：清理 messages、api_messages、api_kwargs、prefill。
- ASCII codec 路径：额外清理 tools、system prompt、headers、credential，并设置 force-ascii 标记。

Aether 应采用相同覆盖范围，但 helper 放在 runtime 层，不把大量 walker 逻辑内联到 run loop。

## 测试

- `tests/runtime/test_unicode_sanitizer.py`
- `tests/engine/test_unicode_payload_recovery.py`
- 消息含 `\udcff`，provider 第一次 raise surrogate `UnicodeEncodeError`，第二次成功，断言 provider 第二次收到的 payload 无 surrogate。
- API key 含非 ASCII 字符，ASCII codec error 后 key 被清洗，warning 日志出现，第二次成功。
- tool schema description 含非 ASCII，ASCII codec path 后被清洗。
- provider 连续抛 UnicodeEncodeError，最多 retry 2 次后正常失败，不无限循环。
- 非 UnicodeEncodeError 不进入该 recovery。

## 验收门

- 默认 UTF-8 环境下不改变正常非 ASCII 用户内容。
- 只有 ASCII codec recovery 才 strip 全部非 ASCII。
- 清洗行为必须在 metadata 中可观察。
