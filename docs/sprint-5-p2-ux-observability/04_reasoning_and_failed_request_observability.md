# PR 5.4 — Reasoning + Failed Request Observability

## 目标

补齐两个调试和运营能力：

- 从当前 turn 抽取最后一段有效 reasoning，填入 `EngineResult.metadata["reasoning"]["last_reasoning"]`。
- provider 失败时按配置 dump 脱敏后的 API request，便于复现和定位。

## Reasoning 抽取

新增 helper：

```python
def extract_last_reasoning(messages: list[dict[str, Any]], turn_start_idx: int) -> str | None:
    ...
```

规则：

- 从 `messages` 末尾向前扫描。
- 遇到 index 小于 `turn_start_idx` 停止。
- 遇到 `role="user"` 停止，避免跨 turn。
- 第一条非空 `reasoning`、`reasoning_content`、`metadata.reasoning_content`、`metadata.reasoning_details` 即为结果。
- 如果 `reasoning_details` 是 list/dict，转为紧凑 JSON 字符串。
- 不把 tool-call arguments 或 silent streaming delta 当 reasoning。

写入：

- `context.metadata["last_reasoning_text"]`。
- 最终继续通过既有 metadata shape 暴露：`metadata["reasoning"]["last_reasoning"]`。

## Failed Request Dump

新增配置：

```python
dump_failed_requests: bool = False
request_dump_dir: Path = Path("./request_dumps")
```

新增 helper：

```python
def dump_api_request_debug(
    api_kwargs: dict[str, Any],
    *,
    model: str,
    provider: str,
    base_url: str | None,
    reason: str,
    error: Exception | None,
    dump_dir: Path,
) -> Path:
    ...
```

触发点：

- max retries exhausted。
- non-retryable client/provider error。
- invalid response after fallback/recovery exhausted。

脱敏规则：

- 递归替换 key 匹配 `api_key`、`authorization`、`cookie`、`token`、`secret`、`password` 的值为 `"<redacted>"`。
- 对超长 message content 做长度上限截断，并保留原长度字段。
- 文件名使用 UTC timestamp、session_id、reason，避免覆盖。

## 文件改动

- 新增 `runtime/reasoning.py`。
- 新增 `runtime/request_dump.py`。
- `runtime/contracts.py`：扩展 `EngineConfig`。
- `agents/core/agent.py`：在 result metadata 构建前调用 reasoning extraction；在 provider failure 分支调用 dump helper。

## 参考实现

Hermes 在 final result 附近抽取 `last_reasoning`，并有 `_dump_api_request_debug` 在 env flag 打开时落盘。Aether 应用 config 控制替代 env-only 方案，避免生产环境 surprise。

## 测试

- `tests/runtime/test_reasoning.py`
- `tests/runtime/test_request_dump.py`
- `tests/engine/test_reasoning_metadata.py`
- 当前 turn 内有 3 条 assistant，最后一条无 reasoning，倒数第二条有 reasoning，断言抽到倒数第二条。
- 前一 turn 的 reasoning 不应被抽到。
- `reasoning_details` 为 list 时输出稳定 JSON。
- 触发 provider error 且 `dump_failed_requests=True`，断言 dump 文件存在，包含 model/provider/base_url/error/reason。
- dump 中 Authorization、api_key、token 被脱敏。
- `dump_failed_requests=False` 时不写文件。

## 验收门

- `metadata["reasoning"]` shape 不变，只把 `last_reasoning` 从占位变为真实值。
- dump 默认关闭。
- dump helper 失败不能覆盖原 provider error。
