# 03 流式回调与 Provider 事件

## 回调定义

`EngineRequest.stream_callback`：可选函数，签名约定为 `Callable[[str], None]`，用于接收文本增量。

## 引擎侧

- `_build_stream_callback()` 封装回调，统计：
  - `streamed_output`
  - `stream_callback_calls`
- provider 调用时透传 `stream_callback`
- 若 provider 未提供实时 delta，但有最终文本，执行 fallback 回调

## Provider 侧

- `ModelProvider.generate(..., stream_callback=None)` 接口统一
- `ScriptedProvider`：文本响应时直接触发回调
- `CodexChatModel`：SSE 解析中尝试提取事件 delta 并逐段回调
- `ClaudeChatModel`：暂以最终文本 fallback 回调

## 事件粒度说明

Codex SSE 事件存在异构结构，当前回调提取策略覆盖：

- `delta` 字段（字符串/对象）
- `output_text` 相关事件
- `item.text` / `item.content[*].text`

无法识别的事件会被忽略，不中断主流程。
