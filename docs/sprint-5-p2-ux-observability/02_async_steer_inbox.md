# PR 5.2 — Async `/steer` Inbox

## 目标

支持用户在模型推理或工具执行期间输入 `/steer`，不中断当前工具调用，把引导文本注入下一次 LLM 调用可见的上下文中。

## 语义

`/steer` 不是 interrupt：

- 不取消当前 tool call。
- 不新增一条 role=user 消息打破 tool-call role 序列。
- 在当前工具 batch 结束后、下一轮 LLM 前，把文本追加到最近一条 `role="tool"` 的 content 末尾。
- 如果当前 turn 没有 tool result，则保留 pending steer，run loop 结束时通过 `EngineResult.metadata["pending_steer"]` 交给上层作为下一轮 user message 处理。

Hermes 的实现采用同样语义：pending steer 用 lock 保护，多次 steer 用换行合并，drain 点在工具结果进入消息列表后。

## 设计

新增 `SteerInbox`：

```python
class SteerInbox:
    def append(self, session_id: str, text: str) -> bool: ...
    def drain(self, session_id: str) -> str | None: ...
    def put_back(self, session_id: str, text: str) -> None: ...
    def clear(self, session_id: str) -> None: ...
```

新增 engine API：

```python
def send_steer(self, session_id: str, text: str) -> bool:
    return self.services.steer_inbox.append(session_id, text)
```

注入规则：

- drain 时机：tool batch result append 完成后，下一次 provider.generate 前。
- target：从本 batch 尾部反向查找最近 `role="tool"`。
- string content：追加 `\n\nUser guidance: {text}`。
- list/multimodal content：追加 `{"type": "text", "text": "User guidance: ..."}`。
- 找不到 tool message：把文本放回 inbox。
- interrupt 或 hard stop 后：清理该 session pending steer，避免晚到注入污染下一轮。

## 文件改动

- 新增 `runtime/steer.py`。
- `runtime/services.py`：把 `SteerInbox` 放到 `EngineServices`，与 interrupt controller 同级。
- `agents/core/agent.py`：新增 `send_steer`，并在工具结果 append 后调用 `_apply_pending_steer_to_tool_results`。
- CLI 层后续可把 `/steer` command 接到 `AgentEngine.send_steer`；本 PR 至少保证 engine API 和测试完整。

## 边界条件

- 空文本或全空白返回 `False`。
- 同一 session 多次 steer 按输入顺序合并，中间用 `\n`。
- 不同 session 互不影响。
- pending steer 不写进 session history，除非成功追加到 tool result；未消费时仅出现在 result metadata。
- 如果 tool content 非字符串也非 list，fallback 为字符串拼接，但要保留原内容可读性。

## 测试

- `tests/engine/test_steer_inbox.py`
- 多线程 append：多个线程同时 `send_steer`，drain 后顺序稳定且无丢失。
- run loop 中 provider 第一次返回 tool call，工具执行期间 append steer，第二次 provider.generate 的最后 tool message 包含 `User guidance`。
- 无 tool result 时 append steer，run loop 结束 metadata 包含 `pending_steer`。
- interrupt 后 pending steer 被清空。
- multimodal tool content list 被追加 text block，而不是转成不可读字符串。

## 验收门

- `/steer` 不会生成额外 user role 消息。
- provider 收到的消息 role 序列仍满足 tool-call 协议。
- 并发 append 不会抛异常或跨 session 泄漏。
