# PR 7.6 - Tests + Observability Polish

## 目标

补齐端到端专项测试、UI 回归测试和 observability。这个 PR 不再引入大逻辑，重点是把“不会闪 function call JSON、不会污染输出框、拒绝不执行”这些用户可见行为 pin 住。

## 观测字段

`EngineResult.metadata["tool_permissions"]` 建议最终形态：

```python
{
    "enabled": True,
    "asked": 2,
    "allowed_once": 1,
    "allowed_session": 1,
    "denied": 0,
    "aborted": 0,
    "non_interactive_denied": 0,
    "session_rules_added": 1,
}
```

每个 denied synthetic `ToolResult.metadata`：

```python
{
    "permission_denied": True,
    "permission_decision": "reject",
    "tool_executed": False,
    "permission_scope": "once",
}
```

CLI stats 可以只显示简短 footer，不要刷屏：

| 场景 | footer/status |
|---|---|
| pending | `Awaiting approval: edit file app.py` |
| approved | 正常 tool-use status |
| denied | `Tool denied by user` 短暂 toast |
| non-interactive deny | 不需要交互 UI，只在 final/result metadata 体现 |

## Function-call JSON 不闪屏回归

已有基础：

| 文件 | 现有能力 |
|---|---|
| `runtime/contracts.py` | `StreamSilentCallback` 统计非可见 tool-call JSON chunk |
| `cli/ui.py` | `strip_tool_blocks()` 清理 `<tool_call>` / `<function_call>` 风格文本 |
| `cli/app.py` | bottom region 统一托管 activity/status |

需要新增针对 permission 的回归：

| 场景 | 预期 |
|---|---|
| 模型流式输出 tool arguments | arguments 不进入可见 body |
| permission pending | request arguments 不进入 scrollback |
| file diff preview | 只在 overlay 当前帧展示，不追加进 transcript |
| rejected tool result | 不把原始 JSON args 打印到输出框 |

## 专项测试清单

推荐新增测试文件：

| 文件 | 覆盖 |
|---|---|
| `tests/runtime/test_tool_permissions.py` | rule matching、默认策略、非交互 deny |
| `tests/engine/test_tool_permission_gate.py` | dispatch 前 gate、synthetic result、session allow、after_tool |
| `tests/cli/test_permission_overlay.py` | overlay render、键盘选择、queue、no scrollback |
| `tests/cli/test_permission_prompter_bridge.py` | thread-safe future bridge、timeout、shutdown abort |
| `tests/tools/test_file_permission_preview.py` | file_edit/write_file preview 和拒绝不变更磁盘 |
| `tests/tools/test_shell_permission.py` | shell 拒绝不启动 subprocess、批准启动一次 |

## 端到端验收场景

### file_edit 拒绝

1. 让模型修改一个临时文件。
2. overlay 出现 `Edit file` 和 diff。
3. 按 `No`。
4. 文件内容保持不变。
5. 模型收到 tool result，说明用户拒绝，不会声称已修改。
6. 输出框没有出现 tool call JSON 或完整 request args。

### file_edit accept once

1. 让模型修改文件。
2. 选择 `Yes, allow this time`。
3. 文件被修改一次。
4. 第二次修改同文件仍然弹确认。

### file_edit accept session

1. 第一次修改同目录文件选择 session allow。
2. 第二次匹配 rule 的修改不再弹确认。
3. 不匹配 rule 的路径仍弹确认。

### write_file 拒绝

1. 让模型创建新文件。
2. overlay 展示 create diff。
3. 按 `No`。
4. 文件和 parent dir 都没有创建。

### shell 拒绝

1. 让模型执行 `touch should_not_exist` 或 mock subprocess。
2. overlay 出现 command block。
3. 按 `No`。
4. subprocess 没有启动，文件不存在。

### plan mode

1. 进入 plan mode。
2. 模型尝试 `file_edit`。
3. 直接收到 plan-mode block，不弹普通 permission。
4. 调用 `exit_plan_mode` 后走现有 plan approval。
5. 退出 plan mode 后写类工具再走普通 permission。

### streaming 中 function call JSON

1. 使用 fake provider 流式产出 tool call argument chunks。
2. UI token/silent counter 增长。
3. 可见输出框不显示 JSON chunks。
4. permission overlay 只展示 human-readable preview。

## 推荐命令

从 `backend/harness/aether` 执行：

```bash
python -m pytest tests/runtime/test_tool_permissions.py -v
python -m pytest tests/engine/test_tool_permission_gate.py -v
python -m pytest tests/cli/test_permission_overlay.py tests/cli/test_permission_prompter_bridge.py -v
python -m pytest tests/tools/test_file_permission_preview.py tests/tools/test_shell_permission.py -v
python -m pytest tests/cli tests/runtime tests/engine tests/tools
```

## 验收门

- 每个 PR 都有专项测试，不接受“后续补测”。
- 拒绝路径通过测试证明没有副作用。
- overlay 和 function-call JSON 闪屏问题有 CLI 层回归测试。
- 全量 CLI/runtime/engine/tools 测试通过。
- 文档中的手测场景至少跑一遍并记录结果。

