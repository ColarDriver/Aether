# Sprint 7 Tool Permission Confirmation - Acceptance Matrix

## PR 验收矩阵

| PR | 必须通过的专项测试 | 必须确认的回归点 |
|---|---|---|
| 7.1 Contracts + session rules | `test_tool_permissions.py` 覆盖 readonly allow、dangerous ask、non-interactive deny、session rule、deny override | runtime 契约不 import CLI；`ApprovalPrompter` 旧语义不变 |
| 7.2 Engine permission gate | `test_tool_permission_gate.py` 覆盖 allow once、reject no dispatch、synthetic result、after_tool、synthesized tool path | unknown tool、schema injection、tool error、middleware short-circuit 不回归 |
| 7.3 CLI permission overlay | `test_permission_overlay.py` 和 bridge 测试覆盖 render、queue、Enter、Esc、timeout、shutdown abort | 确认 UI 不进 scrollback；ESC 优先级链不误触发；输入框可重绘 |
| 7.4 File diff preview | `test_file_permission_preview.py` 覆盖 file_edit/write_file preview、reject no mutation、accept mutation、stale preview | 不调用 execute 做 preview；diff 截断；UTF-8 和大文件错误不回归 |
| 7.5 Shell + plan-mode policy | `test_shell_permission.py` 覆盖 shell reject no subprocess、accept once、session prefix、plan-mode precedence | shell timeout/cwd/spill/interrupt 不回归；plan mode 不能被 session allow 绕过 |
| 7.6 Tests + observability | CLI/engine E2E 覆盖 JSON 不闪屏、metadata、footer/status、non-interactive | `StreamSilentCallback` 语义不变；tool group 不把 denied tool 渲染成已执行 |

## 用户可见完成标准

| 场景 | 预期 |
|---|---|
| 模型请求编辑文件 | 输入框上方出现确认 overlay 和 diff |
| 用户拒绝编辑 | 文件不变，模型收到拒绝 tool result，输出框不打印确认 JSON |
| 用户单次批准 | 工具执行一次，下一次仍询问 |
| 用户 session 批准 | 匹配 session rule 的后续工具不再询问 |
| 模型请求 shell | 先显示 command 确认，批准前 subprocess 不启动 |
| plan mode 内写入 | 不弹普通 permission，直接 plan-mode block |
| function call streaming | tool argument JSON 不闪进可见输出框 |
| 非交互运行 | 危险工具 deterministic deny，进程不挂起等待输入 |

## 推荐测试命令

```bash
cd backend/harness/aether
python -m pytest tests/runtime/test_tool_permissions.py -v
python -m pytest tests/engine/test_tool_permission_gate.py -v
python -m pytest tests/cli/test_permission_overlay.py tests/cli/test_permission_prompter_bridge.py -v
python -m pytest tests/tools/test_file_permission_preview.py tests/tools/test_shell_permission.py -v
python -m pytest tests/cli tests/runtime tests/engine tests/tools
```

## 手测脚本

```text
1. 启动 Aether REPL。
2. 要求模型修改一个小文件。
3. 确认 overlay 出现在输入框上方，输出框没有 JSON。
4. 选择 No，检查文件未变化。
5. 再次要求同一修改，选择 Yes once，检查文件变化。
6. 再次要求另一处修改，确认仍会询问。
7. 选择 session allow，再请求同范围修改，确认不再询问。
8. 请求运行 shell touch 命令，选择 No，检查文件不存在。
9. 进入 plan mode 后请求写入，确认直接 plan-mode block。
```

## Sprint 完成定义

- 6 个 PR 全部实施并通过专项测试。
- 危险工具确认由 engine gate 统一控制，不由工具自行打印。
- CLI permission overlay 与 activity/status/input/footer 共存。
- 拒绝路径没有副作用，且模型能收到可理解的拒绝结果。
- open-claude-code 的 queue-backed confirmation 设计已按 Aether 架构落地。

