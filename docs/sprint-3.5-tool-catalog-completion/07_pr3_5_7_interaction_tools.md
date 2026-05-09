# PR 3.5.7 — `EnterPlanMode` + `ExitPlanMode` + `AskUserQuestion`

> **角色**：把 Cursor / claude-code 的 "Plan / Agent / Ask" 三模态交互能力引入 Aether。
> 这三个工具都需要 CLI 层支持（不是纯计算工具）。

## 一、目标

1. 实现 **mode-state 基础设施**（`EngineConfig` + `TurnContext`）：把当前会话模式
   （`agent` / `plan` / `ask`）作为一等公民暴露给工具。
2. `EnterPlanModeTool`：模型主动声明"我要进入计划阶段"，写工具被禁用。
3. `ExitPlanModeTool`：呈现计划摘要给用户审批，用户确认后回到 agent 模式。
4. `AskUserQuestionTool`：在 CLI 中**结构化提问**用户（不是泛泛的 stdout 输出）。

## 二、为什么要做

### 2.1 现状

Aether 没有 mode 的概念。每个 turn 模型可以自由调用 read 也可以调用 write，
没有"先思考再动手"的强制隔离。

claude-code 的 plan-mode 实测显著降低 over-eager 修改：模型必须显式调用
`EnterPlanMode` 后才能进入"只读探索"，到 `ExitPlanMode` 时把方案给用户审批，
用户可以在动笔前否决。

### 2.2 AskUserQuestion 的价值

模型在 CLI 中往往直接 finalize "我做了 A, B, C，你觉得呢？" — 用户回答后再调下个 turn。
这种隐式 ping-pong 浪费 token。

`AskUserQuestion` 把"中途询问用户"形式化：
* 工具调用 → CLI 弹出结构化 prompt（多选 / 文本输入）
* 用户回答 → ToolResult 返回模型继续推理
* **同一个 turn 内**完成，不重复发送 prompt

## 三、设计

### 3.1 共享基础设施 — Mode 状态

`runtime/contracts.py` 加：

```python
from enum import Enum

class SessionMode(str, Enum):
    AGENT = "agent"
    PLAN = "plan"
    # ASK 不是一个 mode（仅是工具）；保留 enum 以备 v2

# TurnContext 不加字段（保持 dataclass 形状）
# 模式存放在 context.metadata["session_mode"]，默认 "agent"
```

`EngineConfig`：

```python
# Sprint 3.5 / PR 3.5.7
plan_mode_enabled: bool = True  # 总开关
ask_user_question_enabled: bool = True  # 总开关
ask_user_question_timeout_seconds: int = 600  # 用户 10 分钟不回答自动 cancel
```

`SessionStore`（轻量，模块级 dict 复用 PR 3.5.3 的 todo store 模式）：

```python
# runtime/session_state.py - 新文件
_SESSION_MODE: dict[str, str] = {}

def get_mode(session_id: str) -> str:
    return _SESSION_MODE.get(session_id, "agent")

def set_mode(session_id: str, mode: str) -> None:
    _SESSION_MODE[session_id] = mode
```

### 3.2 `EnterPlanModeTool`

#### 3.2.1 输入 schema

```python
{"type": "object", "properties": {}}  # 无参数
```

#### 3.2.2 实现

```python
class EnterPlanModeTool(ToolExecutor):
    NAME = "enter_plan_mode"

    def execute(self, call, context):
        # subagent 不能切模式
        parent_agent = context.metadata.get("_parent_agent")
        if parent_agent and getattr(parent_agent, "delegate_depth", 0) > 0:
            return ToolResult(
                content="EnterPlanMode is not allowed inside subagent contexts",
                is_error=True,
            )

        config = context.metadata.get("_engine_config")
        if not getattr(config, "plan_mode_enabled", True):
            return ToolResult(
                content="plan mode is disabled by configuration",
                is_error=True,
            )

        from aether.runtime.session_state import set_mode
        set_mode(context.session_id, "plan")

        msg = (
            "Entered plan mode. While in plan mode:\n"
            "1. Thoroughly explore the codebase to understand existing patterns.\n"
            "2. Use read_file / grep / glob freely; write_file / file_edit / shell are blocked.\n"
            "3. Use AskUserQuestion to clarify requirements.\n"
            "4. When ready, call ExitPlanMode with your concrete plan for user approval.\n"
            "DO NOT modify any files until ExitPlanMode is approved."
        )
        return ToolResult(call_id=call.id, content=msg)
```

#### 3.2.3 mode 强制

`tool_dispatch.py` 在执行任何工具前检查 mode：

```python
WRITE_TOOLS = frozenset({
    "shell", "write_file", "file_edit", "notebook_edit",
})

def _check_mode_allows_tool(tool_name: str, mode: str) -> str | None:
    if mode == "plan" and tool_name in WRITE_TOOLS:
        return (
            f"tool {tool_name!r} is blocked in plan mode. "
            "Use ExitPlanMode first to present your plan and get approval."
        )
    return None
```

被 block 时工具返回 `is_error=True` 让模型自知（不抛异常）。

### 3.3 `ExitPlanModeTool`

#### 3.3.1 输入 schema

```python
{
    "type": "object",
    "properties": {
        "plan": {
            "type": "string",
            "description": (
                "The complete plan to execute. Markdown formatted. "
                "Will be shown to the user for approval before execution begins."
            ),
        },
    },
    "required": ["plan"],
}
```

#### 3.3.2 实现

```python
class ExitPlanModeTool(ToolExecutor):
    NAME = "exit_plan_mode"

    def __init__(self, prompter):
        self.prompter = prompter  # CLI 注入的 ApprovalPrompter

    def execute(self, call, context):
        from aether.runtime.session_state import get_mode, set_mode

        if get_mode(context.session_id) != "plan":
            return ToolResult(
                content="ExitPlanMode called but session is not in plan mode",
                is_error=True,
            )

        plan = call.arguments["plan"]

        # 调 CLI prompter（CLI 注入；测试时 mock）
        approved = self.prompter.confirm_plan(plan, context=context)
        if approved:
            set_mode(context.session_id, "agent")
            return ToolResult(
                call_id=call.id,
                content=(
                    "Plan approved. Returning to agent mode — proceed with implementation. "
                    "Stay focused on the steps you outlined."
                ),
            )
        return ToolResult(
            call_id=call.id,
            content=(
                "User did not approve the plan. Stay in plan mode and revise. "
                "Consider asking clarifying questions with AskUserQuestion."
            ),
        )
```

#### 3.3.3 CLI 注入

CLI 启动时构造 `ApprovalPrompter` 实例，挂到 `EngineRequest`：

```python
# cli/repl.py
from aether.cli.approval_prompter import ApprovalPrompter
prompter = ApprovalPrompter(stdin=sys.stdin, stdout=sys.stdout)
request = EngineRequest(..., approval_prompter=prompter)
```

`agent.py::_prepare_turn_entry`：

```python
context.metadata["_approval_prompter"] = request.approval_prompter
```

`ExitPlanModeTool` 从 `context.metadata` 拿。

### 3.4 `AskUserQuestionTool`

#### 3.4.1 输入 schema

```python
{
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                            },
                            "required": ["id", "label"],
                        },
                    },
                    "allow_multiple": {"type": "boolean", "default": False},
                    "free_text": {"type": "boolean", "default": False},
                },
                "required": ["id", "prompt"],
            },
        },
    },
    "required": ["questions"],
}
```

#### 3.4.2 实现

```python
class AskUserQuestionTool(ToolExecutor):
    NAME = "ask_user_question"

    def __init__(self, prompter, *, timeout_seconds=600):
        self.prompter = prompter
        self.timeout_seconds = timeout_seconds

    def execute(self, call, context):
        # subagent 不能问用户
        parent_agent = context.metadata.get("_parent_agent")
        if parent_agent and getattr(parent_agent, "delegate_depth", 0) > 0:
            return ToolResult(
                content=(
                    "AskUserQuestion is not allowed in subagent contexts. "
                    "Either return your uncertainty in the subagent summary, "
                    "or have the parent agent ask the user."
                ),
                is_error=True,
            )

        # 非交互模式（stdin 不是 TTY）
        prompter = context.metadata.get("_approval_prompter") or self.prompter
        if not prompter.is_interactive():
            return ToolResult(
                content=(
                    "AskUserQuestion unavailable: running in non-interactive mode. "
                    "Make a best-effort decision and continue."
                ),
                is_error=True,
            )

        questions = call.arguments["questions"]
        try:
            answers = prompter.ask_questions(questions, timeout=self.timeout_seconds)
        except TimeoutError:
            return ToolResult(
                content=f"User did not respond within {self.timeout_seconds}s",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(content=f"prompt failed: {exc}", is_error=True)

        formatted = self._format_answers(questions, answers)
        return ToolResult(call_id=call.id, content=formatted)

    def _format_answers(self, questions, answers):
        lines = ["# User responses\n"]
        for q in questions:
            qid = q["id"]
            ans = answers.get(qid, "(no response)")
            lines.append(f"- {q['prompt']}\n  → {ans}\n")
        return "\n".join(lines)
```

### 3.5 `ApprovalPrompter` CLI 实现

```python
# cli/approval_prompter.py - 新文件
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import radiolist_dialog, checkboxlist_dialog

class ApprovalPrompter:
    def __init__(self, stdin=None, stdout=None):
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout

    def is_interactive(self) -> bool:
        return self.stdin.isatty() and self.stdout.isatty()

    def confirm_plan(self, plan: str, *, context) -> bool:
        # 渲染 plan 到 stdout，再问 yes/no
        self.stdout.write("\n" + "─" * 60 + "\n")
        self.stdout.write("Proposed plan:\n\n" + plan + "\n")
        self.stdout.write("─" * 60 + "\n")
        ans = prompt("Approve? [y/N]: ", default="N").strip().lower()
        return ans in {"y", "yes"}

    def ask_questions(self, questions, *, timeout):
        answers = {}
        for q in questions:
            options = q.get("options", [])
            allow_multiple = q.get("allow_multiple", False)
            if options and not allow_multiple:
                # radiolist
                values = [(o["id"], o["label"]) for o in options]
                result = radiolist_dialog(title=q["prompt"], values=values).run()
                answers[q["id"]] = result
            elif options and allow_multiple:
                values = [(o["id"], o["label"]) for o in options]
                result = checkboxlist_dialog(title=q["prompt"], values=values).run()
                answers[q["id"]] = result
            else:
                # free text
                answers[q["id"]] = prompt(f"{q['prompt']}\n> ").strip()
        return answers
```

## 四、文件改动清单

| 文件 | 类型 | 行数 |
|---|---|---|
| `backend/harness/aether/runtime/session_state.py` | **新文件** | ~40 |
| `backend/harness/aether/cli/approval_prompter.py` | **新文件** | ~120 |
| `backend/harness/aether/tools/builtins/enter_plan_mode.py` | **新文件** | ~80 |
| `backend/harness/aether/tools/builtins/exit_plan_mode.py` | **新文件** | ~80 |
| `backend/harness/aether/tools/builtins/ask_user_question.py` | **新文件** | ~150 |
| `backend/harness/aether/tools/builtins/__init__.py` | 修改 | ~10 |
| `backend/harness/aether/agents/core/tool_dispatch.py` | 修改 | mode 检查 | ~30 |
| `backend/harness/aether/agents/core/agent.py` | 修改 | 注入 prompter / mode 到 metadata | ~10 |
| `backend/harness/aether/cli/repl.py` | 修改 | 构造 prompter | ~5 |
| `backend/harness/aether/runtime/contracts.py` | 修改 | `EngineRequest.approval_prompter` 字段 | ~5 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 mode + ask 配置 | ~15 |
| `backend/harness/aether/tests/test_session_state.py` | **新文件** | ~50 |
| `backend/harness/aether/tests/test_plan_mode.py` | **新文件** | ~250 |
| `backend/harness/aether/tests/test_ask_user_question.py` | **新文件** | ~250 |

## 五、测试用例

### 5.1 测试组 A：session_state

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | get_mode 默认 | 返回 "agent" |
| **T-A2** | set_mode 后 get_mode | 返回设置值 |
| **T-A3** | 跨 session 隔离 | A 设 plan 不影响 B |

### 5.2 测试组 B：EnterPlanMode

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | 调用后 mode 切换为 plan | get_mode == "plan" |
| **T-B2** | subagent 调用 | `is_error=True`；mode 不变 |
| **T-B3** | plan_mode_enabled=False | `is_error=True` |
| **T-B4** | content 含完整指引 | 4 步指令存在 |

### 5.3 测试组 C：write tools 在 plan mode 被拦截

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | mode=plan + 调 shell | tool_dispatch 拦截；返回 is_error=True |
| **T-C2** | mode=plan + 调 write_file | 同上 |
| **T-C3** | mode=plan + 调 file_edit | 同上 |
| **T-C4** | mode=plan + 调 notebook_edit | 同上 |
| **T-C5** | mode=plan + 调 read_file | 通过 |
| **T-C6** | mode=plan + 调 grep | 通过 |
| **T-C7** | mode=agent + 调 shell | 通过 |

### 5.4 测试组 D：ExitPlanMode

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | mode=plan + prompter 确认 | mode 切回 agent；content 含 "approved" |
| **T-D2** | mode=plan + prompter 拒绝 | mode 仍为 plan；content 含 "did not approve" |
| **T-D3** | mode=agent 调 ExitPlanMode | `is_error=True` |
| **T-D4** | prompter 抛异常 | `is_error=True` |

### 5.5 测试组 E：AskUserQuestion

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | mock prompter 返回答案 | content 含 "User responses" + 答案 |
| **T-E2** | subagent 调用 | `is_error=True` |
| **T-E3** | prompter.is_interactive() == False | `is_error=True` |
| **T-E4** | prompter 抛 TimeoutError | `is_error=True`；含 timeout 秒数 |
| **T-E5** | 单选 question | answers[qid] == 选中的 id |
| **T-E6** | 多选 question | answers[qid] == 选中的 id 列表 |
| **T-E7** | free_text question | answers[qid] == 用户输入 |
| **T-E8** | 多个 question 串联 | 每个 qid 都有答案 |

### 5.6 测试组 F：ApprovalPrompter（手动 / pytest-mock）

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | is_interactive 检测 stdin TTY | 在 pytest 中通常 False |
| **T-F2** | confirm_plan 默认 N | 输入空时拒绝 |

## 六、验收门

* [ ] 35+ case 全绿
* [ ] 真实跑（手动 in CLI）：
  * 模型 EnterPlanMode → 尝试 write_file → 被拒 → 用 read_file 探索 → ExitPlanMode → 用户批准 → 模型可写
  * 模型 AskUserQuestion 弹 prompt → 用户选项 → 模型继续

## 七、回滚开关

* `plan_mode_enabled=False` / `ask_user_question_enabled=False` 单工具关闭
* CLI 不传 prompter，工具自动 graceful 降级到 "non-interactive" 错误

## 八、实施顺序（建议 2 天）

| 步骤 | 时长 |
|---|---|
| 1. session_state + 测试 | 1h |
| 2. ApprovalPrompter（CLI 集成） | 2h |
| 3. EnterPlanMode + 测试 | 2h |
| 4. ExitPlanMode + 测试 | 2h |
| 5. AskUserQuestion + 测试 | 3h |
| 6. tool_dispatch mode 检查 + 测试 | 2h |
| 7. CLI repl 注入 + 手动 smoke | 2h |
| 8. 回归 | 1h |

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| 模型滥用 AskUserQuestion 频繁打断用户 | description 提示"只在确实需要时使用"；用户烦了可以禁用 |
| plan_mode 模型忘了 ExitPlanMode 卡死 | PR 3.2 IterationBudget 自然兜底；用户 Ctrl-C 也行 |
| prompt_toolkit 在某些 terminal 不工作 | fallback 到 plain `input()`；is_interactive 检查 |
| mode 状态在进程崩溃后丢失 | 模块级 dict；下次启动默认 agent；可接受 |
| 测试很难模拟 TTY | pytest-mock + monkey-patch is_interactive |

## 十、与后续 PR 的接合

* **CLI footer**：可显示当前 mode（"📋 Plan mode" / "🤖 Agent mode"）
* **Sprint 4 Tier 5 (autocompact)**：autocompact 期间禁止 mode 切换（避免压缩跨 mode 状态）
