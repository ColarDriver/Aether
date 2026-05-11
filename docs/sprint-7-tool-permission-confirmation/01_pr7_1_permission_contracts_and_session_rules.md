# PR 7.1 - Permission Contracts + Session Rules

## 目标

建立通用工具权限契约和 session-scoped rule store。这个 PR 只做纯 runtime 数据结构和策略判断，不接 UI，不执行工具，不改 CLI layout。

## 当前问题

`EngineRequest.approval_prompter` 目前是 `Any`，实际只给 `ExitPlanModeTool` 和 `AskUserQuestionTool` 使用。它表达不了“这个 tool call 是否需要确认”、“用户选择 accept once 还是 accept session”、“非交互模式默认拒绝”这些语义。

同时，`tools/registry.py` 的 `WRITE_TOOLS_BLOCKED_IN_PLAN` 已经定义了写类工具集合，但这只是 plan mode blocker，不是通用 permission policy。继续把逻辑塞进 `ApprovalPrompter.confirm_plan()` 会导致权限系统和交互工具耦合。

## 新增契约

建议新增文件：

| 文件 | 内容 |
|---|---|
| `backend/harness/aether/runtime/tool_permissions.py` | permission dataclass、enum、rule matcher、default policy |
| `backend/harness/aether/tests/runtime/test_tool_permissions.py` | 纯策略专项测试 |

核心类型：

```python
class ToolPermissionMode(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"

class ToolPermissionDecisionType(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"
    ABORT = "abort"

@dataclass(slots=True, frozen=True)
class ToolPermissionRequest:
    session_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    category: str
    risk: str
    preview: ToolPermissionPreview | None = None
    reason: str | None = None

@dataclass(slots=True, frozen=True)
class ToolPermissionDecision:
    type: ToolPermissionDecisionType
    updated_arguments: dict[str, Any] | None = None
    feedback: str | None = None
    rule: ToolPermissionRule | None = None
```

Preview 契约：

```python
@dataclass(slots=True, frozen=True)
class ToolPermissionPreview:
    title: str
    subtitle: str | None = None
    body: str | None = None
    diff: str | None = None
    path: str | None = None
    command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Prompter protocol：

```python
class ToolPermissionPrompter(Protocol):
    def is_interactive(self) -> bool: ...
    def request_tool_permission(
        self,
        request: ToolPermissionRequest,
        *,
        timeout: float | None = None,
    ) -> ToolPermissionDecision: ...
```

`EngineRequest` 扩展：

```python
tool_permission_prompter: Any = None
```

保留 `approval_prompter`，但不要复用同一字段。`approval_prompter` 面向交互工具，`tool_permission_prompter` 面向 engine dispatch gate。

## Session Rule Store

规则先放在 `TurnContext.metadata` 或 engine session runtime 中，生命周期为当前 session。

建议结构：

```python
@dataclass(slots=True, frozen=True)
class ToolPermissionRule:
    tool_name: str
    behavior: ToolPermissionMode
    scope: str = "session"
    path_prefix: str | None = None
    command_prefix: str | None = None
    reason: str | None = None
```

匹配顺序：

| 顺序 | 规则 |
|---|---|
| 1 | 显式 deny rule 优先 |
| 2 | 精确 tool + path/command 规则 |
| 3 | 精确 tool session allow |
| 4 | readonly tool 默认 allow |
| 5 | write/shell/subagent side-effect tool 默认 ask |
| 6 | 非交互模式危险工具默认 deny |

首版危险工具集合可复用并扩展：

```python
DANGEROUS_TOOLS = {
    "shell",
    "write_file",
    "file_edit",
    "notebook_edit",
    "todo_write",
    "task",
    "task_stop",
}
```

## 配置项

建议扩展 `EngineConfig`：

| 字段 | 默认 | 说明 |
|---|---|---|
| `tool_permissions_enabled` | `False` | engine 默认保持 SDK/测试兼容；交互 REPL 显式开启 |
| `tool_permission_default` | `"ask"` | 危险工具默认策略 |
| `tool_permission_auto_allow_readonly` | `True` | read/list/search 默认放行 |
| `tool_permission_non_interactive_default` | `"deny"` | 无 prompter 时危险工具默认拒绝 |
| `tool_permission_session_allow_enabled` | `True` | 是否允许用户选择 session allow |

## 测试

新增 `tests/runtime/test_tool_permissions.py`：

```python
def test_readonly_tool_allowed_by_default(): ...
def test_write_tool_asks_by_default(): ...
def test_non_interactive_write_tool_denied_by_default(): ...
def test_session_allow_rule_matches_same_tool_and_path_prefix(): ...
def test_deny_rule_overrides_session_allow(): ...
def test_unknown_tool_defaults_to_ask_not_allow(): ...
def test_disabled_permission_system_allows_backwards_compat(): ...
```

## 验收门

- 新契约不 import `prompt_toolkit` 或任何 CLI 模块。
- 现有 `ApprovalPrompter` 测试不需要改语义。
- 非交互默认拒绝危险工具的行为被 pin 住。
- `WRITE_TOOLS_BLOCKED_IN_PLAN` 不被删除，后续 PR 统一接入。
