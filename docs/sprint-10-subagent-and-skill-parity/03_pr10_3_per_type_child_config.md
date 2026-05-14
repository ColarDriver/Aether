# PR 10.3 — Per-Type Child Configuration

## 目标 / Goal

让 `DefaultSubagentBuilder.build_child` 真正消费 PR 10.2 注入的 `AgentTypeDefinition`：

- **System prompt**：使用类型自带的 `system_prompt`（非空时）；否则继承父。
- **Tool registry**：根据类型的 `tools`（白名单）+ `disallowed_tools`（黑名单）过滤；保留一组核心元工具 (`task_stop`, `task_output`, `send_message`, `skill`) 不可被剔除。
- **Model**：`definition.model` 非空时覆盖子的 `ModelCallConfig.model`；`AgentTool` 同时支持调用方层面 `model` 参数透传（与 `open-claude-code` 一致）。
- **Max turns**：`definition.max_turns` 非空时覆盖 `child_config.max_iterations`。

参考：`open-claude-code/src/tools/AgentTool/AgentTool.tsx:318-356`（resolve definition）+ `:573-577`（workerTools assembly）+ `src/tools/AgentTool/runAgent.ts:248-329`（per-child config）。

## 当前问题 / Current Problem

`aether/subagents/default_builder.py:21-72` 的 `build_child` 完全不读 `task.metadata["_agent_type_def"]`：

```python
def build_child(
    self, parent: AgentEngine, task: SubagentTask, child_depth: int
) -> AgentEngine:
    provider = task.provider or parent.services.provider
    child_config = EngineConfig(...)            # 全部从 parent.config 复制
    child = AgentEngine(
        provider=provider,
        tool_registry=parent.services.tool_registry if self.inherit_tools else None,
        ...
    )
```

—— Tool registry 整套继承，system_message 由 `task.request.system_message` 直接传入（目前 `AgentTool` 不设），model 走 `provider` 不可换。

## 改动 / Changes

### 1. Tool 过滤 helper

新增 helper（可放 `aether/tools/registry.py` 末尾或新建 `aether/tools/filter.py`）：

```python
# aether/tools/filter.py
"""Filter a ToolRegistry by allow/deny lists.

Used by SubagentBuilder when the agent type definition restricts tools.
Always preserves a small set of "meta tools" that subagents need to
function (task_stop on themselves, task_output to read peer tasks,
send_message to communicate, skill to load on-demand playbooks).
"""
from __future__ import annotations
from typing import Iterable, Optional

from aether.tools.base import ToolExecutor
from aether.tools.registry import ToolRegistry


# Meta-tools that are always allowed regardless of type config.
# Rationale: a child agent must be able to (a) terminate itself on a stop
# signal (b) read its own / peer task output, (c) message peers, and (d)
# load skills on demand.  Removing any of these breaks the runtime
# contract for async subagents.
META_TOOLS_ALWAYS_ALLOWED: frozenset[str] = frozenset({
    "task_stop",
    "task_output",
    "send_message",
    "skill",
})


def filter_tool_registry(
    source: ToolRegistry,
    *,
    allow: Optional[Iterable[str]] = None,
    deny: Iterable[str] = (),
) -> ToolRegistry:
    """Return a new ToolRegistry with the same dispatch behavior as
    *source* but containing only the tools that pass allow/deny checks.

    Behavior:
    - allow=None  -> start from all of source's executors, remove deny.
    - allow=set   -> start from source ∩ allow, remove deny, then re-add META_TOOLS.
    """
    new = ToolRegistry()
    # Reuse source's dispatch gate config so plan-mode / etc. behavior
    # is preserved.  (ToolRegistry has internal state — copy or expose.)
    if hasattr(source, "copy_config_into"):
        source.copy_config_into(new)

    deny_set = {name for name in deny if name}
    if allow is None:
        keep = {d.name for d in source.list_descriptors()} - deny_set
    else:
        allow_set = {name for name in allow if name}
        keep = (allow_set & {d.name for d in source.list_descriptors()}) - deny_set

    keep |= META_TOOLS_ALWAYS_ALLOWED & {d.name for d in source.list_descriptors()}

    for descriptor in source.list_descriptors():
        if descriptor.name in keep:
            executor = source.get(descriptor.name)
            if executor is not None:
                new.register(executor)
    return new
```

If `ToolRegistry` doesn't have `copy_config_into` and `list_descriptors` yet,
add them as part of this PR:

```python
# in aether/tools/registry.py
def list_descriptors(self) -> list[ToolDescriptor]:
    return [exe.descriptor for exe in self._executors.values()]

def copy_config_into(self, other: "ToolRegistry") -> None:
    other._plan_mode = self._plan_mode  # or whatever fields exist
```

### 2. SubagentBuilder 消费类型定义

**改 `aether/subagents/default_builder.py`**：

```python
from aether.agents.types.definition import AgentTypeDefinition
from aether.tools.filter import filter_tool_registry


class DefaultSubagentBuilder(SubagentBuilder):
    def __init__(self, inherit_tools: bool = True, inherit_middlewares: bool = True) -> None:
        self.inherit_tools = inherit_tools
        self.inherit_middlewares = inherit_middlewares

    def build_child(
        self, parent: AgentEngine, task: SubagentTask, child_depth: int
    ) -> AgentEngine:
        provider = task.provider or parent.services.provider
        definition: AgentTypeDefinition | None = task.metadata.get("_agent_type_def")

        # ---------------- tool registry (filtered) ----------------
        if self.inherit_tools:
            base_registry = parent.services.tool_registry
            if definition is not None and (definition.tools is not None or definition.disallowed_tools):
                tool_registry = filter_tool_registry(
                    base_registry,
                    allow=definition.tools,
                    deny=definition.disallowed_tools,
                )
            else:
                tool_registry = base_registry
        else:
            tool_registry = None

        # ---------------- max_iterations ----------------
        max_iterations = task.max_iterations if task.max_iterations is not None else parent.config.max_iterations
        if definition is not None and definition.max_turns is not None:
            max_iterations = min(max_iterations, definition.max_turns)

        child_config = EngineConfig(
            max_iterations=max_iterations,
            # ... unchanged inherited fields ...
        )

        # ---------------- model override ----------------
        model_override = (
            task.metadata.get("model_override")
            or (definition.model if definition is not None else None)
        )
        if model_override and model_override != "inherit":
            task.request.model_config.model = _resolve_model_alias(model_override)

        # ---------------- system prompt ----------------
        if definition is not None and definition.system_prompt:
            # Set on the request so engine picks it up at _prepare_session_and_system_prompt.
            task.request.system_message = definition.system_prompt

        # ---------------- preloaded skills hint ----------------
        if definition is not None and definition.skills:
            existing = task.request.system_message or ""
            hint = _format_preloaded_skills_hint(definition.skills)
            task.request.system_message = existing + "\n\n" + hint if existing else hint

        child = AgentEngine(
            provider=provider,
            tool_registry=tool_registry,
            middleware_pipeline=parent.services.middleware_pipeline if self.inherit_middlewares else None,
            config=child_config,
            ...
        )
        # ... interrupt_signal handling unchanged ...
        return child


_MODEL_ALIAS_MAP = {
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",
    "haiku":  "claude-haiku-4-5-20251001",
}


def _resolve_model_alias(name: str) -> str:
    return _MODEL_ALIAS_MAP.get(name.lower(), name)


def _format_preloaded_skills_hint(skill_names: tuple[str, ...]) -> str:
    bullets = "\n".join(f"- {name}" for name in skill_names)
    return (
        "<system-reminder>\n"
        "You have been hand-picked for this task because the following "
        "skills are likely relevant.  Consider invoking them early via "
        "the `skill` tool:\n"
        f"{bullets}\n"
        "</system-reminder>"
    )
```

### 3. AgentTool 透传 model 参数

**改 `aether/tools/builtins/agent_tool.py`**：

`parameters.properties` 加：

```python
"model": {
    "type": "string",
    "enum": ["sonnet", "opus", "haiku", "inherit"],
    "default": "inherit",
    "description": (
        "Optional model override for this subagent.  Takes precedence "
        "over the agent type definition's model.  If omitted, uses "
        "the agent definition's model, or inherits from the parent."
    ),
},
```

`execute` 把 `args.get("model")` 写入 `task.metadata["model_override"]`。

### 4. 验证 tool_use_contract 用过滤后的列表

`aether/agents/core/system_prompt.py` 的 `augment_system_with_tool_contract(system, descriptors)` 已经接受 descriptors 参数 —— 确保 engine 调它时传的是**子 agent 自己的** registry 的 descriptors，而不是父的。具体在 `_prepare_session_and_system_prompt`：

```python
descriptors = self.services.tool_registry.list_descriptors()
active_system = augment_system_with_tool_contract(request.system_message, descriptors)
```

—— 如果当前实现已经这样，不用改；否则修正。

## 测试 / Tests

### Python

新建 `aether/tests/tools/test_filter.py`：
- `filter_tool_registry(reg, allow=None, deny=("write_file",))` → 不含 write_file，其它都在。
- `filter_tool_registry(reg, allow=("read_file",))` → 只含 read_file + META_TOOLS。
- `filter_tool_registry(reg, allow=("read_file",), deny=("skill",))` → 即便 deny 也保留 skill（META_TOOLS 优先级最高）。

新建 `aether/tests/subagents/test_default_builder_typed.py`：
- `Explore` definition → 子 registry 不含 `write_file` / `file_edit` / `shell`，含 `read_file` / `grep`。
- `Plan` definition → 子 registry 含 `write_file`（plan 允许写），不含 `shell` / `file_edit`。
- `system_prompt` 非空时，子 `request.system_message` 被设为 definition.system_prompt。
- `definition.skills=("tdd",)` → 子 system_message 末尾出现 `tdd` 的 hint 块。
- `definition.max_turns=5` 而父 `max_iterations=100` → 子 `max_iterations=5`。
- `model="haiku"` 别名解析为 `claude-haiku-4-5-20251001`。
- `_CORE_PROTECTED` 工具（实际是 `META_TOOLS_ALWAYS_ALLOWED`）永不被剔除。

新建 `aether/tests/subagents/test_subagent_type_e2e.py`（scripted provider）：
- 父用 `Explore` 跑一个 prompt；子 agent 的第一条 system 含 explore-specific 文本；子的 tool_use_contract 列表里没有 write 类工具名。
- 父用未知 type → AgentTool 返回 error，不 spawn 子。
- 同步路径下 `expected_output` 仍然附加进 result content。

### 验收 / Acceptance

- `uv run pytest aether/tests/tools/test_filter.py aether/tests/subagents/test_default_builder_typed.py aether/tests/subagents/test_subagent_type_e2e.py` 全绿。
- `uv run pyright` 无新告警。
- **手测**：跑 `uv run aether`，让模型 spawn `Explore` 子 agent 并尝试 `task(subagent_type="Explore", prompt="overwrite /tmp/x with hello")`。子 agent 应当回答"I cannot write files"或返回工具未知错误，而不是实际写入。

## 不在本 PR / Deferred

- **`mcpServers` per type**：等 MCP 引入。
- **`hooks` per type**：等 hook 机制成熟。
- **TUI 端的 type picker**：本 sprint 不做。
- **Per-type permission mode**：与 Sprint 7 权限合约耦合，留给后续。
