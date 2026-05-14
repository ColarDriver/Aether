# PR 10.2 — Agent Type Registry

## 目标 / Goal

建立 **`AgentTypeRegistry`**：与 `SkillCatalog` 并列的注册表，负责：

1. 提供内置 agent 类型（`general-purpose` / `Explore` / `Plan` / `VerificationAgent`），与 `open-claude-code/src/tools/AgentTool/builtInAgents.ts:22-72` 对齐。
2. 加载项目级 `.claude/agents/*.md` 与用户级 `~/.aether/agents/*.md` 自定义类型（frontmatter + body）。
3. 让 `AgentTool` 的 `subagent_type` 参数从"自由字符串"升级为"必须 resolve 到注册表"的 enum，并在 tool descriptor 中动态暴露可用类型列表给模型。

本 PR **只建注册表 + 让 AgentTool 校验类型**；让定义真正影响子 agent 行为（system prompt、tool 过滤、model override）放在 PR 10.3。

参考：`open-claude-code/src/tools/AgentTool/loadAgentsDir.ts:106-133`（schema）+ `:296-393`（loader）+ `:541-600`（markdown parser）。

## 当前问题 / Current Problem

### 1. `subagent_type` 是写而不读的字符串

`aether/tools/builtins/agent_tool.py:104`：

```python
subagent_type = str(args.get("subagent_type") or "general-purpose").strip() or "general-purpose"
```

`:142,153,184` 把它写进 metadata；之后**再无任何代码读取**。

### 2. `task` 工具 schema 无 enum

`aether/tools/builtins/agent_tool.py:66-73`：

```python
"subagent_type": {
    "type": "string",
    "default": "general-purpose",
    "description": ("Which subagent persona to use. Defaults to 'general-purpose'."),
},
```

—— 没有 `enum` 字段；模型不知道选什么，只能盲填 `"general-purpose"`。

### 3. 没有 `.claude/agents/` 加载机制

`grep -r "\.claude/agents" /workspace/Aether/aether/` 零命中。

## 改动 / Changes

### 1. 新目录 `aether/agents/types/`

5 个文件：

#### `aether/agents/types/__init__.py`

```python
"""Agent type registry — parallel to ``SkillCatalog``.

Exports:
- :class:`AgentTypeDefinition`        — frozen dataclass describing a type
- :class:`AgentTypeRegistry`          — discovery / lookup over builtins + .md files
- ``BUILTIN_AGENT_TYPES``             — sequence of the four bundled types
- :func:`load_markdown_agent_type`    — parse a single .md file
"""
from aether.agents.types.definition import AgentTypeDefinition
from aether.agents.types.builtin import BUILTIN_AGENT_TYPES
from aether.agents.types.registry import AgentTypeRegistry
from aether.agents.types.markdown_loader import load_markdown_agent_type

__all__ = [
    "AgentTypeDefinition",
    "AgentTypeRegistry",
    "BUILTIN_AGENT_TYPES",
    "load_markdown_agent_type",
]
```

#### `aether/agents/types/definition.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True, frozen=True)
class AgentTypeDefinition:
    """Frozen description of a subagent persona.

    Mirrors `open-claude-code/src/tools/AgentTool/loadAgentsDir.ts:106-133`
    (`BaseAgentDefinition`).  Source-of-truth for what the child agent
    looks like at spawn time.
    """

    agent_type: str
    description: str                          # "when to use" — shown to model
    system_prompt: str = ""                   # child's system_message (no tool_contract; engine adds that)
    tools: Optional[tuple[str, ...]] = None   # allowlist; None = inherit all (minus disallowed)
    disallowed_tools: tuple[str, ...] = ()    # denylist
    model: Optional[str] = None               # e.g. "claude-sonnet-4-6" or alias "sonnet"
    skills: tuple[str, ...] = ()              # preload these skills in child's system reminder
    max_turns: Optional[int] = None           # overrides child config.max_iterations
    isolation: Optional[str] = None           # "worktree" / None
    background: bool = False                  # force async even without run_in_background
    source: str = "builtin"                   # "builtin" | "project" | "user" | "plugin"
    source_path: Optional[Path] = None        # markdown origin (None for builtins)

    def to_snapshot(self) -> dict:
        """Serializable copy for TaskStore freezing."""
        return {
            "agent_type": self.agent_type,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": list(self.tools) if self.tools is not None else None,
            "disallowed_tools": list(self.disallowed_tools),
            "model": self.model,
            "skills": list(self.skills),
            "max_turns": self.max_turns,
            "isolation": self.isolation,
            "background": self.background,
            "source": self.source,
            "source_path": str(self.source_path) if self.source_path else None,
        }
```

#### `aether/agents/types/builtin.py`

```python
"""Four built-in agent types parallel to
`open-claude-code/src/tools/AgentTool/builtInAgents.ts:22-72`.
"""
from __future__ import annotations
from aether.agents.types.definition import AgentTypeDefinition


GENERAL_PURPOSE = AgentTypeDefinition(
    agent_type="general-purpose",
    description=(
        "General-purpose agent for researching complex questions, "
        "searching for code, and executing multi-step tasks. Use when "
        "the task spans the codebase and you need investigation + edits."
    ),
    # Empty system_prompt => child inherits parent's; this is the "default" type.
    system_prompt="",
    tools=None,           # inherit all
    disallowed_tools=(),
    model=None,
    source="builtin",
)


EXPLORE = AgentTypeDefinition(
    agent_type="Explore",
    description=(
        "Fast read-only search agent for locating code. Use it to find "
        "files by pattern, grep for symbols, or answer 'where is X "
        "defined / which files reference Y.'  Cannot edit, run shell, "
        "or write — it reports findings to the parent."
    ),
    system_prompt=(
        "You are a focused read-only exploration agent.  Your job is "
        "to locate code, gather context, and summarize findings for "
        "the parent agent.  You do NOT have write or shell tools.  "
        "Be aggressive about searching multiple locations / naming "
        "conventions, but report concisely: file paths + 1-2 line "
        "summaries beat full file dumps."
    ),
    tools=("read_file", "list_dir", "grep", "glob", "web_fetch", "web_search", "skill"),
    disallowed_tools=(),
    model=None,
    source="builtin",
)


PLAN = AgentTypeDefinition(
    agent_type="Plan",
    description=(
        "Software architect agent for designing implementation plans.  "
        "Use when you need a step-by-step plan, file impact analysis, "
        "or architectural trade-offs.  Read-only + writes to plan file only."
    ),
    system_prompt=(
        "You are an implementation planner.  Read the codebase, identify "
        "critical files, and produce a concise plan.  Output:\n"
        "1. Brief context (1-2 sentences)\n"
        "2. Numbered steps with file paths and line numbers\n"
        "3. Risks / open questions\n\n"
        "You may NOT edit code; you may write to the designated plan "
        "file via `write_file` only inside the plans directory."
    ),
    tools=("read_file", "list_dir", "grep", "glob", "web_fetch", "web_search", "write_file", "skill"),
    disallowed_tools=(),
    model=None,
    source="builtin",
)


VERIFICATION = AgentTypeDefinition(
    agent_type="VerificationAgent",
    description=(
        "Validates that an implemented feature actually works.  Reads "
        "code, runs tests/builds, reports pass/fail with evidence."
    ),
    system_prompt=(
        "You verify implementations end-to-end.  Read the relevant code, "
        "run tests and builds via `shell`, and report explicit pass/fail "
        "with the exact commands and outputs you observed.  Do NOT edit "
        "code; if a test is wrong, flag it for the parent."
    ),
    tools=("read_file", "list_dir", "grep", "glob", "shell", "skill"),
    disallowed_tools=(),
    model=None,
    source="builtin",
)


BUILTIN_AGENT_TYPES: tuple[AgentTypeDefinition, ...] = (
    GENERAL_PURPOSE,
    EXPLORE,
    PLAN,
    VERIFICATION,
)
```

#### `aether/agents/types/markdown_loader.py`

```python
"""Parse a `.claude/agents/<name>.md` file into an AgentTypeDefinition.

Frontmatter schema:
---
name: code-reviewer                     # required (or filename stem)
description: Reviews PRs ...            # required
tools: [read_file, grep, web_fetch]     # optional; JSON array or comma list
disallowed_tools: [write_file]          # optional
model: sonnet                           # optional
skills: [security-review, tdd]          # optional
max_turns: 30                           # optional
isolation: worktree                     # optional
background: false                       # optional
---
<system prompt body>

Reuses the lightweight frontmatter parser from `SkillCatalog`
(no PyYAML dependency).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from aether.agents.types.definition import AgentTypeDefinition
from aether.runtime.tools.skill_catalog import SkillCatalog


def load_markdown_agent_type(
    path: Path,
    *,
    source: str = "project",
) -> Optional[AgentTypeDefinition]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    front, body = SkillCatalog._split_frontmatter(text)  # noqa: SLF001 — internal reuse

    name = (front.get("name") or path.stem).strip()
    if not name:
        return None
    description = front.get("description", "").strip()
    if not description:
        return None

    return AgentTypeDefinition(
        agent_type=name,
        description=description,
        system_prompt=body.strip(),
        tools=_parse_list_field(front.get("tools")) or None,
        disallowed_tools=tuple(_parse_list_field(front.get("disallowed_tools")) or ()),
        model=(front.get("model") or "").strip() or None,
        skills=tuple(_parse_list_field(front.get("skills")) or ()),
        max_turns=_parse_int(front.get("max_turns")),
        isolation=(front.get("isolation") or "").strip() or None,
        background=_parse_bool(front.get("background")),
        source=source,
        source_path=path,
    )


def _parse_list_field(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not raw.strip():
        return None
    txt = raw.strip()
    if txt.startswith("[") and txt.endswith("]"):
        inner = txt[1:-1]
    else:
        inner = txt
    items = [s.strip().strip('"').strip("'") for s in inner.split(",")]
    return tuple(item for item in items if item)


def _parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _parse_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in ("true", "yes", "1", "on")
```

#### `aether/agents/types/registry.py`

```python
"""AgentTypeRegistry — discovery over builtins + .md files."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable, Optional

from aether.agents.types.builtin import BUILTIN_AGENT_TYPES
from aether.agents.types.definition import AgentTypeDefinition
from aether.agents.types.markdown_loader import load_markdown_agent_type

logger = logging.getLogger(__name__)


class AgentTypeRegistry:
    """Resolve agent type names to AgentTypeDefinition.

    Discovery order (last wins on name collision):
    1. Builtins (always present)
    2. User dir (e.g. ~/.aether/agents/*.md)
    3. Project dir (e.g. <cwd>/.claude/agents/*.md)
    """

    def __init__(self, *, search_paths: Iterable[Path]) -> None:
        self.search_paths: list[Path] = [Path(p) for p in search_paths]
        self._types: dict[str, AgentTypeDefinition] = {}
        self._loaded = False

    def discover(self, *, force: bool = False) -> None:
        if self._loaded and not force:
            return
        self._types.clear()
        for builtin in BUILTIN_AGENT_TYPES:
            self._types[builtin.agent_type] = builtin
        for root in self.search_paths:
            if not root.exists() or not root.is_dir():
                continue
            for md in sorted(root.glob("*.md")):
                source = self._infer_source(root)
                definition = load_markdown_agent_type(md, source=source)
                if definition is not None:
                    self._types[definition.agent_type] = definition
                else:
                    logger.warning("skipping malformed agent type at %s", md)
        self._loaded = True

    def get(self, name: str) -> Optional[AgentTypeDefinition]:
        self.discover()
        if not name:
            return None
        if name in self._types:
            return self._types[name]
        lowered = name.lower()
        for k, v in self._types.items():
            if k.lower() == lowered:
                return v
        return None

    def list_all(self) -> list[AgentTypeDefinition]:
        self.discover()
        return list(self._types.values())

    @staticmethod
    def _infer_source(root: Path) -> str:
        parts = root.parts
        if ".aether" in parts:
            return "user"
        if ".claude" in parts:
            return "project"
        return "external"
```

### 2. Wire 进 AgentTool

**改 `aether/tools/builtins/agent_tool.py`**：

#### a. 构造器接收注册表

```python
def __init__(
    self,
    *,
    parent_agent: Any | None = None,
    subagent_manager: Any | None = None,
    agent_type_registry: "AgentTypeRegistry | None" = None,
) -> None:
    self._parent_agent = parent_agent
    self._subagent_manager = subagent_manager
    self._agent_type_registry = agent_type_registry
    # descriptor 仍然延迟构造（见下）
```

#### b. Descriptor 动态枚举

把 `self._descriptor = ToolDescriptor(...)` 改成属性 + 延迟构造：

```python
@property
def descriptor(self) -> ToolDescriptor:
    if self._descriptor is None or self._descriptor_stale:
        self._descriptor = self._build_descriptor()
        self._descriptor_stale = False
    return self._descriptor

def _build_descriptor(self) -> ToolDescriptor:
    registry = self._agent_type_registry
    if registry is not None:
        types = registry.list_all()
        type_names = sorted({t.agent_type for t in types})
        type_desc_lines = "\n".join(
            f"  - {t.agent_type}: {t.description}" for t in sorted(types, key=lambda x: x.agent_type)
        )
        subagent_type_field: dict[str, Any] = {
            "type": "string",
            "default": "general-purpose",
            "enum": type_names,
            "description": (
                "Which subagent persona to use.  Available:\n"
                + type_desc_lines
            ),
        }
    else:
        subagent_type_field = {
            "type": "string",
            "default": "general-purpose",
            "description": "Which subagent persona to use. Defaults to 'general-purpose'.",
        }

    return ToolDescriptor(
        name=self.NAME,
        description=(...),  # unchanged
        parameters={
            "type": "object",
            "properties": {
                "subagent_type": subagent_type_field,
                "prompt": {...},
                "expected_output": {...},
            },
            "required": ["prompt"],
        },
        required=["prompt"],
    )
```

Note: `tool_use_contract` augmentation is recomputed each turn, so dynamic
description changes propagate automatically.

#### c. `execute` 解析类型

```python
def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
    args = call.arguments or {}
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _error(call, "'prompt' is required and must be a non-empty string")

    subagent_type = str(args.get("subagent_type") or "general-purpose").strip() or "general-purpose"

    # NEW: resolve via registry
    registry = self._agent_type_registry or (
        context.metadata.get("_agent_type_registry") if context.metadata else None
    )
    definition = None
    if registry is not None:
        definition = registry.get(subagent_type)
        if definition is None:
            available = ", ".join(sorted(t.agent_type for t in registry.list_all()))
            return _error(
                call,
                f"unknown subagent_type: {subagent_type!r}. Available: {available}",
            )

    # ... rest unchanged, but pass definition to the task metadata:
    task = SubagentTask(
        task_id=task_id,
        goal=goal,
        request=request,
        metadata={
            "subagent_type": subagent_type,
            "expected_output": expected_output,
            "run_in_background": False,           # PR 10.5 will branch on this
            "_agent_type_def": definition,         # NEW — consumed by PR 10.3 builder
        },
    )
```

### 3. 注册表注入到引擎

**改 `aether/agents/core/agent.py`**：

构造 `AgentEngine` 时（与 `_skill_catalog` 同位置）：

```python
self._agent_type_registry: AgentTypeRegistry | None = agent_type_registry
if self._agent_type_registry is None and getattr(self.config, "agent_type_search_paths", None):
    self._agent_type_registry = AgentTypeRegistry(
        search_paths=self.config.agent_type_search_paths
    )
```

`_prepare_turn_entry` 把它挂上 metadata：

```python
context.metadata["_agent_type_registry"] = self._agent_type_registry
```

### 4. 默认 search paths

Engine 启动时若 config 未配置，自动补：

```python
def _default_agent_type_search_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    cwd_project = Path.cwd() / ".claude" / "agents"
    user = Path.home() / ".aether" / "agents"
    paths.append(user)              # earlier => lower priority
    paths.append(cwd_project)       # later wins
    return tuple(paths)
```

### 5. Config 字段

`aether/config/schema.py`：

```python
agent_type_search_paths: tuple[Path, ...] = ()  # empty => use defaults
agent_type_registry_enabled: bool = True
```

### 6. 示例 markdown

新建 `/workspace/Aether/.claude/agents/code-reviewer.md`（仅作演示，可以删）：

```markdown
---
name: code-reviewer
description: Reviews changes for security, style, and correctness issues
tools: [read_file, grep, glob, list_dir, web_fetch]
model: sonnet
max_turns: 25
---
You are a careful code reviewer.  Focus on:
- Security issues (injection, auth bypass, path traversal)
- Correctness (off-by-one, race conditions, type errors)
- Style consistency with the surrounding code

Report findings as a numbered list with file:line citations.
Do not propose fixes unless explicitly asked.
```

## 测试 / Tests

### Python

新建 `aether/tests/agents/types/test_definition.py`：
- `to_snapshot()` 往返 JSON 安全。

新建 `aether/tests/agents/types/test_markdown_loader.py`：
- 完整 frontmatter → 字段全对。
- 缺 `name`，文件名 stem 作 fallback。
- 缺 `description` → 返回 None。
- `tools: [a, b]` 与 `tools: a, b` 两种写法等价。
- 不存在的文件 → 返回 None（不抛）。

新建 `aether/tests/agents/types/test_registry.py`：
- builtins 总在；name 大小写不敏感查询。
- 同名 markdown 类型覆盖 builtin（验证 last-wins）。
- 无效 markdown 文件被跳过且不抛。

新建 `aether/tests/tools/test_agent_tool_type_resolution.py`：
- 未注入 registry → 兼容旧行为，`subagent_type="Explore"` 不报错（degrade 到没 enum 的 descriptor）。
- 注入 registry：
  - 未知类型 → `is_error=True`, content 含 "Available: ..."
  - 已知类型 → `task.metadata["_agent_type_def"]` 是对应 definition
  - descriptor.parameters.properties.subagent_type.enum 非空且包含所有 builtin 名

### 验收 / Acceptance

- `uv run pytest aether/tests/agents/types/ aether/tests/tools/test_agent_tool_type_resolution.py` 全绿。
- `uv run pyright` 无新告警。
- **手测**：跑 `uv run aether`，prompt `task(subagent_type="Explore", prompt="find usages of ToolRegistry")` 不报错；prompt `task(subagent_type="NotARealType", ...)` 报错且列出可用类型。
- 在 `.claude/agents/code-reviewer.md` 落地后，`task(subagent_type="code-reviewer", prompt="...")` 不报"unknown type"。

## 不在本 PR / Deferred

- **真正过滤工具 / 覆盖 system_prompt / model** —— 在 PR 10.3。本 PR 只解析到 `_agent_type_def`，builder 暂不消费。
- **MCP server per type** —— `open-claude-code` 支持 `mcpServers` 字段；Aether 当前没有 MCP，留给后续。
- **`hooks` per type** —— 同上，等 hook 系统成熟。
- **Plugin agent 加载** —— 等 Aether 引入 plugin 机制后再加。
