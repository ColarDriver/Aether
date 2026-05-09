# PR 3.5.8 — `SkillTool` + Skill Catalog 基础设施

> **角色**：让 LLM 主动**发现并加载** SKILL.md 文件作为额外指引。
> Aether 仓库根已有 `/workspace/Aether/skills/` 目录（70+ SKILL.md 文件），
> 本 PR 把这套资产暴露给模型。

## 一、目标

1. 实现 `SkillCatalog` — 扫描配置的 skill 目录，按 frontmatter / 路径建立索引。
2. 实现 `SkillTool` — 模型可按 skill name 调用，工具把 SKILL.md 内容注入下一轮的 system 增强或 ToolResult。
3. 支持两种执行模式：
   * **inline**（默认）：把 SKILL.md 内容作为 ToolResult 返回给主 agent
   * **forked**（可选）：在 subagent 上下文执行 skill（隔离 budget）— v1 实现 inline，forked 留入口

## 二、为什么要做

### 2.1 现状

仓库 `skills/` 目录有大量精心编写的 SKILL.md：
* `software-development/test-driven-development/SKILL.md` — 教模型如何 TDD
* `github/github-pr-workflow/SKILL.md` — 教模型 PR 流程
* `software-development/writing-plans/SKILL.md` — 教模型如何写计划
* …

但模型当前**不知道这些存在**。每次启动只看到 system prompt，不看 skills 目录。

### 2.2 期望工作流

1. 用户："帮我做 TDD"
2. 模型：「我需要 TDD 指引」→ 调用 `skill(skill="test-driven-development")`
3. SkillTool 返回 SKILL.md 内容
4. 模型按指引推进

更**进阶**：在 system prompt 里列出**所有**可用 skill 的名字 + `whenToUse`，
让模型主动选用而无需 user 提示。

### 2.3 与 claude-code 的差别

claude-code 的 SkillTool（1100 行）深度集成了 plugin marketplace / MCP / remote-skill。
Aether **v1 只做本地 skill 目录**。
* 不做 plugin marketplace / MCP / remote download / forked execution（v1）
* 留 hook（forked 模式 / 远程 catalog）给 v2

## 三、设计

### 3.1 Skill 数据模型

```python
# runtime/skill_catalog.py - 新文件
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True, frozen=True)
class Skill:
    name: str  # canonical id, e.g. "test-driven-development"
    path: Path  # absolute path to SKILL.md
    description: str = ""  # frontmatter "description" or first paragraph
    when_to_use: str = ""  # frontmatter "whenToUse"
    body: str = ""  # SKILL.md without frontmatter
    source: str = "local"  # "local" / "user-home" / future "remote"
    version: Optional[str] = None
```

### 3.2 SkillCatalog

```python
class SkillCatalog:
    """Discover and serve skills from configured directories.

    Discovery order (later wins on name conflict):
    1. Bundled (built-in to aether package, if any)
    2. Repo-local: <project_root>/skills/**/SKILL.md
    3. User home: ~/.aether/skills/**/SKILL.md
    """

    def __init__(self, *, search_paths: list[Path]):
        self.search_paths = search_paths
        self._skills: dict[str, Skill] = {}
        self._loaded = False

    def discover(self) -> None:
        """Scan all search paths once. Idempotent."""
        if self._loaded:
            return
        for root in self.search_paths:
            if not root.exists():
                continue
            for skill_md in root.rglob("SKILL.md"):
                skill = self._parse_skill_md(skill_md, root=root)
                if skill is not None:
                    self._skills[skill.name] = skill
        self._loaded = True

    def get(self, name: str) -> Optional[Skill]:
        self.discover()
        # Try exact, then case-fold, then strip leading slash
        if name in self._skills:
            return self._skills[name]
        normalized = name.lstrip("/").strip().lower()
        for k, v in self._skills.items():
            if k.lower() == normalized:
                return v
        return None

    def list_all(self) -> list[Skill]:
        self.discover()
        return list(self._skills.values())

    def _parse_skill_md(self, path: Path, *, root: Path) -> Optional[Skill]:
        """Parse SKILL.md with optional YAML frontmatter."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None

        frontmatter, body = self._split_frontmatter(text)
        # Skill name = directory containing SKILL.md (relative to root, joined by "-")
        rel = path.parent.relative_to(root)
        name = "-".join(rel.parts) or path.parent.name
        if "name" in frontmatter:
            name = frontmatter["name"]

        return Skill(
            name=name,
            path=path,
            description=frontmatter.get("description", "").strip(),
            when_to_use=frontmatter.get("whenToUse", "").strip(),
            body=body,
            source="local",
            version=frontmatter.get("version"),
        )

    def _split_frontmatter(self, text: str) -> tuple[dict, str]:
        """Parse YAML frontmatter delimited by '---'."""
        lines = text.split("\n", 1)
        if not text.startswith("---\n"):
            return {}, text
        try:
            end = text.index("\n---\n", 4)
        except ValueError:
            return {}, text
        front_text = text[4:end]
        body = text[end + 5:]
        # Use stdlib (no yaml dep) — handle simple "key: value" only
        front: dict = {}
        for line in front_text.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                front[k.strip()] = v.strip().strip('"').strip("'")
        return front, body
```

### 3.3 SkillTool

```python
class SkillTool(ToolExecutor):
    NAME = "skill"
    MAX_RESULT_CHARS = 60_000

    def __init__(self, catalog: SkillCatalog):
        self.catalog = catalog

    def execute(self, call, context):
        skill_name = call.arguments.get("skill", "").strip()
        args = call.arguments.get("args", "")

        if not skill_name:
            return ToolResult(content="skill name required", is_error=True)

        skill = self.catalog.get(skill_name)
        if skill is None:
            available = ", ".join(s.name for s in self.catalog.list_all()[:20])
            return ToolResult(
                content=(
                    f"unknown skill: {skill_name!r}\n"
                    f"Available (first 20): {available}\n"
                    f"Use the skill_search tool / list to discover more."
                ),
                is_error=True,
            )

        # Substitute simple placeholders
        body = skill.body
        if args:
            body = body.replace("$ARGUMENTS", args).replace("${ARGUMENTS}", args)
        body = body.replace("${AETHER_SESSION_ID}", context.session_id)

        # Format with header for clarity
        full_output = (
            f"# Loaded skill: {skill.name}\n"
            f"Source: {skill.path}\n"
            f"{f'Description: {skill.description}' if skill.description else ''}\n\n"
            f"--- BEGIN SKILL ---\n{body}\n--- END SKILL ---"
        )
        content = self._maybe_spill(full_output, call=call, context=context, extension="md")
        return ToolResult(call_id=call.id, content=content, is_error=False)
```

### 3.4 Skill 列表注入 system prompt（可选 v1.1 增强）

为了让模型**主动**发现 skill，PR 3.5.8 也可在 prompt builder 阶段拼接 skill 列表：

```python
# agents/core/prompt_builder.py 增量
def _format_skill_list(catalog: SkillCatalog) -> str:
    skills = catalog.list_all()
    if not skills:
        return ""
    lines = ["", "## Available skills (use `skill` tool to load):", ""]
    for s in sorted(skills, key=lambda x: x.name):
        line = f"- `{s.name}`"
        if s.when_to_use:
            line += f" — {s.when_to_use[:120]}"
        lines.append(line)
    return "\n".join(lines)
```

仅在 `EngineConfig.skill_list_in_system_prompt=True` 时启用（默认 `False`，避免无谓 prompt 增长）。

### 3.5 EngineConfig 新字段

```python
# Sprint 3.5 / PR 3.5.8
skill_tool_enabled: bool = True
skill_search_paths: tuple[Path, ...] = ()  # 空 → 默认 [project_root/skills, ~/.aether/skills]
skill_list_in_system_prompt: bool = False  # 让模型主动发现
```

## 四、文件改动清单

| 文件 | 类型 | 行数 |
|---|---|---|
| `backend/harness/aether/runtime/skill_catalog.py` | **新文件** | ~200 |
| `backend/harness/aether/tools/builtins/skill.py` | **新文件** | ~100 |
| `backend/harness/aether/tools/builtins/__init__.py` | 修改 | ~5 |
| `backend/harness/aether/agents/core/prompt_builder.py` | 修改（如有） | skill list 注入 | ~30 |
| `backend/harness/aether/config/schema.py` | 修改 | ~20 |
| `backend/harness/aether/tests/test_skill_catalog.py` | **新文件** | ~250 |
| `backend/harness/aether/tests/test_skill_tool.py` | **新文件** | ~200 |
| `backend/harness/aether/tests/fixtures/skills/test-skill-a/SKILL.md` | **新文件** | ~10 |
| `backend/harness/aether/tests/fixtures/skills/test-skill-b/SKILL.md` | **新文件** | ~10 |

## 五、测试用例

### 5.1 测试组 A：SkillCatalog

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | 单目录 + 2 skill | discover 后 list_all 返回 2 |
| **T-A2** | 嵌套目录（`a/b/SKILL.md`） | name == "a-b" |
| **T-A3** | frontmatter `name:` | name 来自 frontmatter（非路径） |
| **T-A4** | 多目录优先级 | 后扫描的目录覆盖前者同名 skill |
| **T-A5** | 不存在的目录 | 不报错，跳过 |
| **T-A6** | discover 调用 2 次 | 只扫描 1 次（idempotent） |
| **T-A7** | get 大小写不敏感 | "TDD" 命中 "tdd" |
| **T-A8** | get 带前导 "/" | "/tdd" 命中 "tdd" |
| **T-A9** | get 不存在 | 返回 None |
| **T-A10** | 带 frontmatter 的 SKILL.md | description / whenToUse 正确解析 |
| **T-A11** | 没有 frontmatter | description / whenToUse 为空 |
| **T-A12** | 损坏的 SKILL.md（不可读） | 跳过，不报错 |

### 5.2 测试组 B：SkillTool

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | 加载已知 skill | content 含 "Loaded skill" + body |
| **T-B2** | 未知 skill | `is_error=True`；列出前 20 |
| **T-B3** | 空 skill name | `is_error=True` |
| **T-B4** | args 替换 $ARGUMENTS | body 中 $ARGUMENTS 被替换 |
| **T-B5** | $AETHER_SESSION_ID 替换 | 用 context.session_id 替换 |
| **T-B6** | body 巨大触发 spill | spill；preview ≤ 60k |

### 5.3 测试组 C：system prompt 注入

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | skill_list_in_system_prompt=False | system prompt 不含 skill 列表 |
| **T-C2** | skill_list_in_system_prompt=True | system prompt 含 "Available skills" + 名字 + when_to_use 摘要 |

## 六、验收门

* [ ] 22+ case 全绿
* [ ] 真实跑：模型调用 `skill(skill="test-driven-development")` 返回该 SKILL.md 内容
* [ ] 真实跑：开启 skill_list_in_system_prompt，模型在合适场景主动调用 skill

## 七、回滚开关

* `skill_tool_enabled=False` → 不注册工具
* `skill_search_paths=()` → catalog 空，工具能用但找不到 skill

## 八、实施顺序（建议 3 天）

| 步骤 | 时长 |
|---|---|
| 1. SkillCatalog 实现（含 frontmatter 解析） | 4h |
| 2. SkillCatalog 测试 | 3h |
| 3. SkillTool 实现 | 2h |
| 4. SkillTool 测试 + fixture | 3h |
| 5. system prompt 集成（可选） | 2h |
| 6. EngineConfig + 注册 | 1h |
| 7. 真实模型 smoke | 2h |
| 8. 文档（README / overview 提到 skill） | 1h |

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| 70+ skills 全列入 system prompt 导致 token 暴涨 | 默认 `skill_list_in_system_prompt=False`；启用时可加 `skill_filter` 配置 |
| frontmatter 用复杂 YAML 我们 parser 不支持 | v1 仅支持 `key: value`；复杂场景报警跳过；v2 引入 `pyyaml` 依赖 |
| skill body 含恶意指令（prompt injection） | 用户控制 skill_search_paths；信任本地用户管理 |
| skill 之间相互引用（chain） | v1 不递归；模型可显式连续调用多个 skill |
| forked 模式缺失 | v1 文档说明；v2 接 SubagentManager |

## 十、与后续 PR 的接合

* **forked execution（v2）**：复用 PR 3.5.6 的 SubagentManager
* **远程 skill catalog（v3）**：HTTP fetch + 本地缓存
* **PR 3.2 cheap_tool_names**：`"skill_manage"` 应是 cheap（仅枚举 skill），目前未在 v1 实现；
  如做 `skill_list` / `skill_search` 子工具，加入 cheap 名单
