# PR 10.1 — Skill Listing System-Reminder Injection

## 目标 / Goal

把 `SkillCatalog`（`aether/runtime/tools/skill_catalog.py`）发现的所有 skill 的 **frontmatter**（`name` + `description` + 可选 `when_to_use`）拼装成一个 `<system-reminder>` 块，注入到对话 turn 0；并在长会话中按 `skill_nudge_interval` 周期性重发。

**关键纪律不变**：skill body 仍然 lazy-load，仅在模型调 `skill` 工具时才读盘 —— 清单只承载 metadata（每条约 50-150 字符）。

参考 `open-claude-code/src/utils/attachments.ts:2661-2751`（`getSkillListingAttachments`）+ `src/utils/messages.ts:3728-3738`（`wrapMessagesInSystemReminder`）。

## 当前问题 / Current Problem

### 1. 模型完全看不到 skill 菜单

`aether/agents/core/system_prompt.py:43-61` 的 `augment_system_with_tool_contract` 只把工具名列出来：

```python
_CONTRACT_TEMPLATE = (
    "<tool_use_contract>\n"
    "You have these tools available: {names}.\n"
    ...
)
```

—— **没有任何 skill 信息进 system message**。模型若想用 `skill` 工具，必须先知道 skill 的精确名字，否则只能瞎猜。

### 2. `should_review_skills` 是死代码

`aether/agents/core/agent.py:1574-1599`：

```python
def _maybe_set_skill_review(self, context, *, iteration: int) -> None:
    context.metadata.setdefault("should_review_skills", False)
    ...
    interval = max(0, int(getattr(self.config, "skill_nudge_interval", 0)))
    if interval > 0 and iteration > 0 and iteration % interval == 0:
        context.metadata["should_review_skills"] = True
```

—— flag 被置位，但**没有任何代码读取这个 flag 去做 prompt 注入**。`grep -rn "should_review_skills" aether/` 仅命中本写入处与几个 test fixture。

### 3. SkillTool 错误提示里有"available"列表，但只在出错时

`aether/tools/builtins/skill.py:99-109`：

```python
skill = catalog.get(skill_name)
if skill is None:
    available = ", ".join(
        s.name for s in catalog.list_all()[: self._MAX_LIST_PREVIEW]
    )
    suffix = (
        f"\nAvailable (first {self._MAX_LIST_PREVIEW}): {available}"
        if available
        else "\nThe catalog is empty."
    )
    return _error(call, f"unknown skill: {skill_name!r}{suffix}")
```

—— 只在**调错时**才暴露清单，等于"先撞墙再给地图"。

## 改动 / Changes

### 1. 新文件 `aether/runtime/tools/skill_listing.py`

```python
"""Render a SkillCatalog into a `<system-reminder>` block for the model.

Parity with open-claude-code (`src/utils/attachments.ts:2661-2751`):
only the frontmatter (name + description, optional when_to_use) lands
in the prompt — the body stays on disk and is loaded only when the
``skill`` tool is invoked.  This keeps token cost ~150 chars / skill.
"""

from __future__ import annotations

from typing import Iterable

from aether.runtime.tools.skill_catalog import Skill, SkillCatalog


_HEADER = "The following skills are available for use with the `skill` tool:"
_FOOTER_HINT = (
    "Invoke a skill via `skill(skill=\"<name>\", args=\"<optional args>\")`. "
    "Skill bodies are loaded on demand."
)


def format_skill_listing(
    catalog: SkillCatalog | None,
    *,
    budget_chars: int = 4096,
) -> str:
    """Return a `<system-reminder>` block or empty string if no skills."""
    if catalog is None:
        return ""
    skills = sorted(catalog.list_all(), key=lambda s: s.name)
    if not skills:
        return ""

    lines: list[str] = ["<system-reminder>", _HEADER, ""]
    used = sum(len(s) + 1 for s in lines)
    omitted = 0
    for skill in skills:
        entry = _format_entry(skill)
        if used + len(entry) + 1 > budget_chars:
            omitted = len(skills) - (len(lines) - 3)
            break
        lines.append(entry)
        used += len(entry) + 1

    if omitted > 0:
        lines.append(
            f"... ({omitted} more skills omitted; "
            f"use `skill` tool with the exact name to load any of them)"
        )
    lines.extend(["", _FOOTER_HINT, "</system-reminder>"])
    return "\n".join(lines)


def _format_entry(skill: Skill) -> str:
    desc = (skill.description or "").strip().replace("\n", " ")
    if skill.when_to_use:
        when = skill.when_to_use.strip().replace("\n", " ")
        return f"- {skill.name}: {desc} (use when: {when})"
    return f"- {skill.name}: {desc}" if desc else f"- {skill.name}"


__all__ = ["format_skill_listing"]
```

### 2. Turn 0 注入点

**改 `aether/agents/core/agent.py`**，找到 `_prepare_session_and_system_prompt`（约 line 464-468）。在 `augment_system_with_tool_contract` 之后追加：

```python
if self.config.skill_listing_enabled and self._skill_catalog is not None:
    listing = format_skill_listing(
        self._skill_catalog,
        budget_chars=self.config.skill_listing_token_budget,
    )
    if listing:
        # Append after the tool contract; keeps the existing user message
        # in its own turn.  The listing is part of the system message so
        # it sits in the prompt cache prefix.
        active_system = (active_system or "") + "\n\n" + listing
```

`from aether.runtime.tools.skill_listing import format_skill_listing` 加到文件顶部。

### 3. 周期性 nudge — 消费 `should_review_skills` flag

继续在 `agent.py`，找到 PRE_LLM 状态边界（line 1574 周围）。新增方法：

```python
def _maybe_inject_skill_nudge(
    self,
    messages: list[dict[str, Any]],
    context: TurnContext,
) -> None:
    """Re-inject the skill listing as a user-role system-reminder."""
    if not context.metadata.get("should_review_skills"):
        return
    context.metadata["should_review_skills"] = False  # consume
    if not self.config.skill_listing_enabled or self._skill_catalog is None:
        return
    listing = format_skill_listing(
        self._skill_catalog,
        budget_chars=self.config.skill_listing_token_budget,
    )
    if not listing:
        return
    messages.append({
        "role": "user",
        "content": listing,                    # already wrapped in <system-reminder>
        "metadata": {"source": "skill_nudge"}  # tracing
    })
```

在主循环每次 PRE_LLM 前调一次（紧挨 `_maybe_set_skill_review` 之后）。

### 4. 新 config 字段

`aether/config/schema.py`：

```python
skill_listing_enabled: bool = True
skill_listing_token_budget: int = 4096  # 字符数近似
# skill_nudge_interval: int = 0           # 已存在 — 0 关闭周期 nudge
```

### 5. TUI 端

**完全不动**。`<system-reminder>` 是上下文里的内容，永远不进 `ChatItem`，TUI 看不到也不该看到。

## 测试 / Tests

### Python

新建 `aether/tests/runtime/tools/test_skill_listing.py`：

- `test_empty_catalog_returns_empty_string` —— `SkillCatalog([])` → `""`。
- `test_lists_all_within_budget` —— mock catalog with 3 skills, budget=10_000 → 输出含三个 name；首行含 `<system-reminder>`；尾行 `</system-reminder>`。
- `test_truncates_when_over_budget` —— 50 个 skill，budget=400 → 输出末尾含 `... (N more skills omitted; ...)`，N 正确。
- `test_when_to_use_field_rendered` —— skill 含 `when_to_use` → 输出 `(use when: ...)`。
- `test_sort_is_alphabetical` —— 不同 name 输入顺序，输出始终按 name 排序。

新建 `aether/tests/agents/test_skill_listing_injection.py`（用 scripted provider）：

- `test_turn_0_system_includes_listing` —— 构造一个 engine + catalog with 2 skills → 第一次 LLM 请求的 `system` 末尾含 `<system-reminder>` 且列出两个 name。
- `test_disabled_via_config` —— `skill_listing_enabled=False` → system 不含 listing。
- `test_no_listing_when_catalog_empty` —— catalog 空 → 不注入空块，system 不变。

新建 `aether/tests/agents/test_skill_nudge_interval.py`：

- `test_nudge_appears_on_interval` —— `skill_nudge_interval=2`，scripted provider 跑 5 个 iteration → 第 2 次和第 4 次 LLM 调用前 messages 末尾出现 `metadata.source=="skill_nudge"` 的 user turn。
- `test_nudge_zero_means_off` —— `skill_nudge_interval=0` → 永不出现 nudge。

## 验收 / Acceptance

- `uv run pytest aether/tests/runtime/tools/test_skill_listing.py aether/tests/agents/test_skill_*.py` 全绿。
- `uv run pyright` 无新告警。
- **手测**：跑 `uv run aether`，问 "What skills are available?" → 模型应能给出具体 skill 名字而不是 "I don't know"。
- 性能：序列化 4096 字符 budget 的清单耗时 < 1ms（测一次 `time.perf_counter` 包裹 `format_skill_listing`）。
- 旧 wire 客户端零变化：listing 完全不上 wire，gateway 协议未改。

## 不在本 PR / Deferred to other PRs

- `disable-model-invocation` / `user-invocable` 等 frontmatter flag 的解析与过滤 —— 留给后续（与 plugin / mcp skill 一起做）。
- 内联 `` !`shell-cmd` `` 在 skill body 中执行 —— `open-claude-code` 有，但 Aether 暂不需要；如要做单独立项。
- TUI 端的 "skill picker" UI —— 留给 Sprint 11+。
