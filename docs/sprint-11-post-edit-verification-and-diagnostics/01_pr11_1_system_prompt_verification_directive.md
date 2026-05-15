# PR 11.1 — System Prompt: Verification Directive

## 目标 / Goal

把 `open-claude-code/src/constants/prompts.ts:211, 240` 中的两段硬约束语义复刻到 `aether/agents/core/system_prompt.py`：

1. **"完成前必须验证（verify before reporting complete）"** —— 强制模型在写完代码后跑 lint / 类型 / 测试或显式说明 *为什么没法验证*，而不是默认宣布"done"。
2. **"诚实汇报（faithful reporting）"** —— 当 verification 失败时，禁止模型"打哈哈"过去；必须把失败原貌带回给用户。

这两段是纯 prompt 改动，**先合入并观察一周**：
- 风险接近零：不动任何代码路径，不改 wire schema。
- 收益立竿见影：用户复现 "rename foo to bar" 流程时，模型会主动 `python -c "import module"` 或 `pyright` 一下，哪怕诊断 attachment 还没接（PR 11.5 之前）。
- 是 PR 11.5（attachment 注入）能 work 的前提之一：模型必须**愿意**读 diagnostics 块才有意义。

参考 `open-claude-code/src/constants/prompts.ts:198-244` 的 *"# Doing tasks"* 节。

## 当前问题 / Current Problem

### 1. `system_prompt.py` 全文 64 行，只注入了 tool contract

```python
# aether/agents/core/system_prompt.py:25-41
_CONTRACT_TEMPLATE = (
    "<tool_use_contract>\n"
    "You have these tools available: {names}.\n"
    "You MUST invoke them via the structured ``tool_calls`` field of your "
    "response. Do NOT write tool calls in markdown code blocks ..."
    "</tool_use_contract>"
)
```

—— **完全没有**关于 "verify before done" / "faithful report" 的文字。模型只被告知 *如何调工具*，没被告知 *什么时候算完事*。

### 2. 现象佐证

实测 prompt "在 `foo.py` 里把变量 `count` 重命名为 `total`"，Aether agent 通常：
- ✅ 调 `file_edit` 改一处
- ❌ 不会跑 `python -c "from foo import ..."`
- ❌ 不会调 `lsp` 工具看 NameError
- ✅ 立即回 "Done, renamed `count` to `total`"

对照 `open-claude-code` 同 prompt：
- ✅ 调 `file_edit`
- ✅ 主动 `bash python -c "import foo"` 或 `pyright foo.py`
- ✅ 看到 *其他文件引用 `count` 还没改* 后继续修
- ✅ 才说 "Done"

行为差异完全来自 system prompt 那一段 *"Before reporting a task complete, verify it actually works…"*。

### 3. `_CONTRACT_TEMPLATE` 是单字符串，没有 sectional 结构

如果后续想分别开关 *verification 段* 与 *faithful reporting 段*，目前的实现要么整段加要么整段不加。本 PR 顺手重构为多 section 拼装。

## 改动 / Changes

### 1. 重构 `system_prompt.py` 为分段拼装

```python
# aether/agents/core/system_prompt.py
"""System-prompt augmentation utilities used by the engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from aether.config.schema import EngineConfig
from aether.tools.base import ToolDescriptor


_TOOL_CONTRACT = (
    "<tool_use_contract>\n"
    "You have these tools available: {names}.\n"
    "You MUST invoke them via the structured ``tool_calls`` field of your "
    "response. Do NOT write tool calls in markdown code blocks "
    "(```bash...```), ``<function=NAME>{{...}}``, ``<functions.shell:N>{{...}}``, "
    "``<invoke name=...>``, ``<tool_call>``, or any other prose form — "
    "such text will be discarded and the run loop will exit without "
    "executing anything.\n"
    "Common mappings: to run a shell command, call the ``shell`` tool; "
    "to read a file, call ``read_file``; to list a directory, call "
    "``list_dir``; to search file contents, call ``grep``; to find files "
    "by name, call ``glob``; to write a file, call ``write_file``.\n"
    "</tool_use_contract>"
)

# Parity with open-claude-code/src/constants/prompts.ts:211.
# Kept in its own block (not inside <tool_use_contract>) so future
# prompt variants can disable it independently via config.
_VERIFICATION_DIRECTIVE = (
    "<verification_directive>\n"
    "Before reporting a task complete, verify it actually works: run the "
    "test, execute the script, type-check the module, read back the edited "
    "file. Minimum complexity means no gold-plating — it does NOT mean "
    "skipping the finish line. If you cannot verify (no test exists, "
    "cannot run the code, no language server), say so explicitly rather "
    "than claiming success.\n"
    "Specifically after editing source files, you SHOULD:\n"
    "  1. Re-read the changed file or run a syntax/type check (``pyright``, "
    "``tsc --noEmit``, ``python -c 'import module'``, etc.).\n"
    "  2. Search the rest of the codebase for callers of any renamed / "
    "removed symbol via ``grep``; missed callsites are the #1 source of "
    "regressions.\n"
    "  3. If the editor surfaces a ``<diagnostics>`` block in a subsequent "
    "user turn, treat its contents as authoritative and fix them before "
    "moving on.\n"
    "</verification_directive>"
)

# Parity with open-claude-code/src/constants/prompts.ts:240.
_FAITHFUL_REPORTING = (
    "<faithful_reporting>\n"
    "Report outcomes faithfully. If tests fail, say so with the relevant "
    "output; if you did not run a verification step, say that rather than "
    "implying it succeeded. Never claim ``all tests pass`` when the output "
    "shows failures, never suppress or simplify failing checks (tests, "
    "lints, type errors) to manufacture a green result, and never "
    "characterize incomplete or broken work as done. Equally, when a "
    "check did pass or a task is complete, state it plainly — do not hedge "
    "confirmed results with unnecessary disclaimers, downgrade finished "
    "work to ``partial``, or re-verify things you already checked. The "
    "goal is an accurate report, not a defensive one.\n"
    "</faithful_reporting>"
)


@dataclass(slots=True, frozen=True)
class SystemPromptOptions:
    include_tool_contract: bool = True
    include_verification_directive: bool = True
    include_faithful_reporting: bool = True


def augment_system_prompt(
    system: str | None,
    descriptors: Iterable[ToolDescriptor],
    options: SystemPromptOptions = SystemPromptOptions(),
) -> str | None:
    """Prepend Aether's standard system-prompt sections to *system*."""
    sections: list[str] = []
    if options.include_tool_contract:
        names = sorted({d.name for d in descriptors if d.name})
        if names:
            sections.append(
                _TOOL_CONTRACT.format(names=", ".join(f"``{n}``" for n in names))
            )
    if options.include_verification_directive:
        sections.append(_VERIFICATION_DIRECTIVE)
    if options.include_faithful_reporting:
        sections.append(_FAITHFUL_REPORTING)

    if not sections:
        return system

    header = "\n\n".join(sections)
    if system and system.strip():
        return f"{header}\n\n{system}"
    return header


def augment_system_with_tool_contract(
    system: str | None,
    descriptors: Iterable[ToolDescriptor],
) -> str | None:
    """Backwards-compatible alias for the pre-Sprint-11 API."""
    return augment_system_prompt(system, descriptors)


__all__ = [
    "SystemPromptOptions",
    "augment_system_prompt",
    "augment_system_with_tool_contract",
]
```

保留 `augment_system_with_tool_contract` 作为别名，所有现有调用点零修改即可获得三段新文本。

### 2. 新 config 字段

`aether/config/schema.py`：

```python
# In EngineConfig
verification_directive_enabled: bool = True
faithful_reporting_enabled: bool = True
```

默认 `True`：希望模型立即获得新行为。若有用户复用 prompt cache 或自带 system prompt 想关掉，可以 `EngineConfig(verification_directive_enabled=False, ...)`。

### 3. Engine wire-up

`aether/agents/core/agent.py` 里调 `augment_system_with_tool_contract` 的位置（约 `_prepare_session_and_system_prompt`，line 464-468）替换为：

```python
from aether.agents.core.system_prompt import (
    SystemPromptOptions,
    augment_system_prompt,
)

active_system = augment_system_prompt(
    active_system,
    descriptors,
    SystemPromptOptions(
        include_tool_contract=True,
        include_verification_directive=self.config.verification_directive_enabled,
        include_faithful_reporting=self.config.faithful_reporting_enabled,
    ),
)
```

### 4. Subagent 继承

`aether/subagents/default_builder.py` 的 `child_config` 构造（line 26-51）追加：

```python
verification_directive_enabled=parent.config.verification_directive_enabled,
faithful_reporting_enabled=parent.config.faithful_reporting_enabled,
```

—— child 默认与 parent 一致；类型注册表（Sprint 10.2）落地后可以让特定类型（如 `Explore`）关掉 verification（只读类型不需要 verify）。

## 测试 / Tests

新建 `aether/tests/agents/core/test_system_prompt.py`（如已存在则扩展）：

- `test_default_includes_all_three_sections` —— 默认 options → 输出含 `<tool_use_contract>` / `<verification_directive>` / `<faithful_reporting>` 三块且顺序固定。
- `test_disable_verification_drops_block` —— `include_verification_directive=False` → 输出不含 `<verification_directive>`。
- `test_empty_tool_list_drops_contract` —— descriptors 为空 + verification 开 → 输出仅 `<verification_directive>` + `<faithful_reporting>`。
- `test_preserves_user_system_message` —— 传入 user system prompt → header 在前，user message 在 `\n\n` 后。
- `test_backwards_compat_alias_returns_same_default_behavior` —— `augment_system_with_tool_contract(s, ds)` ≡ `augment_system_prompt(s, ds, SystemPromptOptions())`。

新建 `aether/tests/agents/test_engine_system_prompt_wiring.py`（scripted provider）：

- `test_engine_emits_verification_directive_in_first_request` —— `EngineConfig(verification_directive_enabled=True)` → 第一次 `provider.create_message` 收到的 `system` 含 `<verification_directive>`。
- `test_engine_can_disable_directives` —— `EngineConfig(verification_directive_enabled=False, faithful_reporting_enabled=False)` → 仅 `<tool_use_contract>`。
- `test_subagent_inherits_directive_flags` —— 父开 / 子未 override → child engine 的第一次请求也含 directive。

## 验收 / Acceptance

- `uv run pytest aether/tests/agents/core/test_system_prompt.py aether/tests/agents/test_engine_system_prompt_wiring.py` 全绿。
- `uv run pyright` 无新告警。
- 旧调用点零编译错误（`augment_system_with_tool_contract` alias 保留）。
- **手测 A**（PR 11.5 落地前也可观察到改善）：在 Python 仓库 prompt "把 `foo.py` 里的 `count` 重命名为 `total`" → agent 主动 `grep` 其余引用并修补，最后报告 "Done" 时附测试或类型检查输出。
- **手测 B**：故意构造一个会让 agent 失败的任务（如 prompt "实现 X 但 X 依赖一个不存在的库"），agent 应明说 "无法验证因为缺少依赖" 而不是糊弄性"完成"。
- 旧 prompt cache 命中率有所下降是预期内的——前缀多了两段固定文本，仍能享受缓存（属于 prompt prefix 而非 user turn）。

## 不在本 PR / Deferred to other PRs

- `<verification_directive>` 内提到的 `<diagnostics>` 块本身（由 PR 11.5 引入）；本 PR 提前在 prompt 里 "宣告"，等 11.5 落地后自然生效。
- 让 `Explore` 类型 default-disable 的 verification directive —— 等 Sprint 10.3 / 10.2 落地后再追加。
- 把 directive 文本接入 i18n —— 现阶段全英文，与现有 prompt 风格一致。
