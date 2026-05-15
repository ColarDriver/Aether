# PR 11.6 — Verifier Subagent Type

## 目标 / Goal

新增一个内置 subagent 类型 `Verifier`，与 `Explore` / `general-purpose` 并列存在。同时在父 engine 的 system prompt（PR 11.1 的 verification directive 之后）补一段硬约束：**非平凡变更（≥ 3 个文件 / 涉及后端 API / 涉及 CI/infra）必须 spawn 一个 `Verifier` 子 agent 并拿到 PASS 才能 report complete**。

参考 `open-claude-code/src/constants/prompts.ts:394` 与 `src/coordinator/coordinatorMode.ts` 中关于 verifier subagent 的角色描述。

PR 11.5 让模型能 *看见* 错；本 PR 让模型在跨越多个文件的复杂变更中无法仅凭 *自我感觉良好* 宣布完成 —— 必须经独立角色复核。

## 当前问题 / Current Problem

### 1. 没有 verifier 角色

`aether/agents/types/`（Sprint 10.2 落地）目前只有 4 个内置类型（`Explore` / `general-purpose` / 等），都偏向"做事"。没有一个 *只读 + 跑命令 + 投票* 的 verifier 角色。

### 2. `system_prompt.py` 的 verification directive（PR 11.1）只对 *自检* 起作用

PR 11.1 让模型 *愿意* 自查；但模型自查时它仍然是同一个上下文里的同一个角色：

- 它知道自己刚写了什么 → 容易"自证"。
- 它会重用刚才的临时变量、tool history → 容易看不到自己挖的坑。
- 它没有"独立 prompt cache" → 没有 fresh eyes。

OCC 的解决方案是 *分裂上下文*：spawn 一个 verifier subagent，让它 *不知道* 实现过程，只看 *最终代码状态 + 任务原始描述*，独立给出 PASS / FAIL / PARTIAL。

### 3. `AgentTool` 不强制 verifier

`tools/builtins/agent_tool.py` 让父 agent *可以* 调 `task(subagent_type="Verifier", ...)`，但不强制。需要：

- 一段 prompt directive（属于 PR 11.1 与本 PR 共享的 `_VERIFICATION_DIRECTIVE` 的 *延伸段*）告诉模型 "≥ 3 files → MUST spawn Verifier"。
- 一个轻量的 *advisory* gate：engine 跟踪本 session 里 `edited_paths` 的并集，当 ≥ 阈值时 *在每次 PRE_LLM 注入一条 reminder*。

这个 gate 是软的：模型仍然 *可能* 不听话直接 final answer。但配合 attachment + reminder + prompt directive，三层加在一起在 OCC 上已经验证够用。

## 改动 / Changes

### 1. 新内置类型 `Verifier`

`aether/agents/types/builtin/verifier.py`（NEW）：

```python
"""Built-in 'Verifier' subagent type.

The Verifier reads files, runs commands, and assigns a verdict (PASS /
FAIL / PARTIAL).  It does NOT edit code, send messages, or spawn
further subagents.  Its prompt cache prefix is independent of the
parent, so it brings fresh eyes to the work.

Parity with open-claude-code ``src/coordinator/coordinatorMode.ts`` and
the verification-gate paragraph in ``src/constants/prompts.ts:394``.
"""

from __future__ import annotations

from aether.agents.types.contracts import AgentTypeDefinition

VERIFIER_AGENT_TYPE = "Verifier"

VERIFIER_SYSTEM_PROMPT = """\
You are an independent verifier.  A separate agent claims to have
completed a task; you must decide whether the claim is correct.

Your inputs:
  - The original user request (verbatim).
  - The list of files claimed to be changed.
  - The current working tree.

You MUST:
  1. Read the changed files (use ``read_file`` / ``grep``).
  2. Run the project's verification commands (``pyright``, ``pytest``,
     ``tsc --noEmit``, ``npm run lint``, ``python -c 'import X'``,
     depending on the project layout).
  3. Spot-check that the change actually addresses the user's request,
     not just that it compiles.
  4. Assign exactly one verdict:
       - PASS    — everything works, change is correct.
       - PARTIAL — change is partially correct, missing pieces or
                   minor regressions.
       - FAIL    — change does not work or breaks existing behavior.
  5. Every PASS claim MUST be backed by a code block showing the
     command you ran and its output.  Hand-wavy PASS is NOT acceptable.

You MUST NOT:
  - Modify any file.
  - Send messages back to the parent (your verdict IS the message).
  - Spawn further subagents.

Output format (Markdown):

```
## Verdict: PASS | PARTIAL | FAIL

## Evidence
[one section per command you ran]

## Issues
[only present when verdict is PARTIAL or FAIL]
```
"""

DEFINITION = AgentTypeDefinition(
    name=VERIFIER_AGENT_TYPE,
    description=(
        "Independent verifier subagent. Reads the final state of the "
        "code, runs the project's verification commands, and assigns "
        "PASS / PARTIAL / FAIL.  Has no write tools."
    ),
    system_prompt=VERIFIER_SYSTEM_PROMPT,
    allowed_tools=frozenset({
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "shell",
        "lsp",         # may query LSP directly
    }),
    denied_tools=frozenset({
        "file_edit",
        "write_file",
        "notebook_edit",
        "task",        # cannot spawn further subagents
        "send_message",
    }),
    background=False,
    model=None,  # inherit parent's model — verifier is not always cheaper
)
```

`AgentTypeDefinition` 来自 Sprint 10.2；新字段 `allowed_tools` / `denied_tools` / `background` / `model` 这些都已在 Sprint 10.3 落地。

注册到内置表 `aether/agents/types/builtin/__init__.py`：

```python
from aether.agents.types.builtin.verifier import DEFINITION as VERIFIER

BUILTIN_DEFINITIONS = (
    EXPLORE,
    GENERAL_PURPOSE,
    VERIFIER,
    # ... (其他内置)
)
```

### 2. Verification gate prompt 段

在 PR 11.1 的 `_VERIFICATION_DIRECTIVE` 末尾追加一段（**或单独抽出 `_VERIFIER_GATE`** 用 `SystemPromptOptions.include_verifier_gate` 开关；推荐独立 section 以便 A/B）：

```python
_VERIFIER_GATE = (
    "<verifier_gate>\n"
    "When non-trivial implementation has happened on your turn — defined "
    "as 3+ files edited, backend/API changes, or infrastructure changes — "
    "you MUST request an independent verification BEFORE reporting "
    "completion.  Spawn the ``task`` tool with "
    "``subagent_type=\"Verifier\"``.  Pass:\n"
    "  - the original user request,\n"
    "  - the list of files changed (by anyone — you, a fork, or a "
    "    subagent),\n"
    "  - the approach you took,\n"
    "  - any plan file path if you authored one.\n"
    "Your own checks, caveats, and a fork's self-checks do NOT substitute "
    "for the verifier's verdict.  On FAIL: fix, resume the verifier with "
    "the fix, repeat until PASS.  On PARTIAL: report exactly what passed "
    "and what could not be verified.  On PASS: spot-check it — re-run 2–3 "
    "commands from its report; if any PASS lacks a matching command "
    "block or diverges from your re-run, resume the verifier.\n"
    "</verifier_gate>"
)
```

新 config 字段 `verifier_gate_enabled: bool = True` 在 `EngineConfig`（与 11.1 那两个同处）。

### 3. 软 gate：edited_path 累计提醒

`agent.py` 主循环维护 `session.metadata["_session_edited_paths"]: set[str]`。每次 PRE_LLM 时如果：

```python
edited_count = len(session_edited_paths)
verifier_already_invoked = session.metadata.get("_verifier_invoked", False)
threshold = self.config.verifier_gate_file_threshold  # default 3
```

满足 `edited_count >= threshold and not verifier_already_invoked` → 通过 `AttachmentDispatcher`（PR 11.5）追加一条 reminder：

```
<system-reminder>
You have edited {N} files this session and have not yet spawned a
Verifier subagent.  Per <verifier_gate>, complex changes require an
independent PASS before you may report completion.  Spawn one now via
`task(subagent_type="Verifier", ...)` or explain why this change is
trivial enough to skip.
</system-reminder>
```

`_verifier_invoked` 由 `post_tool_use` hook 监听：当 `tool_name == "task"` 且 `tool_args.subagent_type == "Verifier"` → 置位 True。

新 config 字段：

```python
verifier_gate_enabled: bool = True
verifier_gate_file_threshold: int = 3
```

### 4. `agent_tool.py` 强制 Verifier 描述

`tools/builtins/agent_tool.py` 的 `description`（descriptors 渲染给 LLM 的那段）追加：

> When ``subagent_type="Verifier"``, the verifier runs independently and assigns PASS / PARTIAL / FAIL.  Pass the original user request and the list of files changed; do not share your own test results in the prompt.

### 5. Subagent 透传

`DefaultSubagentBuilder` 已能根据 `AgentTypeDefinition.allowed_tools` / `denied_tools` 过滤 child registry（Sprint 10.3）。本 PR 不动 builder。

### 6. Engine 不强制；只软提醒

软 gate 不能阻止模型直接答 "done" —— 那是 prompt-level 教学问题，不在 engine 范围。但 reminder + system prompt 双管齐下后，failure mode 从 *默默不验证* 转为 *显式选择不验证*，至少留下可观测信号（log 含 "session edited 5 files, no Verifier spawned, final answer emitted"）。

## 测试 / Tests

新建 `aether/tests/agents/types/test_verifier_definition.py`：

- `test_verifier_definition_has_no_write_tools` —— `denied_tools` 覆盖 `file_edit` / `write_file` / `notebook_edit` / `task` / `send_message`。
- `test_verifier_allowed_tools_include_read_run` —— `allowed_tools` 覆盖 `read_file` / `shell` / `grep` / `glob` / `list_dir` / `lsp`。
- `test_builtin_registry_includes_verifier` —— `BUILTIN_DEFINITIONS` 含 name=="Verifier"。

新建 `aether/tests/agents/test_verifier_gate_reminder.py`（scripted provider）：

- `test_reminder_appears_at_threshold` —— 模型用 `file_edit` 改 3 个文件 → 第 4 个 turn 的 messages 含 `<system-reminder>` 提示 spawn Verifier。
- `test_reminder_clears_after_verifier_invoked` —— 改 3 个文件 → spawn Verifier → reminder 不再出现。
- `test_reminder_respects_threshold_config` —— `verifier_gate_file_threshold=5` → 改 3 个文件不触发；改 5 个触发。
- `test_reminder_disabled_via_config` —— `verifier_gate_enabled=False` → 改 10 个也不出现 reminder。

新建 `aether/tests/integration/test_verifier_end_to_end.py`（scripted parent + scripted child）：

- `test_parent_spawns_verifier_and_blocks_until_pass` —— 父模型按 prompt 指引调 `task(subagent_type="Verifier")` → 子 agent 用 `shell("pyright")` 拿到 0 错 → 返回 PASS → 父继续 final answer。
- `test_verifier_child_cannot_write_files` —— child registry 不含 `file_edit` —— 子模型若 hallucinate 调它 → tool not found error。

## 验收 / Acceptance

- `uv run pytest aether/tests/agents/types/test_verifier_definition.py aether/tests/agents/test_verifier_gate_reminder.py aether/tests/integration/test_verifier_end_to_end.py` 全绿。
- `uv run pytest aether/tests/` 既有 test 零回归。
- `uv run pyright` 零新增告警。
- **关键手测**：在 Aether 仓库里 prompt "把 Sprint 10 的 `default_builder.py` / `agent_tool.py` / `agent.py` 三个文件里的 `inherit_tools` 行为统一改为 deny-list" → 模型必须在 final answer 前 spawn 一个 Verifier 子 agent 跑 `pyright` + `pytest` 并拿到 PASS。
- **观测**：跑过的 session 在 log 里 grep `verifier_invoked=true` 应能找到对应 spawn 记录；`verifier_gate_reminder_sent` 出现次数 ≤ session edit count 除以 threshold。

## 不在本 PR / Deferred to other PRs

- **硬 gate**：不允许模型在未拿到 Verifier PASS 时给出 final answer —— 这需要解析模型的 stop token / final reply 并 short-circuit；OCC 也没做硬 gate，本 sprint 跟随。
- **多 Verifier round**：当前一次 PR 只描述 "spawn 一个 Verifier, on FAIL fix and resume"；OCC 还有 "并行 spawn 多个 Verifier 投票" 实验，Aether 不做。
- **Verifier 的 prompt cache 复用**：理论上 Verifier 与父共享系统 prompt 大头能省 token；当前实现按 child engine 默认从零开始 —— 优化留给 Sprint 12+。
- **TUI 端 "Verifier verdict" 渲染**：当前 Verifier 输出走普通 ChatMessage 路径；要不要加专门的视觉标识（绿 PASS / 红 FAIL）留待后续。
