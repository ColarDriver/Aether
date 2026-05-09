# PR 3.5.1 — Spill 基础设施 + 6 现有工具升级

> **角色**：Sprint 3.5 流水线的第 0 层。所有后续工具 PR 都建立在本 PR 落地的
> `runtime/tool_result_storage.py` 之上。
>
> **替代关系**：本 PR 取代了 Sprint 3 原计划中的 PR 3.3。原 PR 3.3 文档保留为
> 历史参考（`docs/sprint-3-compaction-pipeline/03_pr3_3_tier1_tool_persistence.md`）。

## 一、目标

1. 建立通用的 spill 基础设施 — `runtime/tool_result_storage.py`，提供
   `SpillReceipt` / `spill_to_disk` / `build_truncation_notice` /
   `cleanup_session_spills` 四个公共原语。
2. 让 6 个内置工具（`shell` / `read_file` / `write_file` / `list_dir` /
   `grep` / `glob`）各自决定阈值和触发条件，超阈值时把完整结果写到磁盘 +
   上下文只留前 N 字符 + 引用提示。
3. 在 `EngineConfig` 加 `tool_result_spill_enabled` / `tool_result_spill_dir`
   两个配置项，提供回滚开关和路径覆盖。
4. 在 `_prepare_turn_entry` 把 `EngineConfig` 注入 `context.metadata["_engine_config"]`，
   工具通过 metadata 拿配置（不动 `TurnContext` dataclass 形状）。
5. 把 spill 计数累加到 `context.metadata["tier1_spilled_count"]`，
   通过 PR 3.1 已经预留的 `EngineResult.metadata["compaction"]["tier1_spilled_count"]` 暴露。

## 二、现状审视

### 2.1 现有工具的截断行为分歧

每个工具现在各自截断，且**形态不一致**：

| 工具 | 现有截断阈值 | 截断方式 | 模型可见的提示 |
|---|---|---|---|
| `shell` | 16 KB / stream（hard） | head + tail | `[... truncated ...]` 内联标记 |
| `read_file` | 256 KB | 直接抛 ToolError | `is_error=True` |
| `write_file` | 不截断 | — | — |
| `list_dir` | 无显式阈值 | — | 大目录直接全部返回 |
| `grep` | 无显式阈值 | — | ripgrep 自带的 `--max-count` |
| `glob` | 无显式阈值 | — | glob 自带的 `**` 匹配 |

问题：
* **不一致** — 同一个失控场景（输出 100MB），`shell` 截断、`read_file` 报错、`list_dir` 撑爆 prompt。
* **不可恢复** — 截断的内容丢了；模型如果想看完整必须重跑（不一定 deterministic）。
* **观测性差** — 没有统一的 spill 事件计数器，无法回答"这个 turn 有多少工具触发了截断"。

### 2.2 claude-code 的等价机制

claude-code 的 BashTool 把 stdout 直接重定向到磁盘文件（subprocess 写文件，进程读文件回内存）。
超过 `getMaxOutputLength()` 时，把那个文件复制到 tool-results 目录，
preview 用前 N 字符 + `... [output too long, see <file>] ...` 提示，
模型用 `FileRead` 读回完整内容。

我们的 6 个工具机制不同（subprocess 拿 stdout、open() 读文件、ripgrep 子进程、Python glob 库），
没法用一个"重定向到 file"的统一前置。所以采用 **per-tool 决定 + 共用 spill helper** 的设计。

## 三、设计

### 3.1 `runtime/tool_result_storage.py` — 共用 spill 工具

新文件 [`backend/harness/aether/runtime/tool_result_storage.py`](../../backend/harness/aether/runtime/tool_result_storage.py)：

```python
"""Disk-spill helpers for large tool results.

Sprint 3.5 / PR 3.5.1.  Each built-in (and future) tool decides its own
threshold and IF/WHEN to spill — this module just provides the shared
file-naming, path-resolution, write logic, and notice-text builder.

Spilled files live under:

    <spill_dir>/<session_id>/<call_id>.<ext>

Default ``spill_dir`` = ``~/.aether/tool_results``, override via
``EngineConfig.tool_result_spill_dir``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True, frozen=True)
class SpillReceipt:
    path: Path
    full_chars: int
    preview_chars: int

    @property
    def relative_hint(self) -> str:
        """Path string for the model-visible preview (HOME-relative when possible)."""
        try:
            home = Path.home()
            rel = self.path.relative_to(home)
            return f"~/{rel}"
        except ValueError:
            return str(self.path)


def resolve_spill_dir(
    *, session_id: str, config_dir: Optional[Path] = None
) -> Path:
    """Return the directory for spilled results in this session, creating it as needed."""
    base = config_dir or (Path.home() / ".aether" / "tool_results")
    target = base / session_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def spill_to_disk(
    content: str,
    *,
    session_id: str,
    call_id: str,
    extension: str = "txt",
    config_dir: Optional[Path] = None,
    preview_chars: int = 0,
) -> SpillReceipt:
    """Write content to a session-scoped file and return the receipt.

    Raises OSError on disk-full / permission denied; caller should catch
    and fall back to plain truncation.
    """
    spill_dir = resolve_spill_dir(session_id=session_id, config_dir=config_dir)
    target = spill_dir / f"{call_id}.{extension}"
    target.write_text(content, encoding="utf-8")
    return SpillReceipt(
        path=target,
        full_chars=len(content),
        preview_chars=preview_chars,
    )


def build_truncation_notice(
    receipt: SpillReceipt,
    *,
    full_lines: Optional[int] = None,
    full_bytes: Optional[int] = None,
) -> str:
    """Build the standard '... [truncated, see file] ...' notice.

    Format (intentionally machine-readable so models learn the pattern):

        ... [output truncated: {N} chars{ / M lines}{ / K bytes} saved to
            {hint} — use read_file to retrieve the full content] ...
    """
    parts = [f"{receipt.full_chars} chars"]
    if full_lines is not None:
        parts.append(f"{full_lines} lines")
    if full_bytes is not None:
        parts.append(f"{full_bytes} bytes")
    metrics = " / ".join(parts)
    return (
        f"\n\n... [output truncated: {metrics} saved to "
        f"{receipt.relative_hint} — use read_file to retrieve the full content] ..."
    )


def cleanup_session_spills(
    *,
    session_id: str,
    config_dir: Optional[Path] = None,
    max_age_seconds: int = 7 * 24 * 3600,
) -> int:
    """Remove session-scoped spill files older than max_age_seconds.

    Returns count of files removed. Failures silently logged (degrading to
    "best-effort" semantics — caller should never depend on a specific count).
    """
    spill_dir = resolve_spill_dir(session_id=session_id, config_dir=config_dir)
    if not spill_dir.exists():
        return 0
    now = time.time()
    removed = 0
    for child in spill_dir.iterdir():
        try:
            age = now - child.stat().st_mtime
            if age > max_age_seconds:
                child.unlink()
                removed += 1
        except OSError:
            continue
    return removed
```

### 3.2 工具侧通用模板

每个工具的 `execute` 在产出完整 output 后插入：

```python
from aether.runtime.tool_result_storage import (
    spill_to_disk,
    build_truncation_notice,
)


def _maybe_spill(
    self,
    full_output: str,
    *,
    call: ToolCall,
    context: TurnContext,
    extension: str = "txt",
    full_lines: int | None = None,
) -> str:
    """Common spill path. Returns content for ToolResult."""
    config = context.metadata.get("_engine_config")
    spill_enabled = bool(getattr(config, "tool_result_spill_enabled", True))
    spill_dir = getattr(config, "tool_result_spill_dir", None)

    if not spill_enabled or len(full_output) <= self.MAX_RESULT_CHARS:
        return full_output

    preview = full_output[: self.MAX_RESULT_CHARS]
    try:
        receipt = spill_to_disk(
            full_output,
            session_id=context.session_id,
            call_id=call.id,
            extension=extension,
            config_dir=spill_dir,
            preview_chars=len(preview),
        )
        notice = build_truncation_notice(receipt, full_lines=full_lines)
        context.metadata["tier1_spilled_count"] = (
            int(context.metadata.get("tier1_spilled_count", 0)) + 1
        )
        return preview + notice
    except OSError as exc:
        return preview + (
            f"\n\n... [output truncated: {len(full_output)} chars total, "
            f"could not spill to disk: {exc}] ..."
        )
```

考虑放在 `aether/tools/base.py`（作为 `ToolExecutor` 的 mixin / helper）方便复用。
**实施时确认**：如果 `ToolExecutor` 是 ABC 不便加方法，则放成模块级函数。

### 3.3 各工具的具体改动

#### 3.3.1 `shell.py`

* `MAX_RESULT_CHARS = 40_000`（取代现有 16KB head+tail，给 spill 更宽空间）
* 删除现有的 head+tail 截断逻辑，统一走 spill 路径
* 在 spill preview 段开头**保留**：`[exit_code=N, stderr_lines=M]\n` 让模型即便看不到完整 stdout 也能判断结果

```python
# 现有逻辑：
# stdout, _ = _truncate_with_head_and_tail(stdout, max_bytes)
# stderr, _ = _truncate_with_head_and_tail(stderr, max_bytes)
# combined = f"...{stdout}...{stderr}..."

# 新逻辑：
combined = f"$ {command}\n"
if proc.returncode != 0:
    combined += f"[exit_code={proc.returncode}, stderr_lines={proc.stderr.count(chr(10))}]\n"
combined += proc.stdout
if proc.stderr:
    combined += "\n--- stderr ---\n" + proc.stderr

content = self._maybe_spill(combined, call=call, context=context, full_lines=combined.count("\n"))
```

#### 3.3.2 `read_file.py`

* `MAX_RESULT_CHARS = 60_000`（用户主动读，给宽松阈值；约 1500 行）
* **防递归 spill** — 如果 `path` 在 `~/.aether/tool_results/` 下，直接返回完整内容不再 spill
* 现有的 256KB hard cap 改成"读取上限"（不再触发 ToolError），交给 spill 处理

```python
spilled_root = (Path.home() / ".aether" / "tool_results").resolve()
try:
    target = path.resolve()
    if target.is_relative_to(spilled_root):
        # 读 spilled 文件本身：完整返回，不再 spill（避免环）
        return ToolResult(call_id=call.id, content=full_content, ...)
except (ValueError, OSError):
    pass

content = self._maybe_spill(formatted_lines, call=call, context=context,
                             full_lines=line_count)
```

#### 3.3.3 `grep.py`

* `MAX_RESULT_CHARS = 30_000`
* spill 时优先按**行边界**截断（避免最后半行）：

```python
preview = full_output[: self.MAX_RESULT_CHARS]
cutoff = preview.rfind("\n", 0, self.MAX_RESULT_CHARS)
if cutoff > 0:
    preview = preview[:cutoff]
```

#### 3.3.4 `glob.py`

* `MAX_RESULT_CHARS = 20_000`（约 800 条 25 字符路径）
* 标准 spill 模板，无特殊处理

#### 3.3.5 `list_dir.py`

* `MAX_RESULT_CHARS = 20_000`
* 标准 spill 模板，无特殊处理

#### 3.3.6 `write_file.py`

* **不 spill**。输出是简短的 success message（`wrote N bytes to <path>`）
* 文档里明确说明"为什么不 spill"

### 3.4 `EngineConfig` 新字段

```python
# Sprint 3.5 / PR 3.5.1 — per-tool result spill controls.

# Master switch.  When False, all tools fall back to plain truncation
# behaviour with the legacy "[output truncated]" marker — emergency
# rollback if the spill directory becomes problematic (filled disk,
# read-only NFS, etc.).
tool_result_spill_enabled: bool = True

# Override directory for spilled results.  Default
# ``~/.aether/tool_results/<session_id>/<call_id>.<ext>``.  Set to a
# faster local disk on systems where $HOME is on slow networked storage.
tool_result_spill_dir: Path | None = None
```

### 3.5 把 `EngineConfig` 注入 `context.metadata`

`agents/core/agent.py::_prepare_turn_entry` 创建 `TurnContext` 时新增一行：

```python
context.metadata["_engine_config"] = self.config
```

加入 `_METADATA_INTERNAL_KEYS` 的 allowlist（内部字段，不暴露到 EngineResult.metadata['turn']）：

```python
_METADATA_INTERNAL_KEYS: frozenset[str] = frozenset({
    "usage_accumulator",
    "_iteration_budget_obj",
    "_engine_config",          # ← Sprint 3.5 / PR 3.5.1
})
```

### 3.6 暴露 `tier1_spilled_count` 到 EngineResult

PR 3.1 已经在 `_build_result` 写了：

```python
metadata["compaction"] = {
    "tier1_spilled_count": int(context.metadata.get("tier1_spilled_count", 0)),
    ...
}
```

本 PR 只需要在工具 execute 时累加 `context.metadata["tier1_spilled_count"]`，
PR 3.1 自动暴露到 `result.metadata["compaction"]["tier1_spilled_count"]`。

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/runtime/tool_result_storage.py` | **新文件** | spill 全套原语 | ~150 |
| `backend/harness/aether/tools/base.py` | 修改 | 给 `ToolExecutor` 加 `_maybe_spill` mixin（或独立模块函数） | ~30 |
| `backend/harness/aether/tools/builtins/shell.py` | 修改 | 删 head+tail 截断；走 spill；preview 段加 exit_code 头 | ~40 净增 |
| `backend/harness/aether/tools/builtins/read_file.py` | 修改 | 防递归 spill；取消 256KB hard cap | ~30 净增 |
| `backend/harness/aether/tools/builtins/grep.py` | 修改 | 行边界截断 + spill | ~25 净增 |
| `backend/harness/aether/tools/builtins/glob.py` | 修改 | 标准 spill | ~20 净增 |
| `backend/harness/aether/tools/builtins/list_dir.py` | 修改 | 标准 spill | ~20 净增 |
| `backend/harness/aether/tools/builtins/write_file.py` | 不改 | 文档说明不 spill 的原因 | 0（注释） |
| `backend/harness/aether/agents/core/agent.py` | 修改 | `_prepare_turn_entry` 注入 `_engine_config`；`_METADATA_INTERNAL_KEYS` 加键 | ~5 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 `tool_result_spill_enabled` + `tool_result_spill_dir` | ~25（含注释） |
| `backend/harness/aether/tests/test_tool_result_storage.py` | **新文件** | 见 § 五.1 | ~200 |
| `backend/harness/aether/tests/test_tool_result_spill.py` | **新文件** | 见 § 五.2 | ~400 |

## 五、测试用例

### 5.1 `test_tool_result_storage.py`（共用原语层）

**测试组 A：SpillReceipt**

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | path 在 HOME 下 | `relative_hint == "~/.aether/tool_results/sid/cid.txt"` |
| **T-A2** | path 不在 HOME 下（如 `/tmp/...`） | `relative_hint == str(path)` 绝对路径 |
| **T-A3** | dataclass frozen 性质 | 修改字段抛 FrozenInstanceError |

**测试组 B：spill_to_disk**

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | 第一次调用，目录不存在 | 自动 mkdir 成功 |
| **T-B2** | 同一 (session_id, call_id) 调两次 | 第二次覆盖第一次 |
| **T-B3** | content 是 unicode（中文 / emoji） | utf-8 写入；可正确读回 |
| **T-B4** | spill_dir 不可写（mock chmod 0o500） | 抛 OSError |
| **T-B5** | extension 自定义为 "json" | 文件名是 `<call_id>.json` |
| **T-B6** | 不同 session_id | 文件分别在 `<dir>/A/...` 和 `<dir>/B/...` |

**测试组 C：build_truncation_notice**

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | 只给 full_chars | notice 含 `chars` |
| **T-C2** | 给 full_chars + full_lines | notice 含 `chars / N lines` |
| **T-C3** | 给全部三个 | notice 含 `chars / lines / bytes` |
| **T-C4** | notice 末尾必含 `use read_file to retrieve` | 模型学习模式的稳定性保证 |

**测试组 D：cleanup_session_spills**

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | 5 个文件，3 个 mtime > 7 天 | 删除 3，返回 3 |
| **T-D2** | 全部新文件 | 返回 0 |
| **T-D3** | 目录不存在 | 不抛错，返回 0 |
| **T-D4** | 一个文件 unlink 抛 OSError（mock）| 跳过该文件继续；返回成功删除数 |

### 5.2 `test_tool_result_spill.py`（per-tool 行为层）

**测试组 E：shell**

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | shell 输出 5000 字符 | 不 spill；content 完整；无 notice |
| **T-E2** | shell 输出 50000 字符 | spill；content ≤ 40k + notice；磁盘文件 ≈ 50k |
| **T-E3** | shell 失败（returncode=1）+ 长 stderr | preview 开头有 `[exit_code=1, stderr_lines=N]` |
| **T-E4** | spill_enabled=False + 50000 字符输出 | 截断到 40k；无磁盘文件；notice 是 plain truncation 文本 |
| **T-E5** | 写盘失败（mock OSError）| 不 crash；plain truncation；notice 含错误说明 |
| **T-E6** | tier1_spilled_count 累加 | 第 3 次 spill 后 count == 3 |

**测试组 F：read_file**

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | 读 10KB 文件 | content 完整 |
| **T-F2** | 读 100KB 文件 | spill；preview 60k；notice 含 saved-to 路径 |
| **T-F3** | 读 spilled 文件本身（path 在 `~/.aether/tool_results/`） | **不**再 spill（防递归）|
| **T-F4** | read_file 5 次同一大文件 | 5 个不同 spill 文件（call_id 唯一） |
| **T-F5** | 256KB hard cap 不再触发 | 旧的 `is_error=True` 路径不再走 |

**测试组 G：grep 行边界**

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | 输出 35000 字符，最后一行未在 30000 处结束 | preview 在 < 30000 的最近 `\n` 处截断 |
| **T-G2** | 第一行就 50000 字符 | preview = 第一行的前 30000 字符（无 `\n` 可找） |
| **T-G3** | 输出恰好 30000 字符且以 `\n` 结尾 | preview == full_output（不触发 spill） |

**测试组 H：glob / list_dir**

| ID | 场景 | 验证 |
|---|---|---|
| **T-H1** | glob 5000 文件，路径合计 25000 字符 | spill |
| **T-H2** | list_dir 大目录（> 20k） | spill |
| **T-H3** | list_dir 普通目录（< 20k） | 不 spill |

**测试组 I：write_file**

| ID | 场景 | 验证 |
|---|---|---|
| **T-I1** | write_file 写 100KB 内容 | success message 短，不 spill |
| **T-I2** | write_file 不读 `_engine_config` | 修改 ToolExecutor 后 write_file 仍正常 |

**测试组 J：跨 session 隔离**

| ID | 场景 | 验证 |
|---|---|---|
| **T-J1** | session A 和 B 各 spill 一次 | 两个文件分别在 `<dir>/A/...` 和 `<dir>/B/...` |
| **T-J2** | 同一 session 5 次 spill | 5 个文件都在同一子目录 |

**测试组 K：notice 文本稳定性**（claude-code 的 contract test 风格）

| ID | 场景 | 验证 |
|---|---|---|
| **T-K1** | notice 严格符合三段格式 | 含 `output truncated:`、`saved to <hint>`、`use read_file` 三段 |
| **T-K2** | hint 路径用 `~/` 前缀 | 不暴露绝对路径 |
| **T-K3** | 模型 mock 解析 notice，提取 spill path | 解析成功 |

## 六、验收门

* [ ] 所有新测试 green（约 30 case）
* [ ] 既有 `test_builtin_tools.py`（26 case）零回归
* [ ] 真实跑 `shell: find / -type f 2>/dev/null` 看上下文 < 50KB（含 notice）
* [ ] 真实跑 `read_file: <100KB 文件>` 看上下文 ≤ 60KB
* [ ] 模型用 `read_file <spilled-path>` 能读回完整内容
* [ ] `result.metadata["compaction"]["tier1_spilled_count"]` 与实际 spill 次数一致
* [ ] `~/.aether/tool_results` 在测试结束后能清理干净（CI 不留垃圾）

## 七、回滚开关

* `tool_result_spill_enabled=False`：所有工具退回 plain truncation
* 完全 revert：删除 `runtime/tool_result_storage.py` + 6 个 builtin 的 spill 段 +
  `_prepare_turn_entry` 的 `_engine_config` 注入

## 八、实施顺序（建议 2 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. 新文件 `runtime/tool_result_storage.py` | 1.5h | 4 个公共原语 |
| 2. `tests/test_tool_result_storage.py` | 1h | 测试组 A-D（约 15 case）|
| 3. `tools/base.py` 加 `_maybe_spill` mixin | 30min | 工具侧公共接入 |
| 4. `agents/core/agent.py` 注入 `_engine_config` + allowlist | 30min | 一处调整 |
| 5. `config/schema.py` 加新字段 | 30min | 含完整注释 |
| 6. 改 6 个 builtin（shell/read_file/grep/glob/list_dir） | 3h | 平均每个 ~30 分钟 |
| 7. `tests/test_tool_result_spill.py` | 4h | 测试组 E-K（约 25 case）|
| 8. 真实 smoke 验证 | 1h | shell + read_file 各跑一次 |
| 9. 既有测试回归 | 1h | unittest discover |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| `~/.aether/tool_results` 占用越来越多磁盘 | 中 | `cleanup_session_spills` 默认 7 天清理；可在 `/clear` 时调用（PR 3.5.1 不实现，留 hook） |
| 模型不会用 read_file 读 spilled 文件 | 中 | notice 文本明确 + 测试 T-K3 验证模式稳定 |
| 跨 session 撞 call_id（理论不可能） | 极低 | 路径含 session_id 隔离，物理不冲突 |
| spill 文件含敏感数据未删除 | 中 | 文档提示 `~/.aether/tool_results` 内容应被视为会话临时文件 |
| read_file 读 spilled 文件无限循环 | 已防御 | T-F3 测试覆盖 |
| 阈值设得太低导致频繁 spill | 中 | 阈值默认偏宽松；落地后看真实数据再调 |
| `_maybe_spill` mixin 与 ABC 不兼容 | 低 | fallback：改成 `tool_result_storage.maybe_spill_for_tool(...)` 函数 |

## 十、与后续 PR 的接合

* **PR 3.5.2-10**：所有新工具直接复用本 PR 的 `_maybe_spill`，新工具自带 spill 能力
* **Sprint 4 / PR 3.4 (Tier 5)**：`CompactionPipeline` 计算 `current_tokens` 时直接受益 —
  spilled 后 ToolResult 占用降到几百字符，messages 总 token 大幅下降
* **Sprint 4 / PR 3.5 (Tier 3)**：`Microcompact` 清理 tool_result 时，对 spilled 的也只看
  preview + notice（已经是小内容）
* **CLI footer**：可以从 `metadata["compaction"]["tier1_spilled_count"]` 渲染
  `↻ 3 results spilled` 信息
