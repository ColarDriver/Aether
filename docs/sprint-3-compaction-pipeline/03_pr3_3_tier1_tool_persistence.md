# PR 3.3 — Tier 1：per-tool 工具结果持久化

> **角色**：流水线第 1 层。最低成本（纯本地写盘）的压缩手段。
> 与 PR 3.1 / PR 3.2 平行落地（不依赖它们的具体字段，只需要 PR 3.1 的
> `metadata["compaction"]` 计数器约定）。

## 一、目标

1. 让 6 个内置工具（shell / read_file / grep / glob / list_dir / write_file）
   各自实现"超阈值时把完整结果写到磁盘 + 上下文只留前 N 字符 + 引用提示"。
2. 提供共用工具函数 `runtime/tool_result_storage.py`，避免每个工具重复实现 spill 逻辑。
3. 让模型能通过 `read_file <spilled-path>` 读回完整结果。
4. 把 spill 计数累加到 `context.metadata["tier1_spilled_count"]`，进入 EngineResult。

## 二、现状分析

### 2.1 当前工具的输出处理

[`backend/harness/aether/tools/builtins/shell.py`](../../backend/harness/aether/tools/builtins/shell.py)
等 6 个 builtin 现在都是把全部 stdout/输出原样塞回 `ToolResult.content`：

```python
# shell.py 当前
def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
    proc = subprocess.run(...)
    output = proc.stdout + proc.stderr
    return ToolResult(call_id=call.id, content=output, is_error=...)
```

问题：
- 用户跑 `find / -type f` 可能产出 100MB 输出，全部塞进 ToolResult，
  下次调 LLM 时整个 100MB 进 prompt → 直接 413 + 撑爆窗口。
- 模型读了一万行的文件，整文件进入对话，后续 5 轮都拖着这 50000 token。

### 2.2 claude-code 的对应做法

[`tools/BashTool/BashTool.tsx:728`](../../tmp/claude-code-references)：

> Large output: the file on disk has more than getMaxOutputLength() bytes.
> stdout already contains the first chunk (from getStdout()). Copy the
> output file to the tool-results dir so the model can read it via FileRead.

也就是说 BashTool 的 stdout 本来就是写到 disk file 的（subprocess 重定向），
超阈值时把那个 file 复制到 tool-results 目录，模型用 FileRead 读回。
PowerShellTool 同样模式。

我们的 6 个 builtin 各自机制不同（shell 用 subprocess，read_file 直接 read，
grep / glob 用 Python ripgrep / glob 库），所以**统一服务**不合适——
你已经选了 per_tool 方案。

## 三、设计

### 3.1 `runtime/tool_result_storage.py` — 共用 spill 工具

新文件 [`backend/harness/aether/runtime/tool_result_storage.py`](../../backend/harness/aether/runtime/tool_result_storage.py)：

```python
"""Disk-spill helpers for large tool results.

Each built-in tool decides ITS OWN threshold and IF/WHEN to spill
(per-tool design choice — see docs/sprint-3-compaction-pipeline/
03_pr3_3_tier1_tool_persistence.md). This module just provides the
shared file-naming, path-resolution, and write logic.

Spilled files live under:
    <spill_dir>/<session_id>/<call_id>.<ext>

Default spill_dir = ~/.aether/tool_results, override via
EngineConfig.tool_result_spill_dir.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True, frozen=True)
class SpillReceipt:
    """Outcome of a successful spill."""

    path: Path
    full_chars: int
    preview_chars: int

    @property
    def relative_hint(self) -> str:
        """Path string suitable for inclusion in the model-visible preview.

        Uses ~ for HOME-relative paths so users see meaningful hints
        (e.g. '~/.aether/tool_results/<sid>/<cid>.txt' instead of an
        absolute /home/user/... path that varies per environment).
        """
        try:
            home = Path.home()
            rel = self.path.relative_to(home)
            return f"~/{rel}"
        except ValueError:
            return str(self.path)


def resolve_spill_dir(
    *,
    session_id: str,
    config_dir: Optional[Path] = None,
) -> Path:
    """Return the directory for spilled results in this session.

    Creates parents on first call. Idempotent.
    """
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
    """Write *content* to a session-scoped file and return the receipt.

    Best-effort: failures (disk full, permission denied) raise OSError to
    the caller — the caller should catch and fall back to plain truncation.

    Args:
        content: full text to persist
        session_id: used to namespace files per session
        call_id: tool-call ID, becomes filename stem
        extension: typically 'txt', could be 'json' / 'md' for grep / etc
        config_dir: override default ~/.aether/tool_results
        preview_chars: how many chars from `content` will end up visible in
            the model's context (used for SpillReceipt.preview_chars metric only)

    Returns:
        SpillReceipt with path and metrics.
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

    Format (model-facing, intentionally machine-readable):

        ... [output truncated: {full_chars} chars{ /  N lines}{ / M bytes}
            saved to {hint} — use read_file to retrieve the full content] ...

    Stable enough for the model to learn the pattern; explicit about WHERE
    the full result lives.
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

    Called opportunistically (e.g. from /clear or session pruning).
    Returns count of files removed. Failures silently logged.
    """
    spill_dir = resolve_spill_dir(session_id=session_id, config_dir=config_dir)
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

### 3.2 6 个 builtin 各自的 spill 策略

每个工具独立决定 `MAX_RESULT_CHARS` 阈值（按典型输出大小调整）和触发条件。

**通用模板**（所有工具适用）：

```python
from aether.runtime.tool_result_storage import (
    spill_to_disk,
    build_truncation_notice,
)

class FooTool(ToolExecutor):
    MAX_RESULT_CHARS = 40_000  # tool-specific

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        full_output = self._do_work(...)

        # === PR 3.3 spill 段 ===
        if (
            context.config.tool_result_spill_enabled
            and len(full_output) > self.MAX_RESULT_CHARS
        ):
            preview = full_output[: self.MAX_RESULT_CHARS]
            try:
                receipt = spill_to_disk(
                    full_output,
                    session_id=context.session_id,
                    call_id=call.id,
                    extension="txt",
                    config_dir=context.config.tool_result_spill_dir,
                    preview_chars=len(preview),
                )
                notice = build_truncation_notice(
                    receipt,
                    full_lines=full_output.count("\n") + 1,
                )
                content = preview + notice
                # observability: bump tier1 counter
                context.metadata["tier1_spilled_count"] = (
                    int(context.metadata.get("tier1_spilled_count", 0)) + 1
                )
            except OSError as exc:
                # Disk full / permission: fall back to plain truncation.
                content = preview + (
                    f"\n\n... [output truncated: {len(full_output)} chars total, "
                    f"could not spill to disk: {exc}] ..."
                )
        else:
            content = full_output

        return ToolResult(call_id=call.id, content=content, is_error=...)
```

**各工具阈值**：

| 工具 | `MAX_RESULT_CHARS` | 备注 |
|---|---|---|
| shell | 40_000 | 长 stdout（约等于 ~10000 行 4 字符行）即触发；`find` / `cat` / `grep` 巨量输出 |
| read_file | 60_000 | 用户主动读文件，给宽松阈值；超阈值（约 1500 行）才 spill |
| grep | 30_000 | grep 通常上下文 + 行号，30k 已经能装下数百匹配 |
| glob | 20_000 | glob 是路径列表，20k 装得下数千路径 |
| list_dir | 20_000 | 列目录通常很小；只有特殊大目录才触发 |
| write_file | **不 spill** | 输出是简短的 success message；本身不大 |

**特殊处理**：

- **read_file**：如果用户读的就是一个先前 spilled 的文件（path 在 `~/.aether/tool_results/`）,
  直接返回完整内容**不再 spill**（避免循环）。
  ```python
  if path.is_relative_to(Path.home() / ".aether" / "tool_results"):
      return ToolResult(call_id=call.id, content=full_output, is_error=False)
  ```
- **shell**：把退出码、stderr 单独保留在 preview 段开头，让模型即便看不到完整 stdout
  也能判断成功/失败：
  ```python
  preview = (
      f"[exit_code={proc.returncode}, stderr_lines={proc.stderr.count(chr(10))}]\n"
      + preview
  )
  ```
- **grep**：`MAX_RESULT_CHARS` 触发后**优先按行截断**而不是按字符（避免最后半行截断）：
  ```python
  cutoff = preview.rfind("\n", 0, self.MAX_RESULT_CHARS)
  if cutoff > 0:
      preview = preview[:cutoff]
  ```

### 3.3 `EngineConfig` 新字段

```python
# Sprint 3 / PR 3.3: per-tool result spill master switch. When False,
# tools fall back to plain truncation with the historical "[output
# truncated]" marker — used as an emergency rollback if the spill
# directory becomes problematic (filled disk, NFS permissions, etc).
tool_result_spill_enabled: bool = True

# Sprint 3 / PR 3.3: directory for spilled tool results. Default
# ~/.aether/tool_results/<session_id>/<call_id>.<ext>. Override
# via env var or CLI to redirect to a faster local disk on systems
# where $HOME is on slow networked storage.
tool_result_spill_dir: Path | None = None  # None = use default
```

### 3.4 `TurnContext` 已具备的字段

`TurnContext` 已经携带 `session_id` 和 `metadata`。`config` 字段需要确认；
如果 context 不直接持 `EngineConfig`，则改为通过 `context.metadata["_engine_config"]`
注入（PR 3.1 阶段顺便加）。

实施时确认：

```bash
grep -n "class TurnContext" backend/harness/aether/runtime/contracts.py
```

如缺 `config` 字段，加：

```python
@dataclass
class TurnContext:
    session_id: str
    iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)
    # === PR 3.3 新增 ===
    config: "EngineConfig | None" = None  # AgentEngine 在创建时注入
```

### 3.5 `tier1_spilled_count` 进 EngineResult

PR 3.1 已经在 `_build_result` 预留了：

```python
metadata["compaction"] = {
    "tier1_spilled_count": context.metadata.get("tier1_spilled_count", 0),
    ...
}
```

本 PR 只需要在工具 execute 时累加 `context.metadata["tier1_spilled_count"]`，
PR 3.1 自动暴露到 `result.metadata["compaction"]["tier1_spilled_count"]`。

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/runtime/tool_result_storage.py` | **新文件** | `SpillReceipt` + `spill_to_disk` + `build_truncation_notice` + `cleanup_session_spills` | ~150 |
| `backend/harness/aether/tools/builtins/shell.py` | 修改 | 加 `MAX_RESULT_CHARS=40_000` + spill 段 + exit_code/stderr preview | ~30 |
| `backend/harness/aether/tools/builtins/read_file.py` | 修改 | 加 `MAX_RESULT_CHARS=60_000` + spill 段 + 防递归 spill 检查 | ~30 |
| `backend/harness/aether/tools/builtins/grep.py` | 修改 | 加 `MAX_RESULT_CHARS=30_000` + spill 段 + 行边界截断 | ~30 |
| `backend/harness/aether/tools/builtins/glob.py` | 修改 | 加 `MAX_RESULT_CHARS=20_000` + spill 段 | ~25 |
| `backend/harness/aether/tools/builtins/list_dir.py` | 修改 | 加 `MAX_RESULT_CHARS=20_000` + spill 段 | ~25 |
| `backend/harness/aether/tools/builtins/write_file.py` | 不改 | 输出本身很小 | 0 |
| `backend/harness/aether/runtime/contracts.py` | 修改 | `TurnContext` 加 `config` 字段（如果还没） | 0-5 |
| `backend/harness/aether/agents/core/agent.py` | 修改 | `_prepare_turn_entry` 把 `EngineConfig` 注入 `context.config` | 1-3 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 `tool_result_spill_enabled` + `tool_result_spill_dir` | ~20 |
| `backend/harness/aether/tests/test_tool_result_spill.py` | **新文件** | 见 § 五.1 | ~350 |
| `backend/harness/aether/tests/test_tool_result_storage.py` | **新文件** | 见 § 五.2 | ~150 |

## 五、测试用例（详细）

### 5.1 `test_tool_result_spill.py`

**测试组 A：shell spill**

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | shell 输出 5000 字符（< 40k） | `result.content` 不含 truncation notice，content 完整 |
| **T-A2** | shell 输出 50000 字符（> 40k） | `result.content` ≤ 40k + notice；磁盘上 `<spill_dir>/<sid>/<call_id>.txt` 存在且 ≈ 50k |
| **T-A3** | shell 输出 50000 字符且 spill_enabled=False | content 截断到 40k 但**不**写盘；notice 是 plain truncation 文本 |
| **T-A4** | shell 失败（returncode != 0）+ 长 stderr | preview 开头有 `[exit_code=N, stderr_lines=M]`，模型可见错误信息 |
| **T-A5** | spill 写盘失败（mock OSError） | 不 crash；content 退回 plain truncation；notice 含错误说明 |
| **T-A6** | tier1_spilled_count 累加 | 第二次 spill 后 `context.metadata["tier1_spilled_count"] == 2` |

**测试组 B：read_file spill**

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | read_file 读 10KB 文件 | content 完整，不 spill |
| **T-B2** | read_file 读 100KB 文件（> 60k） | spill；preview 60k；notice 显示 saved-to 路径 |
| **T-B3** | read_file 读取的是已 spilled 的文件 (`~/.aether/tool_results/.../*`) | **不**再 spill（防递归），返回完整内容 |
| **T-B4** | 同一 turn 内 read_file 同一大文件 5 次 | 每次都 spill 但都用同一个 call_id 不同（call_id 随机），共 5 个 spill 文件 |

**测试组 C：grep spill 行边界**

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | grep 输出 35000 字符且最后一行未在 30000 处结束 | preview 在 < 30000 的最近一个 `\n` 处截断；不出现半行 |
| **T-C2** | grep 输出每行都很长，第一行就 50000 字符 | preview = 第一行的前 30000 字符 + notice（无法找到 `\n`） |

**测试组 D：glob / list_dir 简单情况**

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | glob 匹配 5000 个文件，路径合计 25000 字符（> 20k） | spill |
| **T-D2** | list_dir 列大目录（> 20k 字符） | spill |
| **T-D3** | list_dir 普通目录（< 20k） | 不 spill |

**测试组 E：write_file 不 spill**

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | write_file 写 100KB 内容 | success message 很短，不 spill |

**测试组 F：跨 session 隔离**

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | session A 和 session B 各 spill 一次 | 两个文件分别在 `<dir>/A/...` 和 `<dir>/B/...` |
| **T-F2** | 两个 session 并发 spill 同一 call_id（理论上不可能但防御）| 文件路径不同（session_id 隔离），不冲突 |

**测试组 G：notice 文本稳定性**

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | notice 文本含三段：`output truncated:`、`saved to <hint>`、`use read_file` | 全部存在，且顺序固定（模型学习模式的稳定性） |
| **T-G2** | hint 路径用 `~/` 前缀（HOME-relative） | 不暴露绝对路径，环境无关 |

### 5.2 `test_tool_result_storage.py`

**测试组 H：SpillReceipt 行为**

| ID | 场景 | 验证 |
|---|---|---|
| **T-H1** | 创建 SpillReceipt(path=Path('/home/x/.aether/tool_results/s1/c1.txt'), ...) 在 HOME 下 | `relative_hint == "~/.aether/tool_results/s1/c1.txt"` |
| **T-H2** | path 不在 HOME 下 | `relative_hint == str(path)`（绝对路径） |

**测试组 I：spill_to_disk**

| ID | 场景 | 验证 |
|---|---|---|
| **T-I1** | 第一次调用，目录不存在 | `mkdir(parents=True, exist_ok=True)` 自动创建 |
| **T-I2** | 同一 (session_id, call_id) 调两次 | 第二次覆盖第一次的内容 |
| **T-I3** | content 是 unicode（中文 / emoji） | 写入文件用 utf-8，可正确读回 |
| **T-I4** | spill_dir 给到一个不可写路径 | 抛 OSError（caller 处理） |

**测试组 J：build_truncation_notice**

| ID | 场景 | 验证 |
|---|---|---|
| **T-J1** | 只给 full_chars | notice 只含 chars 度量 |
| **T-J2** | 给 full_chars + full_lines | notice 含 chars + lines |
| **T-J3** | 给 full_chars + full_lines + full_bytes | notice 含全部三个 |

**测试组 K：cleanup_session_spills**

| ID | 场景 | 验证 |
|---|---|---|
| **T-K1** | 5 个文件，3 个 mtime 超过阈值 | 删除 3 个，返回 3 |
| **T-K2** | 全部都是新文件 | 0 |
| **T-K3** | 目录不存在 | 不抛错，返回 0 |
| **T-K4** | 某个文件删除失败（mock OSError） | 跳过该文件继续删其他，返回成功删除的数量 |

## 六、验收门

- [ ] 所有新测试 green
- [ ] 既有 `test_builtin_tools.py` 的 26 个 case 不回归
- [ ] 真实跑 `shell: find / -type f 2>/dev/null` 看上下文 < 5KB（含 notice）
- [ ] 真实跑 `read_file: <100KB 大文件>` 看上下文 ≤ 60KB
- [ ] 模型用 `read_file <spilled-path>` 能读回完整内容
- [ ] `result.metadata["compaction"]["tier1_spilled_count"]` 与实际 spill 次数一致

## 七、回滚开关

- `tool_result_spill_enabled=False`：所有工具退回 plain truncation 行为（不写盘）
- 如需完全 revert：删除 `runtime/tool_result_storage.py` + 6 个 builtin 的 spill 段

## 八、实施顺序（建议 2 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. 新文件 `runtime/tool_result_storage.py` | 1.5h | 完整工具函数 |
| 2. `tests/test_tool_result_storage.py` | 1.5h | 测试组 H-K |
| 3. `runtime/contracts.py` 加 `config` 字段（如缺） | 30min | 注入路径 |
| 4. `agents/core/agent.py` 注入 `context.config` | 30min | 一处调整 |
| 5. `config/schema.py` 加新字段 | 30min | 含详细注释 |
| 6. 改 6 个 builtin（shell/read_file/grep/glob/list_dir） | 3h | 平均每个 30 分钟 |
| 7. `tests/test_tool_result_spill.py` | 4h | 测试组 A-G（约 25 case） |
| 8. 真实场景 sanity check | 1h | shell + read_file 各一次 |
| 9. 既有测试回归 | 1h | unittest discover |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| `~/.aether/tool_results` 占用越来越多磁盘 | 中 | `cleanup_session_spills` 默认 7 天清理；可在 `/clear` 时调用 |
| 模型不会用 read_file 读 spilled 文件 | 中 | notice 文本明确写"use read_file to retrieve"；测试组 G 验证模式稳定 |
| 跨 session 撞 call_id（理论不可能） | 极低 | 路径含 session_id 隔离，物理不冲突 |
| spill 文件包含敏感数据未删除 | 中 | 文档提示 `~/.aether/tool_results` 内容应被视为会话临时文件；session 删除时应级联删除（PR 后续可加） |
| read_file 读 spilled 文件再 spill 自己（无限循环） | 已防御 | T-B3 测试覆盖 |
| 工具阈值设得太低反而频繁 spill | 中 | 阈值默认偏宽松（4-6 万字符），实施后看真实数据再调 |

## 十、与后续 PR 的接合

- **PR 3.4** 的 `CompactionPipeline` 计算 `current_tokens` 时直接受益：
  spilled 后 ToolResult 占用降到几百字符，messages 的 token 总数大幅下降，
  Tier 5 触发概率显著降低。
- **PR 3.5** 的 `Microcompact` 清理 tool_result 时，对 spilled 的 ToolResult
  也只看 content（已经是 preview + notice），效果一致。
- **CLI** 可以用 `metadata["compaction"]["tier1_spilled_count"]` 在 footer 显示
  `↻ 3 results spilled` 信息。
