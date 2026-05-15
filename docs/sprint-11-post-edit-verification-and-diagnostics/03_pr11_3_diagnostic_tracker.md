# PR 11.3 — DiagnosticTracker Service

## 目标 / Goal

新建 `aether/runtime/diagnostics/`（3 个文件）实现 `DiagnosticTracker`：维护每文件的 *diagnostic baseline*，提供 `before_file_edited(path)` 与 `get_new_diagnostics()` 两个核心 API，并与已有的 `LSPManager` 对接。

参考 `open-claude-code/src/services/diagnosticTracking.ts:1-330`（约 11 KB）。

本 PR 不接入任何 tool —— 那些是 PR 11.4 的活；本 PR 只交付一个 *独立可单测* 的服务，签名稳定后才让上游消费。

## 当前问题 / Current Problem

### 1. 没有 baseline 概念

`aether/runtime/resources/lsp_manager.py:42` 提供的是 *"按需返回 LSPClient"* 的池子；它不记得"这次编辑前文件里有什么诊断"。如果模型改一行代码新引入一个 NameError，没有 baseline 就无法"只看新诊断"——直接拉所有诊断会把仓库里既存的所有 lint 警告也灌回给模型，模型分不清"我搞砸的"和"原本就这样"。

### 2. LSP didChange / didSave 没人触发

`aether/runtime/resources/lsp_client.py` 暴露的方法（grep `def `）：

```python
def initialize(...)
def start(...)
def stop(...)
def send_request(method, params, timeout)
def send_notification(method, params)
# (low-level JSON-RPC primitives)
```

—— 全是低层 JSON-RPC；没有"我刚改完一个文件请你重新分析"的语义包装。一次 LSP 编辑通知需要：

1. `textDocument/didOpen`（如果是第一次见这个文件）
2. `textDocument/didChange` + 新内容
3. `textDocument/didSave`
4. 等 `textDocument/publishDiagnostics` 回推（push）

`LSPClient` 已经能发 notification 与监听 server-pushed message，但没有人组装这 4 步成一个 "save & wait for diagnostics" 的 high-level call。`open-claude-code/src/services/lsp/LSPManager.ts:changeFile`/`saveFile` 干的就是这件事。

### 3. `LSPTool` 与 baseline 互不相关

`tools/builtins/lsp.py:94` 的 `LSPTool` 是模型用来 *显式查询* 当前文件诊断的；它每次都问 LSP "现在有什么诊断"。不存 baseline，不 diff。本 PR 的 tracker 与它**并存**：tool 仍然给模型自查，tracker 则是后台采集供 attachment 注入。

## 改动 / Changes

### 1. 新目录 `aether/runtime/diagnostics/`

```
aether/runtime/diagnostics/
├── __init__.py           # public API
├── tracker.py            # DiagnosticTracker (the main service)
└── types.py              # Diagnostic, DiagnosticFile dataclasses
```

#### `types.py`

```python
"""Public diagnostic dataclasses, frozen for safe sharing across threads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Severity = Literal["error", "warning", "info", "hint"]


@dataclass(slots=True, frozen=True)
class Diagnostic:
    """A single LSP diagnostic, language-server-agnostic."""

    message: str
    severity: Severity
    line: int           # 1-based, matches LSP Range.start.line + 1 for display
    column: int         # 1-based
    source: str         # e.g. "pyright", "tsserver", "ruff"
    code: str | None = None


@dataclass(slots=True, frozen=True)
class DiagnosticFile:
    """Diagnostics for one file."""

    path: Path
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)


__all__ = ["Diagnostic", "DiagnosticFile", "Severity"]
```

#### `tracker.py`

```python
"""Tracks diagnostic state across edits.

Parity with open-claude-code ``src/services/diagnosticTracking.ts``.

Flow:

  1. tool decides "I'm about to edit ``foo.py``" → ``before_file_edited(path)``
     freezes the current diagnostic list as the baseline for that file.
  2. tool finishes; engine fires ``post_tool_use`` (PR 11.2) → notifies LSP
     via :class:`LSPManager` (added in this PR — see §3) and waits up to
     ``settle_timeout_ms`` for diagnostics to arrive.
  3. next LLM call's pre-message attachment pass (PR 11.5) calls
     ``get_new_diagnostics()`` → returns only diagnostics that did NOT
     exist in the baseline.  Subsequent calls keep returning the same
     diff until ``clear_delivered(path)`` is invoked.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from pathlib import Path
from threading import Lock

from aether.runtime.diagnostics.types import Diagnostic, DiagnosticFile
from aether.runtime.resources.lsp_manager import LSPManager


_NO_DIAGNOSTICS: tuple[Diagnostic, ...] = ()


class DiagnosticTracker:
    """Per-engine diagnostic tracker.

    Thread-safe via a single :class:`threading.Lock`; expected lock
    holding time is microseconds (dict lookup + set difference).
    """

    def __init__(
        self,
        lsp_manager: LSPManager | None,
        *,
        settle_timeout_ms: int = 1500,
    ) -> None:
        self._lsp = lsp_manager
        self._settle_timeout_ms = settle_timeout_ms
        self._lock = Lock()
        # path → tuple[Diagnostic, ...]
        self._baselines: dict[Path, tuple[Diagnostic, ...]] = {}
        # path → tuple[Diagnostic, ...] already delivered to model (dedup)
        self._delivered: dict[Path, set[Diagnostic]] = {}

    @property
    def enabled(self) -> bool:
        return self._lsp is not None

    # --- baseline management ---

    def before_file_edited(self, path: Path) -> None:
        """Snapshot current diagnostics as the baseline for *path*.

        No-op when ``enabled is False``.  Idempotent within a single
        edit cycle (last call wins).
        """
        if self._lsp is None:
            return
        current = self._fetch_blocking(path)
        with self._lock:
            self._baselines[path] = current

    async def notify_file_changed(self, path: Path, content: str) -> None:
        """Equivalent of ``LSPManager.changeFile`` + ``saveFile`` in OCC.

        Fire-and-forget by intent — exceptions are logged and swallowed
        so a flaky language server never breaks file editing.
        """
        if self._lsp is None:
            return
        try:
            await self._lsp.change_file(path, content)
            await self._lsp.save_file(path)
        except Exception:
            # Logger injected at construction in real wiring; for now
            # we rely on LSPManager's own structured log path.
            pass

    # --- diff + delivery ---

    def get_new_diagnostics(
        self,
        paths: Iterable[Path] | None = None,
    ) -> list[DiagnosticFile]:
        """Return diagnostics introduced since the last baseline.

        If *paths* is None, returns new diagnostics for every path that
        has a baseline.  Diagnostics already returned by a prior call
        are suppressed (one delivery per (path, diagnostic) pair).
        """
        if self._lsp is None:
            return []

        targets: list[Path]
        with self._lock:
            targets = list(paths) if paths is not None else list(self._baselines.keys())

        out: list[DiagnosticFile] = []
        for path in targets:
            current = self._fetch_blocking(path)
            baseline = self._baselines.get(path, _NO_DIAGNOSTICS)
            baseline_set = set(baseline)
            with self._lock:
                already = self._delivered.setdefault(path, set())
                new = tuple(
                    d for d in current
                    if d not in baseline_set and d not in already
                )
                if new:
                    already.update(new)
            if new:
                out.append(DiagnosticFile(path=path, diagnostics=new))
        return out

    def clear_delivered(self, path: Path | None = None) -> None:
        with self._lock:
            if path is None:
                self._delivered.clear()
            else:
                self._delivered.pop(path, None)

    # --- internals ---

    def _fetch_blocking(self, path: Path) -> tuple[Diagnostic, ...]:
        """Pull the latest diagnostics, bounded by ``settle_timeout_ms``."""
        if self._lsp is None:
            return _NO_DIAGNOSTICS
        deadline = time.perf_counter() + self._settle_timeout_ms / 1000.0
        # Run the async LSP query inside whichever loop is hosting us.
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        coro = self._lsp.pull_diagnostics(path, deadline=deadline)
        if loop.is_running():
            # Caller (PostToolUse hook) is sync — schedule and short-block.
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                return tuple(fut.result(timeout=self._settle_timeout_ms / 1000.0))
            except Exception:
                return _NO_DIAGNOSTICS
        return tuple(loop.run_until_complete(coro))


__all__ = ["DiagnosticTracker"]
```

#### `__init__.py`

```python
from aether.runtime.diagnostics.tracker import DiagnosticTracker
from aether.runtime.diagnostics.types import Diagnostic, DiagnosticFile, Severity

__all__ = [
    "Diagnostic",
    "DiagnosticFile",
    "DiagnosticTracker",
    "Severity",
]
```

### 2. 扩展 `LSPManager`

`aether/runtime/resources/lsp_manager.py` 新增三个高层方法，**全部 async**（底层 `LSPClient` 通信本就基于事件循环）：

```python
async def change_file(self, path: Path, content: str) -> None:
    """Send ``textDocument/didChange`` (with synthetic version bump).

    If the document has not been opened yet, send ``didOpen`` first.
    No-op if no LSP server is registered for this file's language.
    """
    client = self.get_client_for(path)
    if client is None:
        return
    if not client.is_open(path):
        await client.did_open(path, content)
        return
    await client.did_change(path, content)


async def save_file(self, path: Path) -> None:
    """Send ``textDocument/didSave`` — triggers most servers to re-lint."""
    client = self.get_client_for(path)
    if client is None:
        return
    await client.did_save(path)


async def pull_diagnostics(
    self,
    path: Path,
    *,
    deadline: float,
) -> list[Diagnostic]:
    """Wait for ``publishDiagnostics`` push for *path* until *deadline*.

    Returns the latest known diagnostic set.  If no push arrives by
    deadline, returns whatever was last pushed (possibly stale, possibly
    empty).  Never raises on timeout.
    """
    client = self.get_client_for(path)
    if client is None:
        return []
    return await client.wait_for_diagnostics(path, deadline=deadline)
```

对应在 `LSPClient` 上补 `did_open` / `did_change` / `did_save` / `wait_for_diagnostics` / `is_open` 五个 helper（JSON-RPC notification 已存在；只是包装）。

注意 import 路径：`LSPManager` 必须 *延迟* 引用 `Diagnostic`（放在方法签名而不是 type-only import），否则会形成 `lsp_manager → diagnostics.types → lsp_manager` 的循环。建议把 `Diagnostic` dataclass 放回 `runtime/resources/lsp_types.py`（NEW 小文件）并让 `diagnostics/types.py` re-export，或者用 `TYPE_CHECKING` guard。**实施时优先 re-export 方案**——它让 `LSPClient` 内部测试更自然。

### 3. Engine 装配点

`aether/agents/core/agent.py` 的 `__init__`（line 275 附近）新增可选参数 `diagnostic_tracker: DiagnosticTracker | None = None`；如果传入则保存到 `self._diagnostic_tracker`。

CLI 入口 `aether/cli/main.py`：

```python
from aether.runtime.diagnostics import DiagnosticTracker

# 已有 lsp_manager 构造之后
diagnostic_tracker = DiagnosticTracker(lsp_manager)
engine = AgentEngine(
    ...,
    diagnostic_tracker=diagnostic_tracker,
)
```

Gateway 路径 `aether/gateway/handlers/agent_methods.py` 同样补一行。

### 4. Subagent 透传

`aether/subagents/default_builder.py` 在 `AgentEngine(...)` 构造参数列表（line 53-69）新增：

```python
diagnostic_tracker=parent._diagnostic_tracker,
```

—— 与 `_skill_catalog` / `_lsp_manager` 共用同一份；子 agent 编辑文件时也走相同的 baseline / diff 路径，与父在同一个仓库里共享 LSP 状态。

## 测试 / Tests

新建 `aether/tests/runtime/diagnostics/test_tracker.py`：

- `test_no_lsp_means_disabled` —— `DiagnosticTracker(None)` → `.enabled is False`；所有 API 退化为 no-op，不抛错。
- `test_baseline_freezes_current_set` —— mock LSPManager 让 `pull_diagnostics` 返回 `[A]` → `before_file_edited(p)` → 之后 `pull_diagnostics` 返回 `[A, B]` → `get_new_diagnostics([p])` 返回仅含 `B` 的 `DiagnosticFile`。
- `test_get_new_diagnostics_dedup_within_session` —— 上面情景中第二次调 `get_new_diagnostics([p])` → 返回 `[]`（已 delivered）。
- `test_clear_delivered_lets_diagnostic_resurface` —— 同上后 `clear_delivered(p)` → 再调 → `B` 再次出现。
- `test_get_new_diagnostics_with_paths_none_walks_all_baselines` —— 两个文件均有 baseline → `paths=None` 同时返回两者新诊断。
- `test_fetch_timeout_returns_empty_not_raise` —— mock `pull_diagnostics` 永久 hang → `before_file_edited` 在 `settle_timeout_ms` 后返回；baseline 设为 `()`。
- `test_notify_file_changed_swallows_exceptions` —— mock `change_file` 抛 → `notify_file_changed` 不 propagate。

新建 `aether/tests/runtime/resources/test_lsp_manager_high_level.py`：

- `test_change_file_calls_did_open_when_new` —— 第一次 `change_file(p, content)` → client 收到 `did_open` 而非 `did_change`。
- `test_change_file_calls_did_change_when_open` —— 已 open 再调 → `did_change`。
- `test_save_file_sends_did_save` —— `save_file` → `did_save` notification 发出。
- `test_pull_diagnostics_returns_empty_when_no_client` —— 文件后缀无服务器（如 `.md`）→ 返回 `[]`。
- `test_pull_diagnostics_respects_deadline` —— mock client `wait_for_diagnostics` 阻塞 → 在 deadline 内返回 `[]`，不抛。

## 验收 / Acceptance

- `uv run pytest aether/tests/runtime/diagnostics/ aether/tests/runtime/resources/test_lsp_manager_high_level.py` 全绿。
- `uv run pyright` 零新增告警。
- 不依赖 LSP 二进制存在的所有 test 必须能在 CI 上跑过（mock 一切）；真实 pyright/tsserver 集成测试单独放到一个 marker `@pytest.mark.requires_lsp` 下，可选执行。
- `DiagnosticTracker(None)` 路径在 LSP 未安装的环境下完全静默（验证：在没装 pyright 的 docker image 里 `uv run aether` 编辑文件不抛错）。
- 性能：mock LSP 立即响应时一次 `before_file_edited` + `get_new_diagnostics` round-trip ≤ 5ms。

## 不在本 PR / Deferred to other PRs

- **`LSPClient.did_open` / `did_change` / `did_save` / `wait_for_diagnostics` 的 LSP 协议细节** —— 本 PR 仅给出签名与基本实现；如有协议合规性问题（如 versioned text doc identifier）独立小 PR 修。
- **诊断在多 LSP server 间合并** —— OCC 把多个 server（如 tsserver + eslint-language-server）的诊断 union 后给模型；Aether 当前每语言只接一个 server，多 server 合并留给 Sprint 12+。
- **Tool 接入**：PR 11.4 才让 `file_edit` / `write_file` 真正调用本 PR 的 API。
- **Attachment 注入**：PR 11.5 才让模型看到诊断。
