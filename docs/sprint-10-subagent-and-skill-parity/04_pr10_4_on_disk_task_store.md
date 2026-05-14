# PR 10.4 — On-Disk Task Store

## 目标 / Goal

把所有 subagent 任务（同步 + 异步）的生命周期状态持久化到 `~/.aether/tasks/{task_id}/`，使得：

1. Gateway 重启后能继续读到历史任务（同步 case 在 result.json，async case 在 task.json + output.log）。
2. PR 10.5 的 async lifecycle 有地方写流式输出；PR 10.6 的 `TaskOutput` 工具有地方读取；PR 10.7 的 `SendMessage` 有地方排队。
3. 启动时扫一遍把"已经在跑但 gateway 重启了"的孤儿任务标记为 `KILLED`，避免无主 RUNNING 记录永远滞留。

参考：`open-claude-code/src/tasks/LocalAgentTask/LocalAgentTask.tsx`（task state shape + lifecycle）；`open-claude-code` 把任务状态放进 React `AppState`，Aether 把它落地磁盘以同时支持持久化和跨进程恢复。

## 当前问题 / Current Problem

`aether/subagents/manager.py:37-42`：

```python
self._stop_events: dict[str, threading.Event] = {}
self._active_children: dict[str, "object"] = {}
```

—— 全部 in-memory。`run_task` 返回的 `SubagentResult` 也只在 caller stack 上活着，gateway 重启即灰飞烟灭。

`aether/tools/builtins/task_output.py` 当前是 sprint 3.5 占位：返回 "not supported in sync mode"。

## 改动 / Changes

### 1. 新目录 `aether/runtime/tasks/`

#### `aether/runtime/tasks/__init__.py`

```python
from aether.runtime.tasks.contracts import TaskRecord, TaskStatus
from aether.runtime.tasks.store import TaskStore
from aether.runtime.tasks.recovery import RecoveryReport, recover_task_store

__all__ = [
    "TaskRecord",
    "TaskStatus",
    "TaskStore",
    "RecoveryReport",
    "recover_task_store",
]
```

#### `aether/runtime/tasks/contracts.py`

```python
"""Data contracts for the on-disk task store."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    KILLED = "killed"           # post-mortem mark for orphaned RUNNING records

    @property
    def is_terminal(self) -> bool:
        return self in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.INTERRUPTED,
            TaskStatus.KILLED,
        }


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    parent_session_id: str
    subagent_type: str
    prompt: str
    status: TaskStatus
    started_at: float                              # unix ts
    finished_at: Optional[float] = None
    last_heartbeat: float = 0.0
    agent_type_def_snapshot: dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    isolation: Optional[str] = None
    worktree_path: Optional[str] = None
    parent_task_id: Optional[str] = None
    child_depth: int = 1
    background: bool = False
    # progress counters
    tool_use_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    # final result fields
    summary: Optional[str] = None
    error: Optional[str] = None
    result_path: Optional[str] = None              # ~/.aether/tasks/{id}/result.json

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.value
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TaskRecord":
        data = {**data}
        data["status"] = TaskStatus(data["status"])
        return cls(**data)
```

#### `aether/runtime/tasks/store.py`

```python
"""TaskStore — directory layout under ~/.aether/tasks/{task_id}/.

Files per task:
  task.json     — TaskRecord (atomic write via .tmp + os.replace)
  output.log    — streaming plain text (append + fsync per chunk)
  messages.jsonl — full message history, one JSON object per line
  pending.jsonl — SendMessage queue, drained by drain_pending_messages()
  result.json   — written once on completion (SubagentResult dump)

Concurrency model:
- Single gateway process owns the store; in-memory per-task threading.Lock
  serializes writes to the same record (multi-thread, single-process).
- Cross-process locking is intentionally NOT implemented; if multiple
  gateways share the same store, the operator must use distinct roots.
"""
from __future__ import annotations
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from aether.runtime.tasks.contracts import TaskRecord, TaskStatus

logger = logging.getLogger(__name__)


_DEFAULT_HEARTBEAT_STALE_SECONDS = 60.0


class TaskStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path.home() / ".aether" / "tasks").expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    # ------------------------------------------------------------------ paths
    def _task_dir(self, task_id: str) -> Path:
        return self.root / task_id

    def _lock_for(self, task_id: str) -> threading.Lock:
        with self._locks_lock:
            lock = self._locks.get(task_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[task_id] = lock
            return lock

    # ------------------------------------------------------------------ CRUD
    def create(self, record: TaskRecord) -> None:
        d = self._task_dir(record.task_id)
        d.mkdir(parents=True, exist_ok=True)
        # touch output.log and messages.jsonl so they exist when readers attach
        (d / "output.log").touch(exist_ok=True)
        (d / "messages.jsonl").touch(exist_ok=True)
        (d / "pending.jsonl").touch(exist_ok=True)
        record.last_heartbeat = time.time()
        self._write_record(record)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        finished_at: Optional[float] = None,
        summary: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock_for(task_id):
            record = self._read_record(task_id)
            if record is None:
                logger.warning("update_status on unknown task %s", task_id)
                return
            record.status = status
            if finished_at is not None:
                record.finished_at = finished_at
            elif status.is_terminal and record.finished_at is None:
                record.finished_at = time.time()
            if summary is not None:
                record.summary = summary
            if error is not None:
                record.error = error
            record.last_heartbeat = time.time()
            self._write_record(record)

    def record_heartbeat(self, task_id: str) -> None:
        with self._lock_for(task_id):
            record = self._read_record(task_id)
            if record is None:
                return
            record.last_heartbeat = time.time()
            self._write_record(record)

    def record_progress(
        self,
        task_id: str,
        *,
        tool_use_count_delta: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        iterations: int | None = None,
    ) -> None:
        with self._lock_for(task_id):
            record = self._read_record(task_id)
            if record is None:
                return
            record.tool_use_count += tool_use_count_delta
            record.input_tokens += input_tokens
            record.output_tokens += output_tokens
            if iterations is not None:
                record.iterations = iterations
            record.last_heartbeat = time.time()
            self._write_record(record)

    # ------------------------------------------------------------------ streams
    def append_output(self, task_id: str, text: str) -> None:
        path = self._task_dir(task_id) / "output.log"
        with self._lock_for(task_id):
            with path.open("a", encoding="utf-8", buffering=1) as fh:
                fh.write(text)

    def read_output_tail(self, task_id: str, *, max_bytes: int = 100_000) -> str:
        path = self._task_dir(task_id) / "output.log"
        if not path.exists():
            return ""
        try:
            size = path.stat().st_size
            with path.open("rb") as fh:
                if size > max_bytes:
                    fh.seek(size - max_bytes)
                    fh.read(1)  # discard partial line
                return fh.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def append_message(self, task_id: str, message: dict) -> None:
        path = self._task_dir(task_id) / "messages.jsonl"
        with self._lock_for(task_id):
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(message, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ pending
    def enqueue_pending_message(self, task_id: str, msg: str) -> None:
        path = self._task_dir(task_id) / "pending.jsonl"
        entry = {"ts": time.time(), "message": msg}
        with self._lock_for(task_id):
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def drain_pending_messages(self, task_id: str) -> list[str]:
        path = self._task_dir(task_id) / "pending.jsonl"
        with self._lock_for(task_id):
            if not path.exists():
                return []
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []
            path.write_text("", encoding="utf-8")
        out: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                out.append(str(entry.get("message", "")))
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------ result
    def write_result(self, task_id: str, result_dump: dict) -> Path:
        d = self._task_dir(task_id)
        target = d / "result.json"
        tmp = d / "result.json.tmp"
        with self._lock_for(task_id):
            tmp.write_text(json.dumps(result_dump, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, target)
            record = self._read_record(task_id)
            if record is not None:
                record.result_path = str(target)
                self._write_record(record)
        return target

    # ------------------------------------------------------------------ queries
    def read(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock_for(task_id):
            return self._read_record(task_id)

    def list_active(self) -> list[TaskRecord]:
        return [r for r in self._iter_records() if not r.status.is_terminal]

    def list_recent(self, limit: int = 50) -> list[TaskRecord]:
        records = sorted(
            self._iter_records(),
            key=lambda r: r.started_at,
            reverse=True,
        )
        return records[:limit]

    def mark_orphaned(self, *, stale_after: float = _DEFAULT_HEARTBEAT_STALE_SECONDS) -> list[str]:
        """Mark RUNNING records with stale heartbeats as KILLED.

        Used by recovery on gateway startup.  Returns the list of task_ids
        that got marked.
        """
        now = time.time()
        marked: list[str] = []
        for record in list(self._iter_records()):
            if record.status == TaskStatus.RUNNING and (now - record.last_heartbeat) > stale_after:
                self.update_status(
                    record.task_id,
                    TaskStatus.KILLED,
                    error="Gateway restarted while task was running",
                )
                marked.append(record.task_id)
        return marked

    # ------------------------------------------------------------------ internals
    def _write_record(self, record: TaskRecord) -> None:
        d = self._task_dir(record.task_id)
        d.mkdir(parents=True, exist_ok=True)
        target = d / "task.json"
        tmp = d / "task.json.tmp"
        tmp.write_text(json.dumps(record.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, target)

    def _read_record(self, task_id: str) -> Optional[TaskRecord]:
        path = self._task_dir(task_id) / "task.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TaskRecord.from_json(data)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("task %s corrupt: %s", task_id, exc)
            return None

    def _iter_records(self) -> Iterator[TaskRecord]:
        if not self.root.exists():
            return
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            record = self._read_record(entry.name)
            if record is not None:
                yield record
```

#### `aether/runtime/tasks/recovery.py`

```python
from __future__ import annotations
import logging
from dataclasses import dataclass

from aether.runtime.tasks.store import TaskStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RecoveryReport:
    orphaned_count: int
    completed_count: int
    active_count: int


def recover_task_store(store: TaskStore) -> RecoveryReport:
    """Reconcile on-disk task records with the now-empty in-memory state.

    Called once at AgentEngine init (delegate_depth==0 only).  Marks any
    RUNNING records with stale heartbeats as KILLED.
    """
    orphans = store.mark_orphaned()
    if orphans:
        logger.info("task recovery: marked %d orphans as KILLED: %s", len(orphans), orphans)
    recent = store.list_recent(limit=10_000)
    completed = sum(1 for r in recent if r.status.value == "completed")
    active = len(store.list_active())
    return RecoveryReport(
        orphaned_count=len(orphans),
        completed_count=completed,
        active_count=active,
    )
```

### 2. Wire 进 `AgentEngine`

**改 `aether/agents/core/agent.py`**：

```python
from aether.runtime.tasks import TaskStore, recover_task_store

class AgentEngine:
    def __init__(
        self,
        ...,
        task_store: TaskStore | None = None,
    ) -> None:
        ...
        self._task_store: TaskStore | None = task_store
        if self._task_store is None and getattr(self.config, "task_store_enabled", True):
            root = getattr(self.config, "task_store_path", None)
            try:
                self._task_store = TaskStore(root=root)
            except Exception as exc:
                logger.warning("disabling task store: %s", exc)
                self._task_store = None

        # Recovery: only at the root engine (subagents share the parent's store).
        if self._task_store is not None and self.delegate_depth == 0:
            try:
                report = recover_task_store(self._task_store)
                logger.info(
                    "task store recovered: %d orphaned, %d completed, %d still active",
                    report.orphaned_count, report.completed_count, report.active_count,
                )
            except Exception:
                logger.exception("task store recovery failed; continuing")
```

`_prepare_turn_entry` 把 store 挂上 metadata（供 tools fallback 用）：

```python
context.metadata["_task_store"] = self._task_store
```

子 agent 共享 父的 store（builder 透传 `task_store=parent._task_store`）。

### 3. Config

`aether/config/schema.py`：

```python
task_store_enabled: bool = True
task_store_path: Path | None = None              # None -> ~/.aether/tasks
task_store_stale_seconds: float = 60.0           # heartbeat staleness threshold
```

## 测试 / Tests

### Python

新建 `aether/tests/runtime/tasks/test_store.py`（用 `tmp_path` fixture，每个 case 一个独立 root）：

- `test_create_and_read_roundtrip` —— create + read 字段全对。
- `test_update_status_terminal_sets_finished_at` —— 更新到 COMPLETED 后 finished_at 自动填。
- `test_append_output_concurrent` —— 10 个线程各 append 100 行；最终 output.log 总行数 = 1000。
- `test_drain_pending_clears_queue` —— enqueue 3 条 → drain 拿到 3 条 → 再 drain 拿到 0 条。
- `test_record_progress_accumulates` —— 多次 record_progress, counters 累加。
- `test_write_result_atomic` —— 中断模拟（写 .tmp 后强制 raise）→ result.json 仍是旧版本/不存在。
- `test_mark_orphaned_threshold` —— 一个 RUNNING 记录 heartbeat 90s 前 → mark_orphaned 标记为 KILLED；另一个 30s 前 → 不动。
- `test_corrupt_task_json_returns_none` —— 写一段乱码到 task.json → read 返回 None 且记日志。

新建 `aether/tests/runtime/tasks/test_recovery.py`：

- `test_recovery_marks_old_running_as_killed`
- `test_recovery_report_counts_match_store`

### 验收 / Acceptance

- `uv run pytest aether/tests/runtime/tasks/` 全绿。
- `uv run pyright` 无新告警。
- **手测**：
  1. 跑 `uv run aether`，spawn 同步 subagent 一次（PR 10.5 之前同步路径不写 store，本步骤验证 store 模块能正确初始化）。
  2. `ls -la ~/.aether/tasks/` —— 目录存在。
  3. 手动写一个假记录 `mkdir ~/.aether/tasks/test-tid; echo '{"task_id":"test-tid","parent_session_id":"s","subagent_type":"general-purpose","prompt":"p","status":"running","started_at":1,"last_heartbeat":1}' > ~/.aether/tasks/test-tid/task.json`，再起 gateway。task.json 中 status 变为 `killed`。

## 不在本 PR / Deferred

- **同步路径写 store** —— 本 PR 仅给 store 类做模块；同步路径要不要落盘等 PR 10.5（async lifecycle）确定写盘 hook 形态再决定。最稳的方案是同步和异步都走同一个 hook，因此本 PR 先准备好基础，PR 10.5 来接。
- **跨进程锁** —— 单 gateway 进程为前提，不实现。
- **TaskStore GC**（清理旧任务）—— 留 hook 但默认不开。
