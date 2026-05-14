# PR 10.8 — Worktree Isolation

## 目标 / Goal

当 subagent 类型定义指定 `isolation: worktree`、或 `task()` 调用层面传 `isolation="worktree"` 时：

1. 用 `git worktree add` 为子 agent 创建一个独立的工作目录与分支。
2. 子 agent 的 CWD 设到 worktree 目录；文件编辑只影响该 worktree。
3. 子 agent 完成后，若 worktree 干净则自动 `git worktree remove`；若有未提交改动或分支独占 commit 则保留，留待用户手动审阅。
4. `TaskRecord.worktree_path` / `worktree_branch` 落盘，`TaskOutput` 能查到隔离环境的位置。

参考：`open-claude-code/src/tools/AgentTool/AgentTool.tsx:590-603`（worktree 创建）+ `src/tools/AgentTool/agentToolUtils.ts`（cleanup 决策）。

## 当前问题 / Current Problem

完全不存在。Aether 中所有子 agent 都直接在父的 working tree 上工作，没有任何文件系统级别的隔离。

## 改动 / Changes

### 1. 新文件 `aether/runtime/tasks/worktree.py`

```python
"""Git worktree helpers for subagent isolation.

Behavior parallels `open-claude-code/src/tools/AgentTool/AgentTool.tsx:590-603`:
- Branch off `worktree.baseRef` ("fresh" => origin/<default> ; "head" => HEAD).
- Path lives under `<repo>/.claude/worktrees/<slug>/`.
- Cleanup is safe-by-default: keeps the worktree if there are uncommitted
  changes or branch-local commits; only removes if it's pristine.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class WorktreeInfo:
    path: Path
    branch: str
    head_commit: str
    base_ref: str
    repo_root: Path


@dataclass(slots=True)
class CleanupReport:
    removed: bool
    reason: str            # "clean" | "dirty" | "branch_has_local_commits" | "forced" | "not_found"
    worktree_path: Path
    branch: str


# ---------------------------------------------------------------- public API

def is_inside_repo(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--git-dir"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def create_agent_worktree(
    repo_root: Path,
    slug: str,
    *,
    base_ref: str = "fresh",
    worktree_root: Optional[Path] = None,
) -> WorktreeInfo:
    """Create a worktree at `<worktree_root>/<slug>` on branch `agent/<slug>`."""
    repo_root = repo_root.resolve()
    if not is_inside_repo(repo_root):
        raise RuntimeError(f"{repo_root} is not inside a git repository")

    target_root = (worktree_root or repo_root / ".claude" / "worktrees").resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    target_path = target_root / slug
    if target_path.exists():
        raise RuntimeError(f"worktree path already exists: {target_path}")

    branch = f"agent/{slug}"
    resolved_base = _resolve_base_ref(repo_root, base_ref)

    _git(repo_root, "worktree", "add", "-b", branch, str(target_path), resolved_base)
    head = _git_capture(target_path, "rev-parse", "HEAD").strip()

    logger.info("created agent worktree %s on %s (head=%s)", target_path, branch, head)
    return WorktreeInfo(
        path=target_path,
        branch=branch,
        head_commit=head,
        base_ref=resolved_base,
        repo_root=repo_root,
    )


def cleanup_worktree(info: WorktreeInfo, *, force: bool = False) -> CleanupReport:
    """Remove the worktree if pristine; else keep it for manual review."""
    if not info.path.exists():
        return CleanupReport(removed=False, reason="not_found", worktree_path=info.path, branch=info.branch)

    if not force:
        if _has_uncommitted_changes(info.path):
            return CleanupReport(removed=False, reason="dirty", worktree_path=info.path, branch=info.branch)
        if _branch_has_local_commits(info.path, info.head_commit):
            return CleanupReport(
                removed=False,
                reason="branch_has_local_commits",
                worktree_path=info.path,
                branch=info.branch,
            )

    try:
        _git(info.repo_root, "worktree", "remove", "--force" if force else "", str(info.path))
        _git(info.repo_root, "branch", "-D" if force else "-d", info.branch)
    except subprocess.CalledProcessError as exc:
        logger.warning("worktree cleanup failed: %s", exc)
        return CleanupReport(removed=False, reason="git_error", worktree_path=info.path, branch=info.branch)

    return CleanupReport(
        removed=True,
        reason="forced" if force else "clean",
        worktree_path=info.path,
        branch=info.branch,
    )


# ---------------------------------------------------------------- internals

def _resolve_base_ref(repo_root: Path, mode: str) -> str:
    if mode == "head":
        return "HEAD"
    # "fresh" — origin/<default-branch>
    try:
        ref = _git_capture(repo_root, "symbolic-ref", "refs/remotes/origin/HEAD").strip()
        # e.g. "refs/remotes/origin/master"
        return ref.replace("refs/remotes/", "")
    except subprocess.CalledProcessError:
        return "HEAD"  # fallback


def _has_uncommitted_changes(path: Path) -> bool:
    status = _git_capture(path, "status", "--porcelain")
    return bool(status.strip())


def _branch_has_local_commits(path: Path, base_commit: str) -> bool:
    try:
        count = _git_capture(path, "rev-list", "--count", f"{base_commit}..HEAD").strip()
        return int(count or "0") > 0
    except (subprocess.CalledProcessError, ValueError):
        return False


def _git(cwd: Path, *args: str) -> None:
    cmd = ["git", "-C", str(cwd), *(a for a in args if a)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)


def _git_capture(cwd: Path, *args: str) -> str:
    cmd = ["git", "-C", str(cwd), *args]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
    return result.stdout
```

### 2. Wire 进 SubagentBuilder

**改 `aether/subagents/default_builder.py`**：

```python
from aether.runtime.tasks.worktree import (
    WorktreeInfo,
    cleanup_worktree,
    create_agent_worktree,
    is_inside_repo,
)

# in build_child(), after definition resolved
isolation = task.metadata.get("isolation") or (definition.isolation if definition else None)

worktree_info: WorktreeInfo | None = None
if isolation == "worktree":
    repo_root = parent._cwd_repo_root()  # NEW helper on AgentEngine
    if repo_root is not None and is_inside_repo(repo_root):
        try:
            slug = f"agent-{task.task_id[:8]}"
            base_ref = getattr(parent.config, "worktree_base_ref", "fresh")
            wt_root = getattr(parent.config, "worktree_root", None)
            worktree_info = create_agent_worktree(
                repo_root, slug, base_ref=base_ref, worktree_root=wt_root,
            )
        except Exception as exc:
            # Worktree creation failures must not silently fall back to
            # the parent's tree — error out so the user sees it.
            raise RuntimeError(f"worktree creation failed: {exc}") from exc
    else:
        raise RuntimeError("isolation=worktree requested but parent is not inside a git repo")

# pass to engine
child = AgentEngine(..., cwd=worktree_info.path if worktree_info else None)

# Write back to task metadata for store / cleanup hook to consume
if worktree_info is not None:
    task.metadata["worktree"] = {
        "path": str(worktree_info.path),
        "branch": worktree_info.branch,
        "head_commit": worktree_info.head_commit,
        "base_ref": worktree_info.base_ref,
    }
    # also push into store record (manager will pick this up at lifecycle finalize)
```

**改 `aether/subagents/manager.py`** —— 在 `_execute_one` / `_execute_one_async` 的 finally 块中，若 `task.metadata.get("worktree")` 存在，调 `cleanup_worktree`：

```python
finally:
    parent._unregister_child(child)
    with self._stop_events_lock:
        self._stop_events.pop(task.task_id, None)
        self._active_children.pop(task.task_id, None)
    wt_meta = task.metadata.get("worktree")
    if wt_meta:
        try:
            info = WorktreeInfo(
                path=Path(wt_meta["path"]),
                branch=wt_meta["branch"],
                head_commit=wt_meta["head_commit"],
                base_ref=wt_meta["base_ref"],
                repo_root=parent._cwd_repo_root() or Path.cwd(),
            )
            report = cleanup_worktree(info)
            self.logger.info("worktree cleanup: %s -> removed=%s reason=%s",
                             info.path, report.removed, report.reason)
            if self._task_store is not None:
                # Record final disposition on the task record
                record = self._task_store.read(task.task_id)
                if record is not None:
                    record.worktree_path = str(info.path) if not report.removed else None
                    self._task_store._write_record(record)  # noqa: SLF001
        except Exception:
            self.logger.exception("worktree cleanup failed for task %s", task.task_id)
```

### 3. Inject worktree notice into child's first user message

`open-claude-code` 在 fork 路径会注入一条系统提示告诉 child "你在 worktree 里，路径要这么对应"。Aether 同理，在 `build_child` 末尾把一条 `<system-reminder>` 加在 `task.request.system_message` 末尾：

```python
if worktree_info is not None:
    notice = (
        "\n\n<system-reminder>\n"
        f"You are working in an isolated git worktree at {worktree_info.path}.\n"
        f"Branch: {worktree_info.branch} (based on {worktree_info.base_ref}).\n"
        "Edits stay in this worktree and do not affect the parent's working tree.\n"
        f"The original repository is at {worktree_info.repo_root}.\n"
        "</system-reminder>"
    )
    task.request.system_message = (task.request.system_message or "") + notice
```

### 4. AgentEngine 暴露 cwd / repo_root

`AgentEngine` 当前可能没有显式 `cwd` 字段。加：

```python
class AgentEngine:
    def __init__(self, ..., cwd: Path | None = None) -> None:
        self._cwd = (cwd or Path.cwd()).resolve()

    def _cwd_repo_root(self) -> Path | None:
        # Walk up looking for .git
        for ancestor in (self._cwd, *self._cwd.parents):
            if (ancestor / ".git").exists():
                return ancestor
        return None
```

Tools that need cwd（如 `shell`, `read_file`, `write_file`, `file_edit`）应该把 `context.metadata["_cwd"] = engine._cwd` 设上，由工具消费——本 PR 不重写所有工具，仅引入 `_cwd` 概念；shell/file tools 渐进采用。

### 5. AgentTool 接收 isolation 参数

**改 `aether/tools/builtins/agent_tool.py`**：

```python
"isolation": {
    "type": "string",
    "enum": ["worktree", "none"],
    "default": "none",
    "description": (
        "Isolation mode for this subagent.  'worktree' creates a "
        "temporary git worktree so the agent works on an isolated "
        "copy of the repo.  Useful for risky edits or parallel tasks."
    ),
},
```

`execute` 把 `args.get("isolation")`（或定义里的 isolation）写进 `task.metadata["isolation"]`。

### 6. Config

`aether/config/schema.py`：

```python
worktree_base_ref: str = "fresh"             # "fresh" | "head"
worktree_root: Path | None = None            # None -> <repo>/.claude/worktrees
worktree_cleanup_force: bool = False         # 默认保守
```

## 测试 / Tests

### Python

新建 `aether/tests/runtime/tasks/test_worktree.py`（使用 tmp git repo fixture）：

- `test_create_then_cleanup_clean` —— create + 不动 → cleanup 成功 removed=True reason="clean"。
- `test_cleanup_keeps_dirty_worktree` —— create + touch 新文件 → cleanup removed=False reason="dirty"。
- `test_cleanup_keeps_branch_with_commits` —— create + 在 worktree 内 commit → cleanup removed=False reason="branch_has_local_commits"。
- `test_force_removes_dirty` —— dirty + force=True → removed=True reason="forced"。
- `test_path_collision_raises` —— 已存在同名 worktree → create 抛 RuntimeError。
- `test_outside_repo_raises` —— tmp_path 不是 git repo → create 抛 RuntimeError。
- `test_base_ref_fresh_resolves_to_origin_default` —— mock `git symbolic-ref` → 返回 `origin/master`。

新建 `aether/tests/subagents/test_worktree_subagent.py`：

- `test_explore_with_worktree_isolation` —— scripted child 在 worktree 内运行，child._cwd 是 worktree path。
- `test_child_dirty_keeps_worktree` —— child 在 worktree 内 touch 文件（通过 hook 模拟）→ cleanup 保留；TaskRecord.worktree_path 不为 None。
- `test_child_clean_removes_worktree` —— child 不写 → cleanup 移除；TaskRecord.worktree_path 为 None。
- `test_worktree_failure_propagates_as_subagent_error` —— mock create_agent_worktree 抛 → AgentTool 返回 is_error 且不 spawn child。

### 验收 / Acceptance

- `uv run pytest aether/tests/runtime/tasks/test_worktree.py aether/tests/subagents/test_worktree_subagent.py` 全绿。
- `uv run pyright` 无新告警。
- **手测**：
  1. 在 Aether 仓库根跑 `uv run aether`。
  2. `task(subagent_type="general-purpose", prompt="create a file foo.txt with 'hello'", isolation="worktree")` → 拿到 task_id。
  3. `git worktree list` → 多一行 `.claude/worktrees/agent-xxxxxxxx`。
  4. 子完成；若该行还在，进入目录看到 foo.txt（dirty，保留）；若该行消失（child 没真的写），表示 cleanup 成功。
  5. `task_output(task_id="...")` → metadata 含 `worktree_path`。

## 不在本 PR / Deferred

- **Remote isolation (`isolation: remote` / CCR)** —— 不实现。
- **Worktree with sparse-checkout / shallow clone** —— 不引入复杂选项。
- **TUI 端 worktree picker / branch viewer** —— 后续 sprint。
- **跨 worktree 的并发协调** —— git worktree 自身处理；本 PR 仅串行 cleanup。
