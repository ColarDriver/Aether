"""Subagent manager with controlled fan-out and depth."""

from __future__ import annotations

import logging
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, List

from aether.runtime.core.contracts import EngineStatus
from aether.runtime.core.hooks import EngineHooks
from aether.runtime.tasks import TaskRecord, TaskStatus, TaskStore
from aether.subagents.builder import SubagentBuilder
from aether.subagents.contracts import SubagentResult, SubagentStatus, SubagentTask
from aether.subagents.default_builder import DefaultSubagentBuilder
from aether.subagents.lifecycle_hooks import TaskStoreFanoutHooks

if TYPE_CHECKING:
    from aether.agents.core.agent import AgentEngine


class SubagentManager:
    """Coordinate child-agent execution with depth and concurrency limits."""

    def __init__(
        self,
        *,
        builder: SubagentBuilder | None = None,
        max_concurrent_children: int = 3,
        max_spawn_depth: int = 2,
        max_concurrent_background: int = 8,
        logger: logging.Logger | None = None,
    ) -> None:
        self.builder = builder or DefaultSubagentBuilder()
        self.max_concurrent_children = max(1, int(max_concurrent_children))
        self.max_spawn_depth = max(1, int(max_spawn_depth))
        self.max_concurrent_background = max(1, int(max_concurrent_background))
        self.logger = logger or logging.getLogger(__name__)
        # Track running tasks so a peer tool (``TaskStopTool``) or the
        # parent agent can request a graceful
        # interrupt.  The mapping is ``task_id -> threading.Event``;
        # the child agent observes the event at iteration boundaries
        # via ``AgentEngine.interrupt`` (set in ``_execute_one``).
        self._stop_events: dict[str, threading.Event] = {}
        self._stop_events_lock = threading.Lock()
        # Active children registry, keyed by task_id, used so
        # ``stop_task`` can route ``interrupt(...)`` to the right
        # ``AgentEngine`` instance.
        self._active_children: dict[str, "object"] = {}
        # Async lifecycle: lazy-built ThreadPoolExecutor + in-flight
        # future tracking so ``shutdown(wait=True)`` can drain cleanly.
        self._async_executor: ThreadPoolExecutor | None = None
        self._async_futures: dict[str, Future[SubagentResult]] = {}
        self._async_lock = threading.Lock()
        # PR 10.7: weakref map session_id → root engine, populated by
        # ``AgentEngine.__init__`` when ``delegate_depth==0``.  Used by
        # ``_on_async_done`` to bubble ``<task-notification>`` messages
        # back to the parent that spawned the (now-finished) child.
        # WeakValueDictionary so a closed CLI session doesn't pin its
        # AgentEngine in memory forever.
        self._root_engines: "weakref.WeakValueDictionary[str, AgentEngine]" = (
            weakref.WeakValueDictionary()
        )

    def run_task(self, parent, task: SubagentTask) -> SubagentResult:
        return self.run_tasks(parent=parent, tasks=[task])[0]

    def run_tasks(
        self,
        *,
        parent,
        tasks: List[SubagentTask],
        max_concurrent_children: int | None = None,
    ) -> List[SubagentResult]:
        if not tasks:
            return []

        if parent.delegate_depth >= self.max_spawn_depth:
            raise RuntimeError(
                f"Delegation depth limit reached: depth={parent.delegate_depth}, max_spawn_depth={self.max_spawn_depth}"
            )

        child_depth = parent.delegate_depth + 1
        worker_limit = max_concurrent_children or self.max_concurrent_children
        worker_limit = max(1, min(worker_limit, len(tasks)))

        if len(tasks) == 1:
            return [self._execute_one(parent=parent, task=tasks[0], child_depth=child_depth)]

        results_by_task_id: dict[str, SubagentResult] = {}
        with ThreadPoolExecutor(max_workers=worker_limit) as executor:
            futures = {
                executor.submit(self._execute_one, parent=parent, task=task, child_depth=child_depth): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    results_by_task_id[task.task_id] = future.result()
                except Exception as exc:  # pragma: no cover - defensive fallback
                    self.logger.exception("Subagent execution failed for task %s: %s", task.task_id, exc)
                    results_by_task_id[task.task_id] = SubagentResult(
                        task_id=task.task_id,
                        status=SubagentStatus.FAILED,
                        summary=None,
                        engine_result=None,
                        error=str(exc),
                    )

        return [results_by_task_id[t.task_id] for t in tasks]

    # ---------------------------------------------------------- public ext.

    def stop_task(self, task_id: str) -> bool:
        """Signal a running task to stop at its next iteration boundary.

        Returns ``True`` if the task was running and the signal was
        delivered, ``False`` if no such task is known (already finished,
        never started, or unknown ``task_id``).
        """
        if not task_id:
            return False
        with self._stop_events_lock:
            event = self._stop_events.get(task_id)
            child = self._active_children.get(task_id)
        if event is None and child is None:
            return False
        if event is not None:
            event.set()
        if child is not None:
            interrupt_fn = getattr(child, "interrupt", None)
            if callable(interrupt_fn):
                try:
                    interrupt_fn(reason=f"stopped by parent (task_id={task_id})")
                except Exception:  # pragma: no cover - defensive
                    pass
        return True

    def is_task_running(self, task_id: str) -> bool:
        with self._stop_events_lock:
            return task_id in self._active_children

    # ---------------------------------------------------------- async path

    def run_task_async(self, *, parent, task: SubagentTask) -> str:
        """Spawn a child on a background thread, return ``task_id`` immediately.

        The caller is expected to track ``task_id`` and use
        :class:`TaskOutputTool` (PR 10.6) or peer ``send_message``
        (PR 10.7) to interact with the running child.  Final state is
        always available from the parent's :class:`TaskStore`.
        """
        store = self._resolve_task_store(parent)
        if store is None:
            raise RuntimeError(
                "run_task_async requires a TaskStore; check "
                "EngineConfig.task_store_enabled on the parent."
            )
        if parent.delegate_depth >= self.max_spawn_depth:
            raise RuntimeError(
                f"Delegation depth limit reached: depth={parent.delegate_depth}, "
                f"max_spawn_depth={self.max_spawn_depth}"
            )

        # Write the initial RUNNING record BEFORE submitting so external
        # observers (e.g. ``task_output`` invoked immediately after) see
        # the task even when the executor is briefly saturated.
        record = self._build_initial_record(parent=parent, task=task, background=True)
        store.create(record)

        executor = self._ensure_async_executor()
        future = executor.submit(
            self._execute_one_async,
            parent=parent,
            task=task,
            child_depth=parent.delegate_depth + 1,
            store=store,
        )
        with self._async_lock:
            self._async_futures[task.task_id] = future
        future.add_done_callback(
            lambda f, tid=task.task_id: self._on_async_done(tid, f)
        )
        return task.task_id

    def shutdown(self, *, wait: bool = True) -> None:
        """Drain the async executor.  Idempotent."""
        with self._async_lock:
            executor = self._async_executor
            self._async_executor = None
        if executor is not None:
            executor.shutdown(wait=wait)

    # ---------------------------------------------------------- root registry

    def register_root_engine(self, engine: "AgentEngine") -> None:
        """Track a root :class:`AgentEngine` so async notifications can
        be routed back to it after a child finishes.

        Called from :meth:`AgentEngine.__init__` when
        ``delegate_depth==0``.  Re-registers on each construction; an
        engine with the same ``_current_session_id`` overwrites a stale
        entry from a previous instantiation.  The map holds weakrefs,
        so closed engines drop out automatically.
        """
        session_id = getattr(engine, "_current_session_id", None)
        # ``_current_session_id`` is set on ``run_loop`` entry, not at
        # init.  Until then we key by the engine's id() so the
        # registration is non-destructive; ``_on_async_done`` will
        # match by both keys.
        key = session_id or f"engine:{id(engine)}"
        self._root_engines[key] = engine

    def _lookup_root_engine_for(
        self, *, session_id: str | None, parent_session_id: str | None
    ) -> "AgentEngine | None":
        """Best-effort lookup: try the parent's session_id first, then
        any engine whose current session_id matches, then fall through
        to ``None``."""
        candidates = [s for s in (parent_session_id, session_id) if s]
        for key in candidates:
            engine = self._root_engines.get(key)
            if engine is not None:
                return engine
        # Second pass: scan for an engine whose live ``_current_session_id``
        # matches one of our candidates (it may have been registered
        # with the placeholder ``engine:<id>`` key before its first
        # ``run_loop`` set the real session id).
        for engine in list(self._root_engines.values()):
            if getattr(engine, "_current_session_id", None) in candidates:
                return engine
        return None

    # ----------------------------------------------------------- internals

    def _ensure_async_executor(self) -> ThreadPoolExecutor:
        with self._async_lock:
            if self._async_executor is None:
                self._async_executor = ThreadPoolExecutor(
                    max_workers=self.max_concurrent_background,
                    thread_name_prefix="aether-async-subagent",
                )
            return self._async_executor

    def _on_async_done(self, task_id: str, future: Future[SubagentResult]) -> None:
        with self._async_lock:
            self._async_futures.pop(task_id, None)
        try:
            result = future.result()
        except Exception:  # noqa: BLE001 - already finalized into the store
            self.logger.exception(
                "async task %s done callback observed exception", task_id
            )
            return

        # PR 10.7: bubble a ``<task-notification>`` back to the parent.
        # If the parent is itself an async subagent (parent_task_id set),
        # we land in their on-disk pending queue so they pick it up at
        # their next iteration boundary.  Otherwise the parent is a
        # root engine — drop the message into its in-memory event
        # queue via ``enqueue_external_event``.
        try:
            self._dispatch_completion_notification(task_id, result)
        except Exception:  # pragma: no cover - defensive
            self.logger.exception(
                "failed to dispatch task-notification for %s", task_id
            )

    def _dispatch_completion_notification(
        self, task_id: str, result: SubagentResult
    ) -> None:
        notification = _build_task_notification(task_id, result)
        parent_task_id = (
            (result.metadata or {}).get("parent_task_id")
            if result.metadata
            else None
        )
        store = None  # resolve from any registered engine if needed
        if parent_task_id:
            for engine in list(self._root_engines.values()):
                store = getattr(engine, "_task_store", None)
                if store is not None:
                    break
            if store is not None:
                try:
                    record = store.read(parent_task_id)
                except Exception:  # pragma: no cover - defensive
                    record = None
                if record is not None:
                    store.enqueue_pending_message(parent_task_id, notification)
                    return
            # Parent task isn't in the store (or store is gone) — fall
            # through to in-memory routing so the message isn't lost.

        parent_session_id = (
            (result.metadata or {}).get("parent_session_id")
            if result.metadata
            else None
        )
        engine = self._lookup_root_engine_for(
            session_id=None, parent_session_id=parent_session_id
        )
        if engine is not None:
            engine.enqueue_external_event(notification)

    @staticmethod
    def _resolve_task_store(parent) -> TaskStore | None:
        store = getattr(parent, "_task_store", None)
        return store if isinstance(store, TaskStore) else None

    @staticmethod
    def _build_initial_record(
        *, parent, task: SubagentTask, background: bool
    ) -> TaskRecord:
        definition = task.metadata.get("_agent_type_def")
        snapshot: dict[str, Any] = {}
        if definition is not None and hasattr(definition, "to_snapshot"):
            try:
                snapshot = definition.to_snapshot()
            except Exception:  # pragma: no cover - defensive
                snapshot = {}
        request = task.request
        request_metadata = getattr(request, "metadata", {}) or {}
        model = None
        if request.model_config is not None:
            model = request.model_config.extra.get("model")
        return TaskRecord(
            task_id=task.task_id,
            parent_session_id=request_metadata.get("parent_session_id", "")
            or getattr(parent, "_current_session_id", "")
            or "",
            subagent_type=task.metadata.get("subagent_type", "general-purpose"),
            prompt=request.user_message or "",
            status=TaskStatus.RUNNING,
            started_at=time.time(),
            last_heartbeat=time.time(),
            agent_type_def_snapshot=snapshot,
            model=model,
            isolation=task.metadata.get("isolation"),
            parent_task_id=request_metadata.get("parent_task_id"),
            child_depth=parent.delegate_depth + 1,
            background=background,
        )

    def _install_task_store_hooks(
        self, child, *, store: TaskStore, task_id: str
    ) -> None:
        original = getattr(child, "_hooks", None) or EngineHooks()
        child._hooks = TaskStoreFanoutHooks(  # noqa: SLF001 — runtime injection
            inner=original, store=store, task_id=task_id
        )

    def _finalize_to_store(
        self, *, store: TaskStore, task_id: str, result: SubagentResult
    ) -> None:
        terminal_map = {
            SubagentStatus.COMPLETED: TaskStatus.COMPLETED,
            SubagentStatus.FAILED: TaskStatus.FAILED,
            SubagentStatus.INTERRUPTED: TaskStatus.INTERRUPTED,
        }
        status = terminal_map.get(result.status, TaskStatus.FAILED)
        try:
            store.update_status(
                task_id, status, summary=result.summary, error=result.error
            )
            store.write_result(
                task_id,
                {
                    "task_id": result.task_id,
                    "status": status.value,
                    "summary": result.summary,
                    "error": result.error,
                    "duration_seconds": result.duration_seconds,
                    "metadata": dict(result.metadata or {}),
                },
            )
        except Exception:  # pragma: no cover - defensive
            self.logger.exception(
                "task store finalize failed for task %s", task_id
            )

    def _execute_one_async(
        self,
        *,
        parent,
        task: SubagentTask,
        child_depth: int,
        store: TaskStore,
    ) -> SubagentResult:
        """Async sibling of :meth:`_execute_one`, instrumented to write
        every lifecycle event into ``store``.

        Mirrors the sync path's stop-event registration / interrupt
        plumbing so ``task_stop`` works the same way for async children.
        """
        started_at = time.monotonic()
        child = self.builder.build_child(parent=parent, task=task, child_depth=child_depth)
        self._install_task_store_hooks(child, store=store, task_id=task.task_id)
        parent._register_child(child)

        stop_event = threading.Event()
        with self._stop_events_lock:
            self._stop_events[task.task_id] = stop_event
            self._active_children[task.task_id] = child
        setattr(child, "_stop_event", stop_event)

        try:
            child_result = child.run_loop(task.request)
            if child_result.status == EngineStatus.COMPLETED:
                status = SubagentStatus.COMPLETED
            elif child_result.status == EngineStatus.INTERRUPTED:
                status = SubagentStatus.INTERRUPTED
            else:
                status = SubagentStatus.FAILED
            result = SubagentResult(
                task_id=task.task_id,
                status=status,
                summary=child_result.final_response,
                engine_result=child_result,
                error=child_result.error,
                duration_seconds=time.monotonic() - started_at,
                metadata={
                    "goal": task.goal,
                    "child_depth": child_depth,
                    "subagent_id": child.subagent_id,
                    "background": True,
                    "subagent_type": task.metadata.get("subagent_type"),
                    "parent_task_id": task.metadata.get("parent_task_id"),
                    "parent_session_id": getattr(parent, "_current_session_id", None),
                },
            )
            self._finalize_to_store(store=store, task_id=task.task_id, result=result)
            return result
        except Exception as exc:
            self.logger.exception(
                "Async subagent task %s crashed: %s", task.task_id, exc
            )
            result = SubagentResult(
                task_id=task.task_id,
                status=SubagentStatus.FAILED,
                summary=None,
                engine_result=None,
                error=str(exc),
                duration_seconds=time.monotonic() - started_at,
                metadata={
                    "goal": task.goal,
                    "child_depth": child_depth,
                    "subagent_id": child.subagent_id,
                    "background": True,
                    "subagent_type": task.metadata.get("subagent_type"),
                    "parent_task_id": task.metadata.get("parent_task_id"),
                    "parent_session_id": getattr(parent, "_current_session_id", None),
                },
            )
            self._finalize_to_store(store=store, task_id=task.task_id, result=result)
            return result
        finally:
            parent._unregister_child(child)
            with self._stop_events_lock:
                self._stop_events.pop(task.task_id, None)
                self._active_children.pop(task.task_id, None)

    def _execute_one(self, *, parent, task: SubagentTask, child_depth: int) -> SubagentResult:
        started_at = time.monotonic()
        child = self.builder.build_child(parent=parent, task=task, child_depth=child_depth)
        parent._register_child(child)

        stop_event = threading.Event()
        with self._stop_events_lock:
            self._stop_events[task.task_id] = stop_event
            self._active_children[task.task_id] = child
        # Best-effort: child may inspect this attribute if it grows a
        # cooperative cancel hook.  The interrupt path above covers the
        # actual stop semantics today.
        setattr(child, "_stop_event", stop_event)

        try:
            child_result = child.run_loop(task.request)
            if child_result.status == EngineStatus.COMPLETED:
                status = SubagentStatus.COMPLETED
            elif child_result.status == EngineStatus.INTERRUPTED:
                status = SubagentStatus.INTERRUPTED
            else:
                status = SubagentStatus.FAILED

            summary = child_result.final_response
            return SubagentResult(
                task_id=task.task_id,
                status=status,
                summary=summary,
                engine_result=child_result,
                error=child_result.error,
                duration_seconds=time.monotonic() - started_at,
                metadata={
                    "goal": task.goal,
                    "child_depth": child_depth,
                    "subagent_id": child.subagent_id,
                },
            )
        except Exception as exc:
            self.logger.exception("Subagent task %s crashed: %s", task.task_id, exc)
            return SubagentResult(
                task_id=task.task_id,
                status=SubagentStatus.FAILED,
                summary=None,
                engine_result=None,
                error=str(exc),
                duration_seconds=time.monotonic() - started_at,
                metadata={
                    "goal": task.goal,
                    "child_depth": child_depth,
                    "subagent_id": child.subagent_id,
                },
            )
        finally:
            parent._unregister_child(child)
            with self._stop_events_lock:
                self._stop_events.pop(task.task_id, None)
                self._active_children.pop(task.task_id, None)


# ----------------------------------------------------------- helpers


def _build_task_notification(task_id: str, result: SubagentResult) -> str:
    """Render the ``<task-notification>`` XML the parent sees as a user turn.

    Format mirrors ``open-claude-code/src/coordinator/coordinatorMode.ts``:
    a small XML block with task_id / subagent_type / status / duration
    / optional summary / optional error.  Escapes ``&`` / ``<`` / ``>``
    so a child's tool output containing those characters cannot break
    the markup.
    """
    raw_status = (
        result.status.value
        if isinstance(result.status, SubagentStatus)
        else str(result.status)
    )
    # SubagentStatus values are upper-case ("COMPLETED" / "FAILED" / ...);
    # TaskStatus uses lower-case.  Render lower-case in the wire payload
    # so consumers see a single convention regardless of which enum the
    # producer used.
    status = raw_status.lower()
    metadata = result.metadata or {}
    subagent_type = str(metadata.get("subagent_type") or "general-purpose")
    summary = (result.summary or "").strip()
    error = (result.error or "").strip()
    parts: list[str] = [
        "<task-notification>",
        f"  <task_id>{_xml_escape(task_id)}</task_id>",
        f"  <subagent_type>{_xml_escape(subagent_type)}</subagent_type>",
        f"  <status>{_xml_escape(status)}</status>",
        f"  <duration_seconds>{result.duration_seconds:.1f}</duration_seconds>",
    ]
    if summary:
        parts.append(f"  <summary>{_xml_escape(summary)}</summary>")
    if error:
        parts.append(f"  <error>{_xml_escape(error)}</error>")
    parts.append("</task-notification>")
    return "\n".join(parts)


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
