"""Task-scoped resource tracking and cleanup helpers."""

from __future__ import annotations

import logging
from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_task_cleanup_metadata(
    cleanup_meta: dict[str, Any],
    *,
    acquired_default: int = 0,
) -> dict[str, Any]:
    """Keep the public cleanup metadata shape stable under bad inputs."""

    cleanup_meta["acquired"] = _safe_int(
        cleanup_meta.get("acquired"),
        acquired_default,
    )
    cleanup_meta["released"] = _safe_int(cleanup_meta.get("released"), 0)
    if not isinstance(cleanup_meta.get("errors"), list):
        cleanup_meta["errors"] = []
    return cleanup_meta


def acquire_task_resource_for_executor(
    executor: Any,
    *,
    task_id: str | None,
    context_metadata: dict[str, Any],
) -> bool:
    """Call an executor's optional acquire hook and track it for release."""

    if not task_id:
        return False
    acquire = getattr(executor, "acquire_task_resource", None)
    release = getattr(executor, "release_task_resource", None)
    if not callable(acquire) or not callable(release):
        return False

    key = (id(executor), task_id)
    acquired_keys = context_metadata.get("_task_resource_keys")
    if not isinstance(acquired_keys, set):
        acquired_keys = set()
        context_metadata["_task_resource_keys"] = acquired_keys
    if key in acquired_keys:
        return False

    acquire(task_id)
    acquired_keys.add(key)
    resources = context_metadata.get("_task_resource_handles")
    if not isinstance(resources, list):
        resources = []
        context_metadata["_task_resource_handles"] = resources
    resources.append(executor)
    cleanup_meta = context_metadata.get("resource_cleanup")
    if not isinstance(cleanup_meta, dict):
        cleanup_meta = {}
        context_metadata["resource_cleanup"] = cleanup_meta
    normalize_task_cleanup_metadata(cleanup_meta)
    cleanup_meta["acquired"] += 1
    return True


def release_task_resources(
    resources: list[Any],
    *,
    task_id: str | None,
    cleanup_meta: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Release tracked task resources in reverse acquisition order."""

    normalize_task_cleanup_metadata(
        cleanup_meta,
        acquired_default=len(resources),
    )
    if not task_id:
        return cleanup_meta

    seen: set[int] = set()
    for resource in reversed(resources):
        resource_id = id(resource)
        if resource_id in seen:
            continue
        seen.add(resource_id)
        release = getattr(resource, "release_task_resource", None)
        if not callable(release):
            continue
        try:
            release(task_id)
            cleanup_meta["released"] += 1
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            cleanup_meta["errors"].append(
                {
                    "resource": _resource_name(resource),
                    "error": str(exc),
                }
            )
            if logger is not None:
                logger.exception("task resource release failed: %s", _resource_name(resource))
    return cleanup_meta


def _resource_name(resource: Any) -> str:
    descriptor = getattr(resource, "descriptor", None)
    name = getattr(descriptor, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(resource).__name__
