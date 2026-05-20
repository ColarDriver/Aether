"""Compression lifecycle orchestration for context engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from aether.runtime.context.engine import ContextEngine
from aether.runtime.core.contracts import TurnContext

CompressionStatus = Literal["skipped", "compressed", "failed"]


@dataclass(slots=True)
class CompressionRequest:
    messages: list[dict[str, Any]]
    context: TurnContext
    trigger_reason: str
    force: bool = False
    focus: str | None = None


@dataclass(slots=True)
class CompressionResult:
    messages: list[dict[str, Any]]
    status: CompressionStatus
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class CompressionLifecycleService:
    """Run compression through a context engine with stable metadata."""

    def __init__(self, *, context_engine: ContextEngine) -> None:
        self._context_engine = context_engine

    @property
    def context_engine(self) -> ContextEngine:
        return self._context_engine

    def compress(self, request: CompressionRequest) -> CompressionResult:
        messages = request.messages
        context = request.context
        source_count = len(messages)
        source_tokens = _safe_int(context.metadata.get("compaction_last_tokens_before"))
        context_engine_meta = _context_engine_metadata(context)
        context_engine_meta["name"] = self._context_engine.name

        if not request.force and not self._context_engine.should_compress_preflight(
            messages,
            context=context,
        ):
            metadata = self._record_completion(
                context,
                status="skipped",
                trigger_reason=request.trigger_reason,
                source_message_count=source_count,
                result_message_count=source_count,
                source_tokens=source_tokens,
                result_tokens=source_tokens,
                reason="not_needed",
            )
            return CompressionResult(
                messages=messages,
                status="skipped",
                metadata=metadata,
            )

        self._record_started(
            context,
            trigger_reason=request.trigger_reason,
            source_message_count=source_count,
            source_tokens=source_tokens,
            force=request.force,
            focus=request.focus,
        )

        try:
            engine_result = self._context_engine.compact_preflight(
                messages,
                context=context,
                trigger_reason=request.trigger_reason,
            )
        except Exception as exc:  # noqa: BLE001 - compression must preserve transcript
            error = type(exc).__name__
            metadata = self._record_completion(
                context,
                status="failed",
                trigger_reason=request.trigger_reason,
                source_message_count=source_count,
                result_message_count=source_count,
                source_tokens=source_tokens,
                result_tokens=source_tokens,
                error=error,
            )
            return CompressionResult(
                messages=messages,
                status="failed",
                metadata=metadata,
                error=error,
            )

        result_messages = engine_result.messages
        validation_error = _validate_messages(result_messages)
        if validation_error is not None:
            metadata = self._record_completion(
                context,
                status="failed",
                trigger_reason=request.trigger_reason,
                source_message_count=source_count,
                result_message_count=source_count,
                source_tokens=source_tokens,
                result_tokens=source_tokens,
                error=validation_error,
            )
            return CompressionResult(
                messages=messages,
                status="failed",
                metadata=metadata,
                error=validation_error,
            )

        result_count = len(result_messages)
        engine_metadata = dict(engine_result.metadata)
        status: CompressionStatus = (
            "compressed"
            if engine_result.changed or result_messages is not messages
            else "skipped"
        )
        result_tokens = _safe_int(engine_metadata.get("tokens_after"))
        if result_tokens is None:
            result_tokens = source_tokens
        metadata = self._record_completion(
            context,
            status=status,
            trigger_reason=request.trigger_reason,
            source_message_count=source_count,
            result_message_count=result_count,
            source_tokens=source_tokens,
            result_tokens=result_tokens,
            reason=engine_result.reason,
            engine_metadata=engine_metadata,
        )
        return CompressionResult(
            messages=result_messages,
            status=status,
            metadata=metadata,
            error=engine_result.error,
        )

    def _record_started(
        self,
        context: TurnContext,
        *,
        trigger_reason: str,
        source_message_count: int,
        source_tokens: int | None,
        force: bool,
        focus: str | None,
    ) -> None:
        compression = _compression_metadata(context)
        compression["status"] = "started"
        compression["trigger_reason"] = trigger_reason
        compression["source_message_count"] = source_message_count
        if source_tokens is not None:
            compression["source_tokens"] = source_tokens
        compression["force"] = bool(force)
        if focus:
            compression["focus_present"] = True

    def _record_completion(
        self,
        context: TurnContext,
        *,
        status: CompressionStatus,
        trigger_reason: str,
        source_message_count: int,
        result_message_count: int,
        source_tokens: int | None,
        result_tokens: int | None,
        reason: str | None = None,
        error: str | None = None,
        engine_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        compression = _compression_metadata(context)
        metadata: dict[str, Any] = {
            "status": status,
            "trigger_reason": trigger_reason,
            "source_message_count": source_message_count,
            "result_message_count": result_message_count,
        }
        if source_tokens is not None:
            metadata["source_tokens"] = source_tokens
        if result_tokens is not None:
            metadata["result_tokens"] = result_tokens
        if reason:
            metadata["reason"] = reason
        if error:
            metadata["error"] = error
        if engine_metadata:
            metadata["engine"] = _json_safe_dict(engine_metadata)

        compression.clear()
        compression.update(metadata)

        context_engine_meta = _context_engine_metadata(context)
        context_engine_meta["name"] = self._context_engine.name
        context_engine_meta["compression"] = dict(metadata)
        context_engine_meta["last_trigger_reason"] = trigger_reason
        if status == "compressed":
            context_engine_meta["compression_count"] = (
                int(context_engine_meta.get("compression_count", 0) or 0) + 1
            )
        else:
            context_engine_meta.setdefault(
                "compression_count",
                int(context_engine_meta.get("compression_count", 0) or 0),
            )
        return dict(metadata)


def _context_engine_metadata(context: TurnContext) -> dict[str, Any]:
    metadata = context.metadata.get("context_engine")
    if not isinstance(metadata, dict):
        metadata = {}
        context.metadata["context_engine"] = metadata
    return metadata


def _compression_metadata(context: TurnContext) -> dict[str, Any]:
    context_engine_meta = _context_engine_metadata(context)
    compression = context_engine_meta.get("compression")
    if not isinstance(compression, dict):
        compression = {}
        context_engine_meta["compression"] = compression
    return compression


def _validate_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return "invalid_messages"
    for message in messages:
        if not isinstance(message, dict):
            return "invalid_message_item"
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return "missing_message_role"
    return None


def _safe_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _json_safe_dict(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        elif isinstance(value, list):
            safe[str(key)] = [
                item
                for item in value
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
        elif isinstance(value, dict):
            safe[str(key)] = _json_safe_dict(value)
        else:
            safe[str(key)] = type(value).__name__
    return safe


__all__ = [
    "CompressionLifecycleService",
    "CompressionRequest",
    "CompressionResult",
    "CompressionStatus",
]
