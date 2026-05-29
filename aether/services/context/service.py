"""Service-backed context compression status and control."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any, Protocol, TypeVar

from aether.agents.core.agent import AgentEngine
from aether.agents.runtime.context_assembly import LegacyContextAssemblyAdapter
from aether.cli.sessions import SessionRecord
from aether.runtime.context import (
    CompressionLifecycleService,
    CompressionRequest,
    CompressionResult,
    DefaultContextEngine,
)
from aether.runtime.core.contracts import TurnContext
from aether.services.common import ServiceValidationError
from aether.services.compact import estimate_messages_tokens
from aether.cli.providers import resolve_provider_name
from aether.services.context.contracts import (
    ContextBreakdownItem,
    ContextCompressRequest,
    ContextCompressResult,
    ContextEstimateRequest,
    ContextStatusResult,
)
from aether.services.providers import MODEL_CATALOG
from aether.services.runs import AgentRunOptions
from aether.services.runs.builder import RunDependencyBuilder
from aether.services.sessions import SessionService


class _CompressionService(Protocol):
    def compress(self, request: CompressionRequest) -> CompressionResult: ...


CompressionServiceFactory = Callable[[SessionRecord], _CompressionService]
ModelWindowResolver = Callable[[str | None, str | None], int | None]
StatusT = TypeVar("StatusT", bound=ContextStatusResult)


class ContextService:
    """Control surface for manual context compression.

    The agent runtime still owns the actual compaction pipeline. This service
    gives browser and other clients a stable API for status and manual compact
    requests without importing gateway handlers.
    """

    def __init__(
        self,
        *,
        session_service: SessionService | None = None,
        builder: RunDependencyBuilder | None = None,
        compression_service_factory: CompressionServiceFactory | None = None,
        model_window_resolver: ModelWindowResolver | None = None,
    ) -> None:
        self._sessions = session_service or SessionService()
        self._builder = builder or RunDependencyBuilder()
        self._compression_service_factory = compression_service_factory
        self._model_window_resolver = model_window_resolver or resolve_model_context_window
        self._status_by_session: dict[str, ContextStatusResult] = {}

    def status(self, session_id_or_prefix: str) -> ContextStatusResult:
        record = self._sessions.resolve_record(_require_non_empty(session_id_or_prefix, "session_id"))
        status = self._status_by_session.get(record.session_id) or _status_from_record(record) or _default_status(record)
        return _enrich_status(record, status, model_window_resolver=self._model_window_resolver)

    def estimate(self, request: ContextEstimateRequest) -> ContextStatusResult:
        record = self._sessions.resolve_record(_require_non_empty(request.session_id, "session_id"))
        draft = request.draft if isinstance(request.draft, str) else ""
        attachments = request.attachments if isinstance(request.attachments, list) else []
        messages = [*record.messages]
        if draft.strip() or attachments:
            draft_message: dict[str, Any] = {"role": "user", "content": draft}
            if attachments:
                draft_message["attachments"] = attachments
            messages.append(draft_message)
        base = _default_status_for_messages(record, messages)
        previous = self._status_by_session.get(record.session_id) or _status_from_record(record)
        return replace(
            _enrich_status(record, base, model_window_resolver=self._model_window_resolver),
            compression_count=previous.compression_count if previous else 0,
            last_compression=previous.last_compression if previous else None,
            status=previous.status if previous else None,
            error=previous.error if previous else None,
        )

    def compress(self, request: ContextCompressRequest) -> ContextCompressResult:
        record = self._sessions.resolve_record(_require_non_empty(request.session_id, "session_id"))
        if not isinstance(request.force, bool):
            raise ServiceValidationError(
                "context compression requires boolean 'force'",
                details={"session_id": record.session_id},
            )
        focus = request.focus.strip() if isinstance(request.focus, str) and request.focus.strip() else None

        if len(record.messages) < 4:
            metadata = {
                "status": "skipped",
                "trigger_reason": "manual",
                "source_message_count": len(record.messages),
                "result_message_count": len(record.messages),
                "reason": "not_enough_context",
                "source_tokens": estimate_messages_tokens(record.messages),
                "result_tokens": estimate_messages_tokens(record.messages),
            }
            return self._store_status(
                record,
                status="skipped",
                metadata=metadata,
                error=None,
            )

        _validate_record_for_compression(record)
        before_messages = copy.deepcopy(record.messages)
        source_tokens = estimate_messages_tokens(before_messages)
        context = TurnContext(session_id=record.session_id, iteration=0, metadata={})
        service = self._build_compression_service(record)
        result = service.compress(
            CompressionRequest(
                messages=before_messages,
                context=context,
                trigger_reason="manual",
                force=request.force,
                focus=focus,
            )
        )
        result_tokens = estimate_messages_tokens(result.messages)
        metadata = {
            **dict(result.metadata),
            "trigger_reason": "manual",
            "source_tokens": source_tokens,
            "result_tokens": result_tokens,
        }
        if result.status == "compressed":
            self._sessions.persist_run_result(
                record.session_id,
                messages=result.messages,
                system_prompt=record.system_prompt,
            )
            record.messages = result.messages
        return self._store_status(
            record,
            status=result.status,
            metadata=metadata,
            error=result.error,
        )

    def reset_for_tests(self) -> None:
        self._status_by_session.clear()

    def _build_compression_service(self, record: SessionRecord) -> _CompressionService:
        if self._compression_service_factory is not None:
            return self._compression_service_factory(record)
        provider = self._builder.build_provider(record)
        config = self._builder.build_engine_config(AgentRunOptions())
        config.use_builtin_tools = False
        config.compression_enabled = True
        engine = AgentEngine(provider, config=config)
        return CompressionLifecycleService(
            context_engine=DefaultContextEngine(
                adapter=LegacyContextAssemblyAdapter(engine),
            )
        )

    def _store_status(
        self,
        record: SessionRecord,
        *,
        status: str,
        metadata: dict[str, Any],
        error: str | None,
    ) -> ContextCompressResult:
        previous = self._status_by_session.get(record.session_id) or _default_status(record)
        compression_count = previous.compression_count + (1 if status == "compressed" else 0)
        envelope = ContextCompressResult(
            session_id=record.session_id,
            context_engine="default",
            compression_count=compression_count,
            last_compression=dict(metadata),
            message_count=len(record.messages),
            token_estimate=estimate_messages_tokens(record.messages),
            provider=record.provider or None,
            model=record.model or None,
            status=status,
            error=error,
        )
        enriched = _enrich_status(record, envelope, model_window_resolver=self._model_window_resolver)
        self._status_by_session[record.session_id] = enriched
        record.metadata["context_status"] = asdict(enriched)
        self._sessions.persist_context_status(record.session_id, asdict(enriched))
        return enriched


def _default_status(record: SessionRecord) -> ContextStatusResult:
    return _default_status_for_messages(record, record.messages)


def _default_status_for_messages(record: SessionRecord, messages: list[dict[str, Any]]) -> ContextStatusResult:
    breakdown = _context_breakdown(record, messages)
    token_estimate = sum(item.tokens for item in breakdown)
    return ContextStatusResult(
        session_id=record.session_id,
        context_engine="default",
        compression_count=0,
        last_compression=None,
        message_count=len(messages),
        token_estimate=token_estimate,
        provider=record.provider or None,
        model=record.model or None,
        prompt_tokens=token_estimate,
        transcript_tokens=_breakdown_tokens(breakdown, "Transcript"),
        system_tokens=_breakdown_tokens(breakdown, "System prompt"),
        attachment_tokens=_breakdown_tokens(breakdown, "Attachments"),
        tool_result_tokens=_breakdown_tokens(breakdown, "Tool results"),
        pressure_level="unknown",
        next_action="none",
        breakdown=breakdown,
    )


def _status_from_record(record: SessionRecord) -> ContextStatusResult | None:
    payload = record.metadata.get("context_status") if isinstance(record.metadata, dict) else None
    if not isinstance(payload, dict):
        return None
    try:
        return ContextStatusResult(
            session_id=str(payload.get("session_id") or record.session_id),
            context_engine=str(payload.get("context_engine") or "default"),
            compression_count=int(payload.get("compression_count") or 0),
            last_compression=payload.get("last_compression") if isinstance(payload.get("last_compression"), dict) else None,
            message_count=int(payload.get("message_count") or len(record.messages)),
            token_estimate=int(payload.get("token_estimate") or estimate_messages_tokens(record.messages)),
            status=str(payload.get("status")) if payload.get("status") is not None else None,
            error=str(payload.get("error")) if payload.get("error") is not None else None,
        )
    except (TypeError, ValueError):
        return None


def _validate_record_for_compression(record: SessionRecord) -> None:
    if not record.provider.strip():
        raise ServiceValidationError(
            f"session has no provider: {record.session_id}",
            details={"session_id": record.session_id},
        )
    if not record.model.strip():
        raise ServiceValidationError(
            f"session has no model: {record.session_id}",
            details={"session_id": record.session_id},
        )


def _require_non_empty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceValidationError(f"{field} is required", details={field: value})
    return value.strip()


def _enrich_status(
    record: SessionRecord,
    status: StatusT,
    *,
    model_window_resolver: ModelWindowResolver,
) -> StatusT:
    breakdown = status.breakdown or _context_breakdown(record, record.messages)
    token_estimate = status.token_estimate or sum(item.tokens for item in breakdown)
    prompt_tokens = status.prompt_tokens or token_estimate
    provider = status.provider or record.provider or None
    model = status.model or record.model or None
    context_window = _valid_context_window(status.context_window) or model_window_resolver(provider, model)
    pressure_level, next_action = _pressure(prompt_tokens, context_window)
    return replace(
        status,
        provider=provider,
        model=model,
        context_window=context_window,
        token_estimate=token_estimate,
        prompt_tokens=prompt_tokens,
        transcript_tokens=status.transcript_tokens or _breakdown_tokens(breakdown, "Transcript"),
        system_tokens=status.system_tokens or _breakdown_tokens(breakdown, "System prompt"),
        attachment_tokens=status.attachment_tokens or _breakdown_tokens(breakdown, "Attachments"),
        tool_result_tokens=status.tool_result_tokens or _breakdown_tokens(breakdown, "Tool results"),
        pressure_level=pressure_level,
        next_action=next_action,
        breakdown=breakdown,
    )


def resolve_model_context_window(provider: str | None, model: str | None) -> int | None:
    if not provider or not model:
        return None
    raw_provider = provider.strip()
    raw_model = model.strip()
    if not raw_provider or not raw_model:
        return None
    try:
        provider_key = resolve_provider_name(raw_provider)
    except Exception:
        provider_key = raw_provider
    candidates = MODEL_CATALOG.get(provider_key) or MODEL_CATALOG.get(raw_provider) or []
    for candidate in candidates:
        if candidate.id == raw_model:
            return _valid_context_window(candidate.context_window)
    normalized_model = raw_model.lower()
    for candidate in candidates:
        if candidate.id.lower() == normalized_model:
            return _valid_context_window(candidate.context_window)
    return None


def _valid_context_window(value: int | None) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _context_breakdown(record: SessionRecord, messages: list[dict[str, Any]]) -> list[ContextBreakdownItem]:
    rows: list[ContextBreakdownItem] = []
    system_prompt = record.system_prompt.strip() if isinstance(record.system_prompt, str) else ""
    if system_prompt:
        rows.append(ContextBreakdownItem(label="System prompt", tokens=_estimate_text_tokens(system_prompt), detail="session system prompt"))
    transcript_messages = [message for message in messages if message.get("role") != "tool"]
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    transcript_tokens = estimate_messages_tokens(transcript_messages)
    if transcript_tokens > 0:
        rows.append(ContextBreakdownItem(label="Transcript", tokens=transcript_tokens, detail=str(len(transcript_messages)) + " messages"))
    tool_tokens = estimate_messages_tokens(tool_messages)
    if tool_tokens > 0:
        rows.append(ContextBreakdownItem(label="Tool results", tokens=tool_tokens, detail=str(len(tool_messages)) + " tool messages"))
    attachment_tokens = _estimate_attachment_tokens(messages)
    if attachment_tokens > 0:
        rows.append(ContextBreakdownItem(label="Attachments", tokens=attachment_tokens, detail="selected file or media context"))
    if not rows:
        rows.append(ContextBreakdownItem(label="Transcript", tokens=0, detail="empty session"))
    return rows


def _estimate_attachment_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        attachments = message.get("attachments")
        if not isinstance(attachments, list):
            continue
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            text = "\n".join(
                str(value)
                for key, value in attachment.items()
                if key in {"content", "_llm_content", "quote", "data"} and isinstance(value, str)
            )
            total += _estimate_text_tokens(text)
    return total


def _estimate_text_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, len(text) // 4)


def _breakdown_tokens(rows: list[ContextBreakdownItem], label: str) -> int:
    return sum(row.tokens for row in rows if row.label == label)


def _pressure(tokens: int, context_window: int | None) -> tuple[str, str]:
    if not context_window or context_window <= 0:
        return "unknown", "none"
    ratio = tokens / context_window
    if ratio >= 0.98:
        return "critical", "blocked"
    if ratio >= 0.85:
        return "high", "compress"
    if ratio >= 0.65:
        return "medium", "none"
    return "low", "none"


__all__ = ["ContextService", "resolve_model_context_window"]
