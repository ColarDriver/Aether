"""Stream callback construction for provider invocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from aether.config.schema import EngineConfig
from aether.runtime.core.contracts import (
    EngineRequest,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.runtime.core.exceptions import EngineInterrupted
from aether.runtime.core.services import EngineServices
from aether.runtime.session.session_runtime import TURN_KEY_STREAMED_ASSISTANT_TEXT


@dataclass(slots=True)
class StreamCallbacks:
    visible: StreamDeltaCallback | None = None
    silent: StreamSilentCallback | None = None


class StreamEventAdapter(Protocol):
    def is_interrupted(self, session_id: str, context: TurnContext | None = None) -> bool: ...


class LegacyStreamEventAdapter:
    """Delegate interrupt checks to the existing engine helper."""

    def __init__(self, engine: object) -> None:
        self._engine = engine

    def is_interrupted(self, session_id: str, context: TurnContext | None = None) -> bool:
        return getattr(self._engine, "_is_interrupted")(session_id, context)


class StreamController:
    """Create visible and silent callbacks for one turn."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        services: EngineServices,
        adapter: StreamEventAdapter,
    ) -> None:
        self._config = config
        self._services = services
        self._adapter = adapter

    def build_callbacks(
        self,
        *,
        request: EngineRequest,
        context: TurnContext,
    ) -> StreamCallbacks:
        return StreamCallbacks(
            visible=self.build_visible_callback(request=request, context=context),
            silent=self.build_silent_callback(request=request, context=context),
        )

    def build_visible_callback(
        self,
        *,
        request: EngineRequest,
        context: TurnContext,
    ) -> StreamDeltaCallback | None:
        callback = request.stream_callback

        if not getattr(self._config, "streaming_enabled", True):
            self._services.logger.debug(
                "streaming_enabled=False; suppressing stream_callback for session %s",
                request.session_id,
            )
            return None

        if callback is None and not getattr(
            self._config,
            "empty_response_partial_stream_recovery_enabled",
            True,
        ):
            return None

        def _wrapped(delta: str) -> None:
            if not isinstance(delta, str) or not delta:
                return

            if self._adapter.is_interrupted(request.session_id, context):
                partial = str(
                    context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or ""
                )
                raise EngineInterrupted(
                    reason="user-interrupt",
                    partial_text=partial,
                    was_in_tool_call=False,
                )

            if getattr(
                self._config,
                "empty_response_partial_stream_recovery_enabled",
                True,
            ):
                current = str(
                    context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or ""
                )
                context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT] = current + delta
            if callback is None:
                return
            try:
                callback(delta)
                context.metadata["streamed_output"] = True
                context.metadata["stream_callback_calls"] = (
                    int(context.metadata.get("stream_callback_calls", 0)) + 1
                )
            except Exception:
                self._services.logger.exception("stream callback failed")

        return _wrapped

    def build_silent_callback(
        self,
        *,
        request: EngineRequest,
        context: TurnContext,
    ) -> StreamSilentCallback | None:
        callback = request.stream_silent_callback
        if callback is None:
            return None

        if not getattr(self._config, "streaming_enabled", True):
            return None

        def _wrapped_silent(delta: str) -> None:
            if not isinstance(delta, str) or not delta:
                return
            if self._adapter.is_interrupted(request.session_id, context):
                partial = str(
                    context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or ""
                )
                raise EngineInterrupted(
                    reason="user-interrupt",
                    partial_text=partial,
                    was_in_tool_call=False,
                )
            try:
                callback(delta)
                context.metadata["stream_silent_callback_calls"] = (
                    int(context.metadata.get("stream_silent_callback_calls", 0)) + 1
                )
            except Exception:
                self._services.logger.exception("stream silent callback failed")

        return _wrapped_silent


__all__ = [
    "LegacyStreamEventAdapter",
    "StreamCallbacks",
    "StreamController",
    "StreamEventAdapter",
]
