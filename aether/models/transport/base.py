"""Transport contract for provider-specific data conversion."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from aether.config.schema import ModelCallConfig
from aether.runtime.core.contracts import NormalizedResponse, TurnContext
from aether.tools.base import ToolDescriptor


class ProviderTransport(Protocol):
    """Pure provider conversion boundary.

    A transport converts canonical Aether messages/tools into provider-native
    request payloads and projects raw provider responses back into
    ``NormalizedResponse``. Network IO, credential lookup, retry policy, and
    client lifecycle stay in ``ModelProvider`` implementations.
    """

    api_mode: str

    def convert_messages(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        ...

    def convert_tools(
        self,
        tools: Iterable[ToolDescriptor],
        **kwargs: Any,
    ) -> Any:
        ...

    def build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Iterable[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...

    def normalize_response(
        self,
        response: Any,
        **kwargs: Any,
    ) -> NormalizedResponse:
        ...

    def validate_raw_response(self, response: Any) -> tuple[bool, list[str]]:
        ...


__all__ = ["ProviderTransport"]
