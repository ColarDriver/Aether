"""Base model provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from aether.config.schema import ModelCallConfig
from aether.runtime.contracts import NormalizedResponse, StreamDeltaCallback, TurnContext
from aether.tools.base import ToolDescriptor


class ModelProvider(ABC):
    """Provider contract used by AgentEngine."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        raise NotImplementedError

    def validate_response(
        self,
        response: NormalizedResponse,
    ) -> tuple[bool, list[str]]:
        """Inspect a freshly-returned response and report any structural issues.

        Sprint 1 / PR 1.1 — engine calls this immediately after
        ``generate()`` returns.  If the result is ``(False, [...])`` the
        engine raises ``ResponseInvalidError`` (carrying the reasons list)
        and routes through the recovery layer.

        Default implementation returns ``(True, [])`` — providers without
        any provider-specific shape rules opt out by inheriting this.
        Concrete providers SHOULD override to enforce their own invariants
        (e.g. chat-completions: ``choices`` non-empty, ``message`` is a
        dict; codex: ``output_text`` present; anthropic: non-empty
        ``content`` list).

        The returned reasons list is human-readable and ends up in
        observability logs and the recovery-decision trail.
        """
        return True, []

    def list_models(self) -> List[str]:
        """Return the model identifiers this provider can serve.

        Default returns an empty list — the CLI's ``/model`` picker treats
        an empty list as "no listing endpoint available; show the
        currently-active model only".  Subclasses that hit a real
        ``/v1/models`` style endpoint should override this.
        """
        return []

    def set_model(self, model: str) -> None:
        """Swap the active model in place (used by the ``/model`` picker).

        Default implementation just rebinds ``self.model`` if the attribute
        exists; providers with more complex state (e.g. cached tokenizers
        keyed by model) should override.
        """
        if hasattr(self, "model"):
            setattr(self, "model", model)
