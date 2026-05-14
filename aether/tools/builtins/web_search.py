"""Built-in ``web_search`` tool.

Calls the Brave Search REST API and returns top-N results formatted
as markdown.  Brave Search is selected for v1 because:

* JSON API (no HTML scraping).
* No CAPTCHAs at the free tier (vs. Google).
* 2 000 queries / month free tier — enough for development and most
  individual users.

Configuration: the API key comes from
``EngineConfig.web_search_api_key`` (highest priority) or the
``BRAVE_API_KEY`` environment variable.  When neither is set the tool
returns a structured ``is_error=True`` result that explains the missing
configuration so the model can adapt without crashing the turn.

The implementation deliberately keeps an injection seam
(``client_factory``) so unit tests can swap in a stub instead of
making real HTTP calls.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool

logger = logging.getLogger(__name__)


_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_USER_AGENT = "Aether/0.1 (+https://github.com/ColarDriver/Aether)"


class WebSearchTool(ToolExecutor):
    interrupt_behavior = "cancel"
    """Search the web via Brave Search; return top-N results as markdown."""

    NAME = "web_search"
    MAX_RESULT_CHARS = 30_000
    DEFAULT_TIMEOUT = 15.0
    MAX_RESULTS_HARD_CAP = 20
    DEFAULT_MAX_RESULTS = 10

    def __init__(
        self,
        *,
        client_factory: Optional[Any] = None,
        endpoint: str = _BRAVE_ENDPOINT,
    ) -> None:
        self._client_factory = client_factory
        self._endpoint = endpoint
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Run a web search via the configured search provider "
                "(Brave Search by default) and return up to `max_results` "
                "matches as markdown. Use this for finding documentation, "
                "GitHub issues, blog posts, etc. Returns an error if the "
                "search provider API key is missing — set BRAVE_API_KEY "
                "or EngineConfig.web_search_api_key."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self.MAX_RESULTS_HARD_CAP,
                        "default": self.DEFAULT_MAX_RESULTS,
                    },
                },
                "required": ["query"],
            },
            required=["query"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return _error(call, "'query' is required and must be a non-empty string")

        config = context.metadata.get("_engine_config") if context.metadata else None
        if not bool(getattr(config, "web_search_enabled", True)):
            return _error(call, "web_search is disabled by configuration")

        provider = str(getattr(config, "web_search_provider", "brave") or "brave").lower()
        if provider != "brave":
            return _error(
                call,
                f"web_search provider {provider!r} not implemented in v1 "
                "(only 'brave' is supported)",
            )

        api_key = getattr(config, "web_search_api_key", None) or os.environ.get(
            "BRAVE_API_KEY"
        )
        if not api_key:
            return _error(
                call,
                "WebSearch unavailable: no API key configured. "
                "Set BRAVE_API_KEY environment variable or "
                "EngineConfig.web_search_api_key to use this tool.",
            )

        max_results = self._coerce_max_results(args.get("max_results"))

        try:
            results = self._brave_search(
                query.strip(), api_key=api_key, max_results=max_results, context=context
            )
        except httpx.TimeoutException:
            return _error(call, "search timed out")
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            return _error(call, f"search returned HTTP {status}")
        except httpx.HTTPError as exc:
            if context.interrupt_signal is not None and context.interrupt_signal.is_aborted():
                return _error(call, "search interrupted by user", metadata={"interrupted": True})
            return _error(call, f"search failed: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("web_search unexpected failure")
            return _error(call, f"search failed: {exc}")

        body = self._format_results(query.strip(), results)
        content = maybe_spill_for_tool(
            body,
            call=call,
            context=context,
            max_chars=self.MAX_RESULT_CHARS,
            extension="md",
            full_lines=body.count("\n") + 1,
        )
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=False,
            metadata={
                "query": query.strip(),
                "provider": provider,
                "result_count": len(results),
                "max_results": max_results,
            },
        )

    # ---------------------------------------------------------- helpers

    def _coerce_max_results(self, raw: Any) -> int:
        if raw is None:
            return self.DEFAULT_MAX_RESULTS
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return self.DEFAULT_MAX_RESULTS
        if value < 1:
            return 1
        if value > self.MAX_RESULTS_HARD_CAP:
            return self.MAX_RESULTS_HARD_CAP
        return value

    def _brave_search(
        self, query: str, *, api_key: str, max_results: int, context: TurnContext
    ) -> list[dict[str, str]]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
            "User-Agent": _DEFAULT_USER_AGENT,
        }
        params = {"q": query, "count": max_results}

        if self._client_factory is not None:
            ctx = self._client_factory()
        else:
            ctx = httpx.Client(timeout=self.DEFAULT_TIMEOUT, headers=headers)
        with ctx as client:
            listener = None
            if context.interrupt_signal is not None:
                listener = lambda _reason: client.close()
                context.interrupt_signal.add_listener(listener)
            try:
                response = client.get(self._endpoint, params=params)
                response.raise_for_status()
                data = response.json()
            finally:
                if context.interrupt_signal is not None and listener is not None:
                    context.interrupt_signal.remove_listener(listener)

        web = data.get("web") or {}
        items = web.get("results") or []
        out: list[dict[str, str]] = []
        for item in items[:max_results]:
            out.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                    "snippet": str(item.get("description") or "").strip(),
                }
            )
        return out

    @staticmethod
    def _format_results(query: str, results: list[dict[str, str]]) -> str:
        if not results:
            return f"# Web search: {query}\n\nNo results found.\n"
        lines = [f"# Web search: {query}", "", f"Found {len(results)} results:", ""]
        for i, r in enumerate(results, 1):
            title = r["title"] or "(no title)"
            lines.append(f"{i}. **{title}**")
            if r["url"]:
                lines.append(f"   {r['url']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet']}")
            lines.append("")
        return "\n".join(lines)


def _error(
    call: ToolCall,
    message: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=message,
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["WebSearchTool"]
