"""Built-in ``web_search`` tool.

Calls a local search backend and returns top-N results formatted as
markdown.  This is intentionally separate from provider-native hosted
search tools (for example Claude server-side web search): Aether's
OpenAI-compatible path needs a local backend it can run regardless of
model provider.

Configuration:

* ``EngineConfig.web_search_provider`` or ``WEB_SEARCH_PROVIDER`` chooses
  the backend (``brave`` by default).
* ``EngineConfig.web_search_api_key`` or ``WEB_SEARCH_API_KEY`` supplies
  a generic key.

When no usable key is set the tool returns a structured ``is_error=True``
result that explains the missing configuration so the model can adapt
without crashing the turn.

The implementation deliberately keeps an injection seam
(``client_factory``) so unit tests can swap in a stub instead of
making real HTTP calls.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool

logger = logging.getLogger(__name__)


_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_TAVILY_BASE_URL = "https://api.tavily.com"
_BOCHA_ENDPOINT = "https://api.bochaai.com/v1/web-search"
_DEFAULT_USER_AGENT = "Aether/0.1 (+https://github.com/ColarDriver/Aether)"
_SUPPORTED_PROVIDERS = {"brave", "tavily", "bocha"}


class WebSearchTool(ToolExecutor):
    interrupt_behavior = "cancel"
    """Search the web via a configured local backend."""

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
                "(brave by default; also supports tavily and bocha) and "
                "return up to `max_results` "
                "matches as markdown. Use this for finding documentation, "
                "GitHub issues, blog posts, etc. Returns an error if the "
                "search provider API key is missing — set WEB_SEARCH_API_KEY "
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
                    "allowed_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only include results from these domains.",
                    },
                    "blocked_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exclude results from these domains.",
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

        allowed_domains = _coerce_domains(args.get("allowed_domains"))
        blocked_domains = _coerce_domains(args.get("blocked_domains"))
        if allowed_domains is None:
            return _error(call, "'allowed_domains' must be an array of strings")
        if blocked_domains is None:
            return _error(call, "'blocked_domains' must be an array of strings")
        if allowed_domains and blocked_domains:
            return _error(call, "allowed_domains and blocked_domains cannot both be set")

        provider = _resolve_provider(config)
        if provider not in _SUPPORTED_PROVIDERS:
            return _error(
                call,
                f"web_search provider {provider!r} is not supported "
                "(supported: brave, tavily, bocha)",
            )

        api_key = _resolve_api_key(config)
        if not api_key:
            return _error(
                call,
                "WebSearch unavailable: no API key configured. "
                "Set WEB_SEARCH_API_KEY or EngineConfig.web_search_api_key "
                f"to use the {provider} backend.",
            )

        max_results = self._coerce_max_results(args.get("max_results"))

        try:
            results = self._search(
                provider,
                query.strip(),
                api_key=api_key,
                max_results=max_results,
                context=context,
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

        results = _filter_results_by_domains(
            results,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )
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
                "allowed_domains": allowed_domains,
                "blocked_domains": blocked_domains,
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

    def _search(
        self,
        provider: str,
        query: str,
        *,
        api_key: str,
        max_results: int,
        context: TurnContext,
    ) -> list[dict[str, str]]:
        if provider == "brave":
            return self._brave_search(
                query,
                api_key=api_key,
                max_results=max_results,
                context=context,
            )
        if provider == "tavily":
            return self._tavily_search(
                query,
                api_key=api_key,
                max_results=max_results,
                context=context,
            )
        if provider == "bocha":
            return self._bocha_search(
                query,
                api_key=api_key,
                max_results=max_results,
                context=context,
            )
        raise ValueError(f"unsupported web_search provider: {provider}")

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
                response = client.get(
                    os.getenv("BRAVE_SEARCH_ENDPOINT") or self._endpoint,
                    params=params,
                )
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

    def _tavily_search(
        self, query: str, *, api_key: str, max_results: int, context: TurnContext
    ) -> list[dict[str, str]]:
        base_url = os.getenv("TAVILY_BASE_URL", _TAVILY_BASE_URL).rstrip("/")
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": min(max_results, self.MAX_RESULTS_HARD_CAP),
            "include_raw_content": False,
            "include_images": False,
        }

        if self._client_factory is not None:
            ctx = self._client_factory()
        else:
            ctx = httpx.Client(
                timeout=self.DEFAULT_TIMEOUT,
                headers={"User-Agent": _DEFAULT_USER_AGENT},
            )
        with ctx as client:
            listener = None
            if context.interrupt_signal is not None:
                listener = lambda _reason: client.close()
                context.interrupt_signal.add_listener(listener)
            try:
                response = client.post(f"{base_url}/search", json=payload)
                response.raise_for_status()
                data = response.json()
            finally:
                if context.interrupt_signal is not None and listener is not None:
                    context.interrupt_signal.remove_listener(listener)

        out: list[dict[str, str]] = []
        for item in (data.get("results") or [])[:max_results]:
            out.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                    "snippet": str(
                        item.get("content")
                        or item.get("snippet")
                        or item.get("description")
                        or ""
                    ).strip(),
                }
            )
        return out

    def _bocha_search(
        self, query: str, *, api_key: str, max_results: int, context: TurnContext
    ) -> list[dict[str, str]]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": _DEFAULT_USER_AGENT,
        }
        payload = {"query": query, "count": max_results, "summary": True}

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
                response = client.post(
                    os.getenv("BOCHA_SEARCH_ENDPOINT") or _BOCHA_ENDPOINT,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            finally:
                if context.interrupt_signal is not None and listener is not None:
                    context.interrupt_signal.remove_listener(listener)

        web_pages = ((data.get("data") or {}).get("webPages") or {})
        items = (
            web_pages.get("value")
            or web_pages.get("results")
            or data.get("results")
            or []
        )
        out: list[dict[str, str]] = []
        for item in items[:max_results]:
            out.append(
                {
                    "title": str(item.get("name") or item.get("title") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                    "snippet": str(
                        item.get("summary")
                        or item.get("snippet")
                        or item.get("description")
                        or ""
                    ).strip(),
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


def _resolve_provider(config: Any) -> str:
    configured = getattr(config, "web_search_provider", None)
    raw = (
        configured
        or os.getenv("WEB_SEARCH_PROVIDER")
        or "brave"
    )
    return str(raw).strip().lower().replace("_", "-")


def _resolve_api_key(config: Any) -> str | None:
    configured = getattr(config, "web_search_api_key", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    generic = os.getenv("WEB_SEARCH_API_KEY")
    if generic and generic.strip():
        return generic.strip()
    return None


def _coerce_domains(raw: Any) -> list[str] | None:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return None
    domains: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            return None
        normalized = _normalize_domain(item)
        if normalized:
            domains.append(normalized)
    return domains


def _filter_results_by_domains(
    results: list[dict[str, str]],
    *,
    allowed_domains: list[str],
    blocked_domains: list[str],
) -> list[dict[str, str]]:
    if not allowed_domains and not blocked_domains:
        return results
    filtered: list[dict[str, str]] = []
    for result in results:
        host = _host_for_url(result.get("url", ""))
        if allowed_domains and not any(
            _domain_matches(host, domain) for domain in allowed_domains
        ):
            continue
        if blocked_domains and any(
            _domain_matches(host, domain) for domain in blocked_domains
        ):
            continue
        filtered.append(result)
    return filtered


def _normalize_domain(value: str) -> str:
    stripped = value.strip().lower()
    if not stripped:
        return ""
    if "://" not in stripped:
        stripped = f"https://{stripped}"
    host = urlparse(stripped).hostname or ""
    return host[4:] if host.startswith("www.") else host


def _host_for_url(value: str) -> str:
    try:
        host = urlparse(value).hostname or ""
    except ValueError:
        host = ""
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def _domain_matches(host: str, domain: str) -> bool:
    return bool(host) and (host == domain or host.endswith(f".{domain}"))


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
