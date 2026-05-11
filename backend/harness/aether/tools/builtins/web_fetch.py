"""Built-in ``web_fetch`` tool — Sprint 3.5 / PR 3.5.5.

Fetches an HTTP(S) URL, converts the body to markdown, and returns
a structured response that includes the model's free-form ``prompt``
verbatim so the surrounding turn can run its extraction logic over the
markdown without an extra round-trip to a utility model.

Why a dedicated tool (instead of letting the model ``shell: curl ...``):

* SSRF defence — :func:`aether.runtime.web_safety.is_url_safe` rejects
  loopback / private / link-local / multicast IPs *and* unsupported
  schemes (``file:`` / ``ftp:`` / ``gopher:``).  ``shell`` cannot give
  this guarantee.
* HTML → markdown — raw HTML burns 5–10x more tokens than the
  semantic markdown form; we run ``markdownify`` server-side so the
  model only ever sees the cheap representation.
* Disk spill (Sprint 3.5 / PR 3.5.1) — pages bigger than
  :data:`WebFetchTool.MAX_RESULT_CHARS` get spilled to
  ``~/.aether/tool_results/<session>/<call>.md`` and only a preview
  comes back inline.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.interrupt_messages import FETCH_INTERRUPTED_MESSAGE
from aether.runtime.web_safety import is_url_safe
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool

logger = logging.getLogger(__name__)


_DEFAULT_USER_AGENT = "Aether/0.1 (+https://github.com/ColarDriver/Aether)"


class WebFetchTool(ToolExecutor):
    interrupt_behavior = "cancel"
    """Fetch a URL, convert to markdown, return for the model to read."""

    NAME = "web_fetch"
    MAX_RESULT_CHARS = 80_000
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024  # 5 MiB
    MAX_REDIRECTS = 5

    def __init__(
        self,
        *,
        client_factory: Optional[Any] = None,
        user_agent: str = _DEFAULT_USER_AGENT,
    ) -> None:
        # Optional injection seam for tests — pass any callable that
        # returns an httpx.Client-compatible context manager.  Production
        # uses ``httpx.Client`` configured per-call from EngineConfig.
        self._client_factory = client_factory
        self._user_agent = user_agent
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Fetch a public HTTP(S) URL, convert HTML to markdown, and "
                "return the result so the calling model can extract the "
                "information described in `prompt`. Refuses to access "
                "loopback, private and link-local addresses (SSRF guard); "
                "responses larger than ~80 KB are spilled to disk and only "
                "a preview is returned inline."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "format": "uri",
                        "description": "Absolute http:// or https:// URL.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What to extract from the page. The tool does not "
                            "run inference itself; it just appends this prompt "
                            "next to the markdown so your subsequent reasoning "
                            "can target it."
                        ),
                    },
                },
                "required": ["url", "prompt"],
            },
            required=["url", "prompt"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    # ---------------------------------------------------------- execute

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        url = args.get("url")
        prompt = args.get("prompt", "")

        config = context.metadata.get("_engine_config") if context.metadata else None
        if not bool(getattr(config, "web_fetch_enabled", True)):
            return _error(call, "web_fetch is disabled by configuration")

        if not isinstance(url, str) or not url.strip():
            return _error(call, "'url' is required and must be a string")
        if not isinstance(prompt, str):
            return _error(call, "'prompt' must be a string")

        ok, reason = is_url_safe(url)
        if not ok:
            return _error(call, f"refused: {reason}", metadata={"url": url})

        timeout = float(getattr(config, "web_fetch_timeout_seconds", self.DEFAULT_TIMEOUT))
        max_bytes = int(
            getattr(config, "web_fetch_max_download_bytes", self.DEFAULT_MAX_DOWNLOAD_BYTES)
        )

        try:
            response_status, response_body, response_ct = self._http_get(
                url, timeout=timeout, max_bytes=max_bytes, context=context
            )
        except httpx.TimeoutException:
            return _error(call, f"fetch timed out after {timeout:.0f}s", metadata={"url": url})
        except httpx.HTTPError as exc:
            if context.interrupt_signal is not None and context.interrupt_signal.is_aborted():
                return _error(call, FETCH_INTERRUPTED_MESSAGE, metadata={"url": url, "interrupted": True})
            return _error(call, f"fetch failed: {exc}", metadata={"url": url})
        except OSError as exc:
            return _error(call, f"network error: {exc}", metadata={"url": url})

        markdown = self._html_to_markdown(response_body, content_type=response_ct)
        full_output = self._format_output(
            url=url,
            prompt=prompt,
            status=response_status,
            markdown=markdown,
            byte_count=len(response_body),
        )
        content = maybe_spill_for_tool(
            full_output,
            call=call,
            context=context,
            max_chars=self.MAX_RESULT_CHARS,
            extension="md",
            full_lines=full_output.count("\n") + 1,
            full_bytes=len(response_body),
        )
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=False,
            metadata={
                "url": url,
                "status": response_status,
                "content_type": response_ct,
                "bytes_downloaded": len(response_body),
                "markdown_chars": len(markdown),
            },
        )

    # ------------------------------------------------------------- impl

    def _http_get(
        self, url: str, *, timeout: float, max_bytes: int, context: TurnContext
    ) -> tuple[int, bytes, str]:
        if self._client_factory is not None:
            ctx = self._client_factory()
        else:
            ctx = httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                max_redirects=self.MAX_REDIRECTS,
                headers={"User-Agent": self._user_agent, "Accept": "text/html, */*;q=0.5"},
            )
        with ctx as client:
            listener = None
            if context.interrupt_signal is not None:
                listener = lambda _reason: client.close()
                context.interrupt_signal.add_listener(listener)
            try:
                response = client.get(url)
                content = response.content[:max_bytes]
                content_type = response.headers.get("content-type", "")
                return int(response.status_code), bytes(content), str(content_type)
            finally:
                if context.interrupt_signal is not None and listener is not None:
                    context.interrupt_signal.remove_listener(listener)

    @staticmethod
    def _html_to_markdown(body: bytes, *, content_type: str) -> str:
        ctype = (content_type or "").split(";", 1)[0].strip().lower()
        try:
            text = body.decode("utf-8", errors="replace")
        except Exception:
            text = body.decode("latin-1", errors="replace")
        if not text:
            return ""
        if ctype in {"application/json", "application/ld+json"}:
            try:
                parsed = json.loads(text)
                return "```json\n" + json.dumps(parsed, indent=2, ensure_ascii=False) + "\n```"
            except Exception:
                return text
        if ctype.startswith("text/") and ctype != "text/html":
            return text
        try:
            from markdownify import markdownify  # type: ignore
        except ImportError:
            return text
        try:
            return markdownify(text, heading_style="ATX")
        except Exception as exc:
            logger.warning("markdownify failed: %s", exc)
            return text

    @staticmethod
    def _format_output(
        *,
        url: str,
        prompt: str,
        status: int,
        markdown: str,
        byte_count: int,
    ) -> str:
        return (
            f"# Fetched {url}\n"
            f"- HTTP status: {status}\n"
            f"- Bytes downloaded: {byte_count}\n"
            f"\n## Prompt\n{prompt.strip()}\n"
            f"\n## Content (markdown)\n{markdown}"
        )


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


__all__ = ["WebFetchTool"]
