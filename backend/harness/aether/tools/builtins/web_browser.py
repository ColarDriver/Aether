"""Built-in ``web_browser`` tool — Sprint 3.5 / PR-3 (PR 3.5.10).

Headless Chromium navigation via Playwright.  Complements
:class:`WebFetchTool` for JavaScript-rendered pages — WebFetch returns
the pre-render HTML, WebBrowser waits for the page's JS to settle and
returns the rendered DOM (or a screenshot, click result, or form fill
result).

Default state
-------------
``EngineConfig.web_browser_enabled`` is **False** by default.  The tool
will refuse with a clear message until the operator opts in, and even
then a missing Playwright install or unavailable Chromium binary
surfaces as a friendly error rather than a crash.

SSRF
----
We share the same :func:`is_url_safe` guard the other web tools use,
so loopback / private IP / unsupported scheme URLs are rejected before
any browser process starts.  Note that the v1 implementation does not
re-check redirect destinations (Playwright follows redirects
internally) — the docs flag this as a known limitation to revisit when
we add user-facing browse confirmations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from aether.runtime.browser_manager import BrowserManager, BrowserUnavailable
from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tool_result_storage import resolve_spill_dir
from aether.runtime.web_safety import is_url_safe
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool

logger = logging.getLogger(__name__)


__all__ = ["WebBrowserTool"]


_OPERATIONS: frozenset[str] = frozenset({"fetch", "screenshot", "click", "type"})


class WebBrowserTool(ToolExecutor):
    NAME = "web_browser"
    MAX_RESULT_CHARS = 80_000
    DEFAULT_NAVIGATION_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        manager: Optional[BrowserManager] = None,
        *,
        navigation_timeout_seconds: int = DEFAULT_NAVIGATION_TIMEOUT_SECONDS,
    ) -> None:
        self._manager = manager
        self._navigation_timeout = int(navigation_timeout_seconds)
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Render a URL in headless Chromium (via Playwright) and "
                "either return its post-JS DOM as markdown, take a "
                "screenshot, click an element or fill a form. Disabled "
                "by default; enable via EngineConfig.web_browser_enabled "
                "after `pip install playwright && playwright install "
                "chromium`. Use web_fetch first for static pages — this "
                "tool is much heavier."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": sorted(_OPERATIONS),
                    },
                    "url": {
                        "type": "string",
                        "format": "uri",
                        "description": "Absolute http(s) URL.",
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for click / type.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type for the type op.",
                    },
                    "wait_for": {
                        "type": "string",
                        "enum": ["load", "domcontentloaded", "networkidle"],
                        "default": "domcontentloaded",
                    },
                    "wait_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Extra sleep after page load.",
                    },
                },
                "required": ["operation", "url"],
            },
            required=["operation", "url"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    # ------------------------------------------------------------- execute

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        op = args.get("operation")
        url = args.get("url")

        if op not in _OPERATIONS:
            return _error(call, f"unknown operation: {op!r}; supported: {sorted(_OPERATIONS)}")
        if not isinstance(url, str) or not url.strip():
            return _error(call, "'url' is required and must be a string")

        config = context.metadata.get("_engine_config") if context.metadata else None
        if not bool(getattr(config, "web_browser_enabled", False)):
            return _error(
                call,
                "web_browser is disabled by default. Set "
                "EngineConfig.web_browser_enabled=True after installing "
                "Playwright (`pip install playwright && playwright "
                "install chromium`) to enable.",
            )

        ok, reason = is_url_safe(url)
        if not ok:
            return _error(call, f"refused: {reason}", metadata={"url": url})

        if op in ("click", "type") and not isinstance(args.get("selector"), str):
            return _error(call, f"'selector' is required for the {op} op")
        if op == "type" and not isinstance(args.get("text"), str):
            return _error(call, "'text' is required for the type op")

        manager = self._manager or (
            context.metadata.get("_browser_manager") if context.metadata else None
        )
        if manager is None:
            idle = int(getattr(config, "web_browser_idle_timeout_seconds", 30))
            manager = BrowserManager(idle_timeout_seconds=idle)
            self._manager = manager

        nav_timeout = int(
            getattr(config, "web_browser_navigation_timeout_seconds", self._navigation_timeout)
        )

        try:
            browser = manager.get_browser()
        except BrowserUnavailable as exc:
            return _error(call, str(exc), metadata={"url": url})
        except Exception as exc:  # pragma: no cover - defensive
            return _error(call, f"browser error: {exc}", metadata={"url": url})

        try:
            from playwright.sync_api import TimeoutError as PWTimeoutError  # type: ignore
        except ImportError:
            PWTimeoutError = TimeoutError  # type: ignore[assignment]

        page = None
        try:
            page = browser.new_page()
            page.set_default_navigation_timeout(nav_timeout * 1000)
            wait_for = args.get("wait_for", "domcontentloaded") or "domcontentloaded"
            page.goto(url, wait_until=wait_for)
            extra = float(args.get("wait_seconds", 0) or 0)
            if extra > 0:
                page.wait_for_timeout(extra * 1000)

            if op == "fetch":
                body = self._do_fetch(page, url)
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
                    metadata={"operation": op, "url": url},
                )

            if op == "click":
                page.click(args["selector"])
                page.wait_for_load_state(wait_for)
                body = self._do_fetch(
                    page, url, header=f"# Browser clicked {args['selector']!r} on {url}"
                )
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
                    metadata={"operation": op, "url": url, "selector": args["selector"]},
                )

            if op == "type":
                selector = args["selector"]
                text = args["text"]
                page.fill(selector, text)
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=(
                        f"# Browser filled {selector!r} on {url}\n"
                        f"- characters typed: {len(text)}\n"
                    ),
                    is_error=False,
                    metadata={"operation": op, "url": url, "selector": selector, "chars": len(text)},
                )

            # screenshot
            spill_dir = resolve_spill_dir(
                session_id=context.session_id,
                config_dir=getattr(config, "tool_result_spill_dir", None),
            )
            out_path = spill_dir / f"{call.id}.png"
            page.screenshot(path=str(out_path), full_page=True)
            try:
                size = out_path.stat().st_size
            except OSError:
                size = -1
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=(
                    f"# Browser screenshot saved\n"
                    f"- url: {url}\n"
                    f"- path: {out_path}\n"
                    f"- size: {size} bytes\n"
                ),
                is_error=False,
                metadata={"operation": op, "url": url, "screenshot_path": str(out_path), "bytes": size},
            )

        except PWTimeoutError as exc:  # type: ignore[misc]
            return _error(call, f"browser timed out: {exc}", metadata={"url": url})
        except Exception as exc:
            return _error(call, f"browser error: {exc}", metadata={"url": url})
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass

    # ----------------------------------------------------------- helpers

    @staticmethod
    def _do_fetch(page: Any, url: str, *, header: Optional[str] = None) -> str:
        try:
            from markdownify import markdownify  # type: ignore
        except ImportError:
            markdownify = None  # type: ignore[assignment]
        title = ""
        try:
            title = page.title() or ""
        except Exception:
            title = ""
        try:
            html = page.content()
        except Exception as exc:
            return (header or f"# Browser fetched {url}") + f"\n\n_(failed to read page content: {exc})_\n"
        body = (
            (markdownify(html, heading_style="ATX") if markdownify else html)
            if html
            else "_(empty page)_"
        )
        head = header or f"# Browser fetched {url}"
        meta = [head]
        if title:
            meta.append(f"- title: {title}")
        meta.append(f"- bytes (raw HTML): {len(html or '')}")
        return "\n".join(meta) + "\n\n## Content (markdown)\n" + body + "\n"


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
