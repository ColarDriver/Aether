"""Tests for ``aether.tools.builtins.web_browser.WebBrowserTool``.

Sprint 3.5 / PR-3 (PR 3.5.10).

We use a stub :class:`BrowserManager`-shaped object that returns a
fake browser/page so the tool's argument validation, SSRF gate,
config gate, dispatch + result formatting can all be exercised
without Playwright actually being installed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

from aether.config.schema import EngineConfig
from aether.runtime.contracts import ToolCall, TurnContext
from aether.runtime.browser_manager import BrowserUnavailable
from aether.tools.builtins.web_browser import WebBrowserTool


class _FakePage:
    def __init__(self, *, html: str = "<h1>hello</h1><p>world</p>", title: str = "Sample") -> None:
        self.html = html
        self._title = title
        self.goto_calls: list[tuple[str, str]] = []
        self.click_calls: list[str] = []
        self.fill_calls: list[tuple[str, str]] = []
        self.screenshot_path: Optional[str] = None
        self.closed = False
        self.timeout_ms: Optional[int] = None
        self.set_default_navigation_timeout_called = False
        self._extra_wait_ms = 0

    def set_default_navigation_timeout(self, ms: int) -> None:
        self.timeout_ms = ms
        self.set_default_navigation_timeout_called = True

    def goto(self, url: str, *, wait_until: str = "domcontentloaded") -> None:
        self.goto_calls.append((url, wait_until))

    def wait_for_timeout(self, ms: int) -> None:
        self._extra_wait_ms += ms

    def wait_for_load_state(self, state: str) -> None:  # pragma: no cover - trivial
        pass

    def title(self) -> str:
        return self._title

    def content(self) -> str:
        return self.html

    def click(self, selector: str) -> None:
        self.click_calls.append(selector)

    def fill(self, selector: str, value: str) -> None:
        self.fill_calls.append((selector, value))

    def screenshot(self, *, path: str, full_page: bool = True) -> None:
        Path(path).write_bytes(b"PNGDATA" * 100)
        self.screenshot_path = path

    def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self, *, page: Optional[_FakePage] = None, raise_on_new_page: bool = False) -> None:
        self.page = page or _FakePage()
        self.raise_on_new_page = raise_on_new_page
        self.created = 0

    def new_page(self) -> _FakePage:
        self.created += 1
        if self.raise_on_new_page:
            raise RuntimeError("page failed to launch")
        return self.page

    def is_connected(self) -> bool:
        return True


class _FakeManager:
    def __init__(
        self,
        *,
        browser: Optional[_FakeBrowser] = None,
        raises: Optional[Exception] = None,
    ) -> None:
        self.browser = browser
        self.raises = raises
        self.touch_count = 0
        self.shutdown_count = 0

    def get_browser(self) -> _FakeBrowser:
        if self.raises is not None:
            raise self.raises
        return self.browser or _FakeBrowser()

    def touch(self) -> None:
        self.touch_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


def _ctx(*, config: Optional[EngineConfig] = None) -> TurnContext:
    cfg = config or EngineConfig(web_browser_enabled=True)
    return TurnContext(session_id="ses-browser", iteration=0, metadata={"_engine_config": cfg})


class GateAndValidationTests(unittest.TestCase):
    def test_a1_disabled_by_default(self) -> None:
        # EngineConfig() has web_browser_enabled=False.
        tool = WebBrowserTool()
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "fetch", "url": "https://example.com"}),
            TurnContext(session_id="s1", iteration=0, metadata={"_engine_config": EngineConfig()}),
        )
        self.assertTrue(out.is_error)
        self.assertIn("disabled by default", out.content)

    def test_a2_unknown_operation_rejected(self) -> None:
        tool = WebBrowserTool()
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "magic", "url": "https://example.com"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("unknown operation", out.content)

    def test_a3_missing_url_rejected(self) -> None:
        tool = WebBrowserTool()
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "fetch"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("'url'", out.content)

    def test_a4_unsupported_scheme_rejected(self) -> None:
        tool = WebBrowserTool(manager=_FakeManager(browser=_FakeBrowser()))
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "fetch", "url": "file:///etc/passwd"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("refused", out.content)

    def test_a5_loopback_url_rejected(self) -> None:
        tool = WebBrowserTool(manager=_FakeManager(browser=_FakeBrowser()))
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "fetch", "url": "http://127.0.0.1/admin"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("refused", out.content)

    def test_a6_click_requires_selector(self) -> None:
        tool = WebBrowserTool(manager=_FakeManager(browser=_FakeBrowser()))
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "click", "url": "https://example.com"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("'selector'", out.content)

    def test_a7_type_requires_text(self) -> None:
        tool = WebBrowserTool(manager=_FakeManager(browser=_FakeBrowser()))
        out = tool.execute(
            ToolCall(
                id="c1",
                name="web_browser",
                arguments={"operation": "type", "url": "https://example.com", "selector": "#q"},
            ),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("'text'", out.content)


class HappyPathTests(unittest.TestCase):
    def test_b1_fetch_returns_markdown(self) -> None:
        page = _FakePage(html="<h1>Hello</h1><p>World</p>", title="Greeting")
        manager = _FakeManager(browser=_FakeBrowser(page=page))
        tool = WebBrowserTool(manager=manager)
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "fetch", "url": "https://example.com"}),
            _ctx(),
        )
        self.assertFalse(out.is_error, out.content)
        self.assertIn("https://example.com", out.content)
        self.assertIn("Hello", out.content)
        self.assertEqual(page.goto_calls, [("https://example.com", "domcontentloaded")])
        self.assertTrue(page.closed)
        self.assertEqual(out.metadata.get("operation"), "fetch")

    def test_b2_click_records_selector(self) -> None:
        page = _FakePage(html="<button>Click</button>", title="Clicked")
        manager = _FakeManager(browser=_FakeBrowser(page=page))
        tool = WebBrowserTool(manager=manager)
        out = tool.execute(
            ToolCall(
                id="c1",
                name="web_browser",
                arguments={"operation": "click", "url": "https://example.com", "selector": "button"},
            ),
            _ctx(),
        )
        self.assertFalse(out.is_error, out.content)
        self.assertEqual(page.click_calls, ["button"])
        self.assertEqual(out.metadata.get("selector"), "button")

    def test_b3_type_fills_field(self) -> None:
        page = _FakePage()
        manager = _FakeManager(browser=_FakeBrowser(page=page))
        tool = WebBrowserTool(manager=manager)
        out = tool.execute(
            ToolCall(
                id="c1",
                name="web_browser",
                arguments={
                    "operation": "type",
                    "url": "https://example.com",
                    "selector": "#search",
                    "text": "hello",
                },
            ),
            _ctx(),
        )
        self.assertFalse(out.is_error, out.content)
        self.assertEqual(page.fill_calls, [("#search", "hello")])
        self.assertEqual(out.metadata.get("chars"), 5)

    def test_b4_screenshot_saves_to_spill_dir(self) -> None:
        page = _FakePage()
        manager = _FakeManager(browser=_FakeBrowser(page=page))
        tool = WebBrowserTool(manager=manager)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = EngineConfig(
                web_browser_enabled=True,
                tool_result_spill_dir=Path(tmp),
            )
            ctx = TurnContext(
                session_id="ses-shot",
                iteration=0,
                metadata={"_engine_config": cfg},
            )
            out = tool.execute(
                ToolCall(
                    id="cshot",
                    name="web_browser",
                    arguments={"operation": "screenshot", "url": "https://example.com"},
                ),
                ctx,
            )
            self.assertFalse(out.is_error, out.content)
            shot_path = Path(out.metadata["screenshot_path"])
            self.assertTrue(shot_path.exists())
            self.assertGreater(shot_path.stat().st_size, 0)
            self.assertIn("Browser screenshot", out.content)

    def test_b5_extra_wait_seconds_propagates(self) -> None:
        page = _FakePage()
        manager = _FakeManager(browser=_FakeBrowser(page=page))
        tool = WebBrowserTool(manager=manager)
        tool.execute(
            ToolCall(
                id="c1",
                name="web_browser",
                arguments={
                    "operation": "fetch",
                    "url": "https://example.com",
                    "wait_seconds": 0.5,
                },
            ),
            _ctx(),
        )
        self.assertEqual(page._extra_wait_ms, 500)

    def test_b6_navigation_timeout_passed_through(self) -> None:
        page = _FakePage()
        manager = _FakeManager(browser=_FakeBrowser(page=page))
        tool = WebBrowserTool(manager=manager)
        cfg = EngineConfig(web_browser_enabled=True, web_browser_navigation_timeout_seconds=3)
        tool.execute(
            ToolCall(
                id="c1",
                name="web_browser",
                arguments={"operation": "fetch", "url": "https://example.com"},
            ),
            _ctx(config=cfg),
        )
        self.assertEqual(page.timeout_ms, 3000)


class FailureTests(unittest.TestCase):
    def test_c1_browser_unavailable_returns_clean_error(self) -> None:
        manager = _FakeManager(raises=BrowserUnavailable("install Playwright"))
        tool = WebBrowserTool(manager=manager)
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "fetch", "url": "https://example.com"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("install Playwright", out.content)

    def test_c2_navigation_failure_is_wrapped(self) -> None:
        class _BadPage(_FakePage):
            def goto(self, url: str, *, wait_until: str = "domcontentloaded") -> None:
                raise RuntimeError("net::ERR_NAME_NOT_RESOLVED")

        page = _BadPage()
        manager = _FakeManager(browser=_FakeBrowser(page=page))
        tool = WebBrowserTool(manager=manager)
        # Use example.com so SSRF DNS lookup succeeds (RFC 2606 reserved
        # but globally resolvable); the *page-level* goto then fails.
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "fetch", "url": "https://example.com"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("browser error", out.content)

    def test_c3_browser_get_failure_is_wrapped(self) -> None:
        manager = _FakeManager(raises=RuntimeError("boom"))
        tool = WebBrowserTool(manager=manager)
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "fetch", "url": "https://example.com"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("browser error", out.content)

    def test_c4_new_page_failure_is_wrapped(self) -> None:
        manager = _FakeManager(browser=_FakeBrowser(raise_on_new_page=True))
        tool = WebBrowserTool(manager=manager)
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "fetch", "url": "https://example.com"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("browser error", out.content)

    def test_c5_metadata_contains_url_on_error(self) -> None:
        manager = _FakeManager(raises=BrowserUnavailable("nope"))
        tool = WebBrowserTool(manager=manager)
        out = tool.execute(
            ToolCall(id="c1", name="web_browser", arguments={"operation": "fetch", "url": "https://example.com/x"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertEqual(out.metadata.get("url"), "https://example.com/x")


class SpillTests(unittest.TestCase):
    def test_d1_large_fetch_spills_to_disk(self) -> None:
        # Fabricate ~120KB HTML so we trip the 80KB MAX_RESULT_CHARS gate.
        big_html = "<p>" + ("X" * 120_000) + "</p>"
        page = _FakePage(html=big_html, title="Big")
        manager = _FakeManager(browser=_FakeBrowser(page=page))
        tool = WebBrowserTool(manager=manager)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = EngineConfig(
                web_browser_enabled=True,
                tool_result_spill_enabled=True,
                tool_result_spill_dir=Path(tmp),
            )
            ctx = TurnContext(
                session_id="ses-spill",
                iteration=0,
                metadata={"_engine_config": cfg},
            )
            out = tool.execute(
                ToolCall(
                    id="cspill",
                    name="web_browser",
                    arguments={"operation": "fetch", "url": "https://example.com"},
                ),
                ctx,
            )
            self.assertFalse(out.is_error, out.content)
            self.assertIn("output truncated", out.content)


if __name__ == "__main__":
    unittest.main()
