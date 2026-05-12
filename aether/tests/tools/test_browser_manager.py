"""Tests for ``aether.runtime.resources.browser_manager.BrowserManager``.

Sprint 3.5 / PR-3 (PR 3.5.10).

Playwright is an optional / heavy dep; CI doesn't have it installed.
The manager is built around that — when import fails we should raise
:class:`BrowserUnavailable`, *not* crash.  The tests verify both the
graceful-degradation path and the lifecycle bookkeeping (lazy launch
+ shutdown idempotence).
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from aether.runtime.resources import browser_manager as bm
from aether.runtime.resources.browser_manager import BrowserManager, BrowserUnavailable


class _FakeBrowser:
    def __init__(self) -> None:
        self.closed = False

    def is_connected(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self) -> None:
        self.launched = 0

    def launch(self, *, headless: bool = True) -> _FakeBrowser:
        self.launched += 1
        return _FakeBrowser()


class _FakePlaywright:
    def __init__(self) -> None:
        self.chromium = _FakeChromium()
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakeSyncPlaywright:
    def __init__(self) -> None:
        self.instance = _FakePlaywright()

    def start(self) -> _FakePlaywright:
        return self.instance


def _patch_playwright(success: bool = True, *, launch_raises: Exception | None = None) -> Any:
    fake_module = mock.MagicMock()
    fake_sync = _FakeSyncPlaywright()
    if launch_raises is not None:
        fake_sync.instance.chromium.launch = mock.MagicMock(side_effect=launch_raises)
    fake_module.sync_api.sync_playwright = lambda: fake_sync
    if success:
        return mock.patch.dict(
            "sys.modules",
            {"playwright": fake_module, "playwright.sync_api": fake_module.sync_api},
        )
    # When Playwright is missing, importing should raise ImportError.
    return mock.patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None})


class GracefulDegradationTests(unittest.TestCase):
    def test_a1_missing_playwright_raises_browser_unavailable(self) -> None:
        with _patch_playwright(success=False):
            mgr = BrowserManager()
            with self.assertRaises(BrowserUnavailable):
                mgr.get_browser()

    def test_a2_launch_failure_raises_browser_unavailable(self) -> None:
        with _patch_playwright(launch_raises=RuntimeError("chromium missing")):
            mgr = BrowserManager()
            with self.assertRaises(BrowserUnavailable) as ctx:
                mgr.get_browser()
            self.assertIn("Failed to launch", str(ctx.exception))

    def test_a3_unavailable_does_not_leave_partial_state(self) -> None:
        with _patch_playwright(launch_raises=RuntimeError("nope")):
            mgr = BrowserManager()
            try:
                mgr.get_browser()
            except BrowserUnavailable:
                pass
            self.assertFalse(mgr.is_running())


class LazyLifecycleTests(unittest.TestCase):
    def test_b1_lazy_launch_only_runs_once(self) -> None:
        with _patch_playwright(success=True):
            mgr = BrowserManager(idle_timeout_seconds=60)
            b1 = mgr.get_browser()
            b2 = mgr.get_browser()
            self.assertIs(b1, b2)
            mgr.shutdown()

    def test_b2_shutdown_is_idempotent(self) -> None:
        with _patch_playwright(success=True):
            mgr = BrowserManager(idle_timeout_seconds=60)
            mgr.get_browser()
            mgr.shutdown()
            mgr.shutdown()  # must not raise
            self.assertFalse(mgr.is_running())

    def test_b3_get_browser_after_shutdown_relaunches(self) -> None:
        with _patch_playwright(success=True):
            mgr = BrowserManager(idle_timeout_seconds=60)
            b1 = mgr.get_browser()
            mgr.shutdown()
            b2 = mgr.get_browser()
            self.assertIsNot(b1, b2)
            mgr.shutdown()

    def test_b4_disconnected_browser_is_replaced(self) -> None:
        with _patch_playwright(success=True):
            mgr = BrowserManager(idle_timeout_seconds=60)
            b1 = mgr.get_browser()
            b1.closed = True  # simulate Chromium dying mid-session
            b2 = mgr.get_browser()
            self.assertIsNot(b1, b2)
            mgr.shutdown()

    def test_b5_touch_does_not_launch(self) -> None:
        with _patch_playwright(success=True):
            mgr = BrowserManager(idle_timeout_seconds=60)
            mgr.touch()
            self.assertFalse(mgr.is_running())

    def test_b6_idle_watcher_shuts_down_after_timeout(self) -> None:
        with _patch_playwright(success=True):
            mgr = BrowserManager(idle_timeout_seconds=1)
            mgr.get_browser()
            self.assertTrue(mgr.is_running())
            # Wait long enough for the watcher (sleeps in 1s slices).
            import time

            for _ in range(40):
                time.sleep(0.1)
                if not mgr.is_running():
                    break
            self.assertFalse(mgr.is_running())


if __name__ == "__main__":
    unittest.main()
