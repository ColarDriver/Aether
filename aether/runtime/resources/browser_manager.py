"""Playwright browser lifecycle manager.

Reasons we *don't* let :class:`WebBrowserTool` own the browser
directly:

* Cold start of headless Chromium is 1–2s.  Sharing a single browser
  across many tool calls inside one session keeps that cost off the
  hot path.
* Playwright is a hard process-tree dependency: forgetting to call
  ``browser.close()`` leaves orphan Chromium processes around.  A
  single owner makes ``atexit`` cleanup straightforward.
* The browser must be shut down after some idle period or it pins
  ~150 MB of RAM for nothing — easier to enforce when one object
  watches the timer.

The manager is a *lazy singleton*: the actual ``sync_playwright()``
import only happens inside :meth:`get_browser`, so callers that never
trigger the tool pay zero import cost.

Failure handling: if Playwright is not installed or Chromium cannot
launch, ``get_browser`` raises a structured exception (see
:class:`BrowserUnavailable`) that the tool catches and surfaces as a
helpful "install via ..." message rather than crashing the turn.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


__all__ = ["BrowserManager", "BrowserUnavailable"]


class BrowserUnavailable(RuntimeError):
    """Raised when Playwright or its Chromium binary cannot be obtained."""


class BrowserManager:
    """Holds a single headless Chromium browser, restarts on demand."""

    def __init__(
        self,
        *,
        idle_timeout_seconds: int = 30,
        headless: bool = True,
    ) -> None:
        self.idle_timeout = max(1, int(idle_timeout_seconds))
        self.headless = bool(headless)
        self._playwright: Any = None
        self._browser: Any = None
        self._last_use = 0.0
        self._lock = threading.Lock()
        self._idle_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ----------------------------------------------------------- public API

    def is_running(self) -> bool:
        with self._lock:
            return self._browser is not None and self._is_browser_connected_unlocked()

    def get_browser(self) -> Any:
        """Return a connected Chromium browser, launching one if needed.

        Raises :class:`BrowserUnavailable` when Playwright is not
        installed or Chromium fails to launch.
        """
        with self._lock:
            if self._browser is not None and self._is_browser_connected_unlocked():
                self._last_use = time.monotonic()
                return self._browser
            self._launch_unlocked()
            self._last_use = time.monotonic()
            assert self._browser is not None
            return self._browser

    def shutdown(self) -> None:
        with self._lock:
            self._stop_event.set()
            self._shutdown_unlocked()

    def touch(self) -> None:
        """Bump the idle timer without taking a fresh browser handle."""
        with self._lock:
            self._last_use = time.monotonic()

    # ----------------------------------------------------------- internals

    def _is_browser_connected_unlocked(self) -> bool:
        if self._browser is None:
            return False
        is_connected = getattr(self._browser, "is_connected", None)
        if is_connected is None:
            return True
        try:
            return bool(is_connected())
        except Exception:
            return False

    def _launch_unlocked(self) -> None:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError as exc:
            raise BrowserUnavailable(
                "Playwright is not installed. Install via "
                "`pip install playwright && playwright install chromium`."
            ) from exc
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
        except Exception as exc:
            self._browser = None
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None
            raise BrowserUnavailable(
                f"Failed to launch Chromium via Playwright: {exc}. "
                "Run `playwright install chromium` once after installation."
            ) from exc
        self._stop_event.clear()
        self._start_idle_watcher_unlocked()

    def _start_idle_watcher_unlocked(self) -> None:
        if self._idle_thread is not None and self._idle_thread.is_alive():
            return

        def watch() -> None:
            while not self._stop_event.is_set():
                # Sleep in short slices so shutdown() unblocks promptly.
                if self._stop_event.wait(timeout=min(self.idle_timeout, 1.0)):
                    return
                with self._lock:
                    if self._browser is None:
                        return
                    if time.monotonic() - self._last_use >= self.idle_timeout:
                        logger.info("BrowserManager: idle %.1fs >= %ds; shutting down browser",
                                    time.monotonic() - self._last_use, self.idle_timeout)
                        self._shutdown_unlocked()
                        return

        self._idle_thread = threading.Thread(target=watch, daemon=True, name="browser-idle-watcher")
        self._idle_thread.start()

    def _shutdown_unlocked(self) -> None:
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        self._stop_event.set()
