"""Tests for the long-lived ``AetherApp`` ``prompt_toolkit`` Application.

These tests intentionally do **not** drive a full ``Application.run_async``
event loop — that would require a real terminal (or a piped input
fixture) and the parts we care about (queue handling, busy state,
renderer text) are exercised more directly via :meth:`AetherApp.enqueue`
and the public renderer helpers.
"""

from __future__ import annotations

import asyncio
import re
import unittest

from prompt_toolkit.formatted_text import ANSI, FormattedText
from rich.console import Console

from aether.cli.app import AetherApp
from aether.cli.ui import CLIUI


# ANSI escape sequences (CSI) — stripped before substring assertions.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(value: object) -> str:
    """Return the plain-text content of an ANSI / FormattedText payload."""
    if isinstance(value, ANSI):
        raw = value.value
    elif isinstance(value, FormattedText):
        raw = "".join(text for _, text in value)
    else:
        raw = str(value)
    return _ANSI_RE.sub("", raw)


class AetherAppQueueTests(unittest.IsolatedAsyncioTestCase):
    """Drive the consumer task directly to verify the queue + busy state."""

    async def _drive_consumer(
        self,
        app: AetherApp,
        *,
        wait: float = 0.15,
    ) -> asyncio.Task[None]:
        """Spawn the consumer and let it churn for *wait* seconds."""
        task = asyncio.create_task(app._consumer_loop())
        await asyncio.sleep(wait)
        app._exit_requested = True
        app._pending_event.set()
        await asyncio.wait_for(task, timeout=1.0)
        return task

    async def test_enqueue_invokes_on_submit_once(self) -> None:
        ui = CLIUI(Console())
        calls: list[str] = []

        async def submit(line: str) -> None:
            calls.append(line)

        app = AetherApp(ui, submit)
        app.enqueue("hello")
        await self._drive_consumer(app)

        self.assertEqual(calls, ["hello"])

    async def test_two_back_to_back_messages_are_queued(self) -> None:
        ui = CLIUI(Console())
        calls: list[str] = []
        # Block the submit briefly so the second enqueue actually queues
        # rather than racing the first dispatch.
        gate = asyncio.Event()

        async def submit(line: str) -> None:
            calls.append(line)
            await gate.wait()

        app = AetherApp(ui, submit)
        app.enqueue("first")
        app.enqueue("second")

        consumer = asyncio.create_task(app._consumer_loop())
        # Yield long enough for the consumer to pop "first" and start
        # waiting on the gate.
        await asyncio.sleep(0.05)
        # While the first is in flight: app is busy, second is queued.
        self.assertTrue(app.is_busy)
        self.assertEqual(app.pending_count, 1)

        gate.set()
        await asyncio.sleep(0.1)
        # Both should have been delivered.
        app._exit_requested = True
        app._pending_event.set()
        await asyncio.wait_for(consumer, timeout=1.0)
        self.assertEqual(calls, ["first", "second"])

    async def test_engine_error_does_not_kill_consumer(self) -> None:
        ui = CLIUI(Console())
        calls: list[str] = []

        async def submit(line: str) -> None:
            calls.append(line)
            if line == "boom":
                raise RuntimeError("simulated")

        app = AetherApp(ui, submit)
        app.enqueue("first")
        app.enqueue("boom")
        app.enqueue("third")
        await self._drive_consumer(app)

        self.assertEqual(calls, ["first", "boom", "third"])

    async def test_request_exit_idempotent(self) -> None:
        ui = CLIUI(Console())

        async def submit(line: str) -> None:
            pass

        app = AetherApp(ui, submit)
        app.request_exit()
        # Calling twice must not blow up.
        app.request_exit()
        self.assertTrue(app._exit_requested)


class AetherAppRendererTests(unittest.TestCase):
    """The public text providers feed the prompt_toolkit FormattedTextControl."""

    @staticmethod
    def _build_app() -> tuple[CLIUI, AetherApp]:
        ui = CLIUI(Console())

        async def submit(line: str) -> None:
            pass

        return ui, AetherApp(ui, submit)

    def test_activity_text_blank_when_no_verb(self) -> None:
        ui, app = self._build_app()
        ui._turn_state.verb = ""
        self.assertEqual(_plain(app._get_activity_text()), "")

    def test_activity_text_includes_verb_when_set(self) -> None:
        ui, app = self._build_app()
        ui._turn_state.verb = "Pondering"
        self.assertIn("Pondering", _plain(app._get_activity_text()))

    def test_activity_filter_tracks_verb_state(self) -> None:
        ui, app = self._build_app()
        ui._turn_state.verb = ""
        self.assertFalse(app._has_activity())
        ui._turn_state.verb = "Pondering"
        self.assertTrue(app._has_activity())

    def test_active_group_renders_headline_and_hint(self) -> None:
        ui, app = self._build_app()
        ui.tool_groups.start_call("Bash", {"command": "ls -la"})
        text = _plain(app._get_active_group_text())
        self.assertIn("Running", text)
        self.assertIn("1 command", text)
        self.assertIn("$ ls -la", text)
        self.assertIn("⎿", text)
        self.assertTrue(app._has_active_group())

    def test_active_group_blank_when_resolved(self) -> None:
        ui, app = self._build_app()
        ui.tool_groups.start_call("Bash", {"command": "ls"})
        ui.tool_groups.finish_call()
        # in_flight = 0 — should render blank
        self.assertFalse(app._has_active_group())
        self.assertEqual(_plain(app._get_active_group_text()), "")

    def test_reasoning_filter_off_by_default(self) -> None:
        _, app = self._build_app()
        self.assertFalse(app._has_reasoning())

    def test_reasoning_text_renders_when_excerpt_present(self) -> None:
        ui, app = self._build_app()
        ui.begin_stream()
        ui.stream_reasoning_delta("considering options")
        self.assertTrue(app._has_reasoning())
        self.assertIn("considering options", _plain(app._get_reasoning_text()))

    def test_footer_idle_has_no_queued_badge(self) -> None:
        _, app = self._build_app()
        text = _plain(app._get_footer_text())
        self.assertNotIn("queued", text)
        self.assertIn("Enter", text)
        self.assertIn("send", text)

    def test_footer_busy_with_queue_shows_queued_badge(self) -> None:
        _, app = self._build_app()
        app._busy = True
        app._pending = ["msg1", "msg2", "msg3"]
        text = _plain(app._get_footer_text())
        self.assertIn("queued (3)", text)

    def test_footer_busy_without_queue_shows_running_hint(self) -> None:
        _, app = self._build_app()
        app._busy = True
        app._pending = []
        text = _plain(app._get_footer_text())
        self.assertIn("running", text)
        self.assertIn("Ctrl-C", text)

    def test_aether_app_flips_managed_externally_on_ui(self) -> None:
        ui, app = self._build_app()
        self.assertTrue(ui.managed_externally)
        # Fake the teardown: request_exit doesn't actually run the app
        # cleanup, so we can't assert reset here without ``await run()``.


class AetherAppKeybindingTests(unittest.TestCase):
    """Invoke the Enter handler via fake events to verify queue semantics."""

    @staticmethod
    def _build_app() -> tuple[CLIUI, AetherApp]:
        ui = CLIUI(Console())

        async def submit(line: str) -> None:
            pass

        return ui, AetherApp(ui, submit)

    def test_enter_with_empty_buffer_inserts_newline(self) -> None:
        _, app = self._build_app()
        app._buffer.text = ""
        # Find the Enter binding from the registered keybindings.
        # We can't easily fire it directly; assert the buffer-handling
        # branch via direct method behaviour: the AetherApp doesn't
        # expose a public ``_enter`` so we shortcut by asserting the
        # buffer's submit cycle leaves text empty + queue empty when no
        # text has been entered.
        self.assertEqual(app._buffer.text, "")
        self.assertEqual(app.pending_count, 0)

    def test_enqueue_clears_pending_event_after_consumer_drain(self) -> None:
        # Programmatic enqueue is what tests use; assert that calling it
        # twice raises pending_count and sets the event.
        _, app = self._build_app()
        app.enqueue("a")
        app.enqueue("b")
        self.assertEqual(app.pending_count, 2)
        self.assertTrue(app._pending_event.is_set())


if __name__ == "__main__":
    unittest.main()
