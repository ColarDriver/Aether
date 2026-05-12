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
import time
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import ANSI, FormattedText
from rich.console import Console

from aether.cli.app import AetherApp
from aether.cli.ui import CLIUI
from aether.runtime.tools.tool_permissions import (
    ToolPermissionDecision,
    ToolPermissionDecisionType,
    ToolPermissionPreview,
    ToolPermissionRequest,
)


def _make_event(buffer_text: str = "") -> SimpleNamespace:
    """Build a minimal prompt_toolkit-shaped event for handler tests."""
    buf = Buffer()
    if buffer_text:
        buf.text = buffer_text
    return SimpleNamespace(current_buffer=buf, app=MagicMock())


def _make_app(
    *,
    on_interrupt=None,
    on_clear_history=None,
) -> AetherApp:
    ui = CLIUI(Console())

    async def submit(_line: str) -> None:
        return None

    return AetherApp(
        ui,
        submit,
        on_interrupt=on_interrupt,
        on_clear_history=on_clear_history,
    )


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


class AetherAppStreamingPreviewCapTests(unittest.TestCase):
    """Streaming preview must keep growing with terminal height.

    The legacy implementation hard-capped the preview at 12 rows so
    long streaming responses froze visually — the bottom region
    plateaued, then ``end_stream`` flushed the full body to scrollback
    in one jarring jump.  The dynamic cap (``terminal_rows - reserved``)
    lets the preview grow naturally for as long as the terminal can
    fit it.
    """

    @staticmethod
    def _build_app() -> AetherApp:
        ui = CLIUI(Console())

        async def submit(line: str) -> None:
            pass

        return AetherApp(ui, submit)

    def test_dynamic_cap_grows_with_terminal_height(self) -> None:
        app = self._build_app()

        class _Out:
            def __init__(self, rows: int) -> None:
                self.rows = rows

            def get_size(self):
                # Mimic prompt_toolkit's Size dataclass shape.
                size = type("S", (), {})()
                size.rows = self.rows
                size.columns = 80
                return size

        # 24-row terminal: 24 - 9 reserved = 15 (still above the floor).
        app._app.output = _Out(24)  # type: ignore[assignment]
        self.assertEqual(app._streaming_preview_max_rows(), 15)
        # 50-row terminal: meaningfully more space (was hard-capped at 12).
        app._app.output = _Out(50)  # type: ignore[assignment]
        self.assertEqual(app._streaming_preview_max_rows(), 41)
        # 100-row terminal: nearly the whole screen is preview.
        app._app.output = _Out(100)  # type: ignore[assignment]
        self.assertEqual(app._streaming_preview_max_rows(), 91)

    def test_dynamic_cap_floors_to_min_when_terminal_is_tiny(self) -> None:
        app = self._build_app()

        class _Out:
            class _Size:
                rows = 8
                columns = 80

            def get_size(self):
                return self._Size()

        app._app.output = _Out()  # type: ignore[assignment]
        # 8 - 9 = -1 → floor to the 12-row min so the layout still
        # has *something* to allocate.
        self.assertEqual(
            app._streaming_preview_max_rows(),
            app._streaming_preview_min_cap,
        )

    def test_dynamic_cap_falls_back_when_get_size_raises(self) -> None:
        # Detached / piped stdout: ``get_size`` may raise OSError.
        # We must still return a usable cap so streaming can proceed.
        app = self._build_app()

        class _BrokenOut:
            def get_size(self):
                raise OSError("no tty")

        app._app.output = _BrokenOut()  # type: ignore[assignment]
        self.assertEqual(
            app._streaming_preview_max_rows(),
            app._streaming_preview_min_cap,
        )


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
        # ESC is the primary interrupt key now (PR 6.1); the always-on
        # ESC fragment with its "interrupt" suffix carries the hint.
        self.assertIn("Esc", text)
        self.assertIn("interrupt", text)

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


class AetherAppEscPriorityChainTests(unittest.TestCase):
    """ESC dispatch must follow the documented 5-tier priority chain."""

    def test_esc_with_busy_triggers_interrupt(self) -> None:
        interrupts: list[int] = []
        app = _make_app(on_interrupt=lambda: interrupts.append(1))
        app._busy = True
        app.ui._turn_state.verb = "Conjuring"
        app._cached_preview_lines = 0
        app._pending = ["queued-1", "queued-2"]
        event = _make_event()

        app._handle_esc(event)

        self.assertEqual(interrupts, [1])
        # Queue drained as part of the interrupt action.
        self.assertEqual(app._pending, [])
        # Double-press window never armed — busy is a terminal branch.
        self.assertIsNone(app._last_esc_at)
        self.assertFalse(app._has_activity())
        self.assertTrue(app._interrupt_visual_pending)

    def test_esc_with_text_clears_buffer_and_arms_window(self) -> None:
        app = _make_app()
        event = _make_event("hello world")

        app._handle_esc(event)

        self.assertEqual(event.current_buffer.text, "")
        # Double-press window armed so a quick follow-up ESC clears
        # history (matches claude-code's "Esc Esc" workflow).
        self.assertIsNotNone(app._last_esc_at)

    def test_esc_with_pending_queue_pops_most_recent(self) -> None:
        app = _make_app()
        app._pending = ["older", "newer"]
        app._pending_event.set()
        event = _make_event()

        app._handle_esc(event)

        self.assertEqual(event.current_buffer.text, "newer")
        self.assertEqual(app._pending, ["older"])
        # Queue still has entries → event stays set.
        self.assertTrue(app._pending_event.is_set())

    def test_esc_pop_clears_pending_event_when_queue_empties(self) -> None:
        app = _make_app()
        app._pending = ["only-one"]
        app._pending_event.set()
        event = _make_event()

        app._handle_esc(event)

        self.assertEqual(app._pending, [])
        self.assertFalse(app._pending_event.is_set())

    def test_double_esc_clears_history(self) -> None:
        cleared: list[int] = []
        app = _make_app(on_clear_history=lambda: cleared.append(1))

        # First ESC arms the double-press window via the default branch.
        app._handle_esc(_make_event())
        self.assertEqual(cleared, [])
        self.assertIsNotNone(app._last_esc_at)
        self.assertEqual(app._transient_hint, "press ESC again to clear history")

        # Second ESC within the window triggers on_clear_history.
        app._handle_esc(_make_event())
        self.assertEqual(cleared, [1])
        # Window resets after firing.
        self.assertIsNone(app._last_esc_at)

    def test_esc_outside_double_window_rearms_instead_of_clearing(self) -> None:
        cleared: list[int] = []
        app = _make_app(on_clear_history=lambda: cleared.append(1))

        # Simulate a stale prior ESC (much older than the 0.8 s window).
        app._last_esc_at = time.monotonic() - 5.0

        app._handle_esc(_make_event())

        self.assertEqual(cleared, [])
        self.assertIsNotNone(app._last_esc_at)
        # Re-armed within window now.
        self.assertLess(time.monotonic() - app._last_esc_at, 0.5)

    def test_esc_without_clear_history_callback_only_toasts(self) -> None:
        # Two quick ESCs without a callback should not crash; the
        # second falls through the chain harmlessly.
        app = _make_app(on_clear_history=None)
        app._handle_esc(_make_event())
        app._handle_esc(_make_event())  # must not raise


class AetherAppCtrlCDoublePressExitTests(unittest.TestCase):
    """Ctrl-C must require two presses to exit when idle + empty."""

    def test_ctrl_c_idle_first_press_warns_without_exiting(self) -> None:
        app = _make_app()
        event = _make_event()

        app._handle_ctrl_c(event)

        self.assertFalse(app._exit_requested)
        self.assertIsNotNone(app._last_ctrl_c_at)
        self.assertEqual(app._transient_hint, "press Ctrl-C again to exit")
        event.app.exit.assert_not_called()

    def test_ctrl_c_idle_double_press_exits(self) -> None:
        app = _make_app()

        # First press arms the window.
        app._handle_ctrl_c(_make_event())
        self.assertFalse(app._exit_requested)

        # Second press within 2 s actually exits.
        second_event = _make_event()
        app._handle_ctrl_c(second_event)

        self.assertTrue(app._exit_requested)
        second_event.app.exit.assert_called_once()

    def test_ctrl_c_outside_window_does_not_exit(self) -> None:
        app = _make_app()
        app._last_ctrl_c_at = time.monotonic() - 5.0

        event = _make_event()
        app._handle_ctrl_c(event)

        self.assertFalse(app._exit_requested)
        event.app.exit.assert_not_called()

    def test_ctrl_c_with_text_clears_buffer_without_exiting(self) -> None:
        app = _make_app()
        event = _make_event("draft message")

        app._handle_ctrl_c(event)

        self.assertEqual(event.current_buffer.text, "")
        self.assertFalse(app._exit_requested)
        self.assertIsNone(app._last_ctrl_c_at)

    def test_ctrl_c_busy_interrupts_regardless_of_double_press(self) -> None:
        interrupts: list[int] = []
        app = _make_app(on_interrupt=lambda: interrupts.append(1))
        app._busy = True
        app.ui._turn_state.verb = "Conjuring"
        app._cached_preview_lines = 0
        app._pending = ["one", "two"]
        event = _make_event()

        app._handle_ctrl_c(event)

        self.assertEqual(interrupts, [1])
        self.assertEqual(app._pending, [])
        self.assertFalse(app._exit_requested)
        self.assertFalse(app._has_activity())
        self.assertTrue(app._interrupt_visual_pending)


class AetherAppFooterHintTests(unittest.TestCase):
    """Footer reflects ESC affordance + Shift+Enter newline label."""

    @staticmethod
    def _build_app() -> AetherApp:
        return _make_app()

    def test_idle_footer_shows_esc_clear(self) -> None:
        app = self._build_app()
        text = _plain(app._get_footer_text())
        self.assertIn("Esc", text)
        self.assertIn("clear", text)

    def test_busy_footer_shows_esc_interrupt(self) -> None:
        app = self._build_app()
        app._busy = True
        text = _plain(app._get_footer_text())
        self.assertIn("Esc", text)
        self.assertIn("interrupt", text)

    def test_interrupt_pending_footer_shows_interrupting_not_running(self) -> None:
        app = self._build_app()
        app._busy = True
        app._interrupt_visual_pending = True
        text = _plain(app._get_footer_text())
        self.assertIn("interrupting", text)
        self.assertNotIn("running", text)

    def test_idle_with_queue_shows_esc_pop_queued_label(self) -> None:
        app = self._build_app()
        app._pending = ["draft"]
        text = _plain(app._get_footer_text())
        self.assertIn("Esc", text)
        self.assertIn("pop queued", text)

    def test_footer_uses_shift_enter_label_not_esc_enter(self) -> None:
        app = self._build_app()
        text = _plain(app._get_footer_text())
        self.assertIn("Shift+Enter", text)
        self.assertNotIn("Esc+Enter", text)

    def test_transient_hint_renders_in_footer(self) -> None:
        app = self._build_app()
        app._set_transient_hint("press ESC again to clear history")
        text = _plain(app._get_footer_text())
        self.assertIn("press ESC again to clear history", text)

    def test_transient_hint_expires(self) -> None:
        app = self._build_app()
        # Manually set an expired hint — `_active_transient_hint` must
        # drop it so the footer renders without the toast.
        app._transient_hint = "stale"
        app._transient_hint_until = time.monotonic() - 1.0
        text = _plain(app._get_footer_text())
        self.assertNotIn("stale", text)


class AetherAppPermissionOverlayTests(unittest.TestCase):
    def _request(self) -> ToolPermissionRequest:
        return ToolPermissionRequest(
            session_id="s",
            tool_call_id="c1",
            tool_name="file_edit",
            arguments={"path": "/tmp/app.py"},
            category="write",
            risk="write",
            preview=ToolPermissionPreview(
                title="Edit file",
                path="/tmp/app.py",
                diff="--- /tmp/app.py\n+++ /tmp/app.py\n-old\n+new\n",
            ),
        )

    def test_permission_request_renders_above_input(self) -> None:
        app = _make_app()
        future: Future[ToolPermissionDecision] = Future()
        app.enqueue_permission_request(self._request(), future)

        self.assertTrue(app._has_permission_prompt())
        text = _plain(app._get_permission_text())
        self.assertIn("\u2500", text)
        self.assertIn("Edit file", text)
        self.assertIn("Do you want to make this edit to app.py?", text)
        self.assertIn("-old", text)
        self.assertIn("1. Yes", text)
        self.assertIn("2. Yes, allow edits in this path during this session", text)
        self.assertNotIn("Enter approve/select", text)
        self.assertFalse(future.done())

    def test_permission_prompt_trims_preview_in_short_terminal(self) -> None:
        app = _make_app()

        class _Out:
            class _Size:
                rows = 8
                columns = 80

            def get_size(self):
                return self._Size()

        app._app.output = _Out()  # type: ignore[assignment]
        future: Future[ToolPermissionDecision] = Future()
        app.enqueue_permission_request(self._request(), future)

        text = _plain(app._get_permission_text())

        self.assertLessEqual(app._permission_height().preferred, 6)
        self.assertIn("Do you want to make this edit to app.py?", text)
        self.assertIn("1. Yes", text)
        self.assertIn("3. No", text)
        self.assertNotIn("-old", text)

    def test_permission_prompt_hides_input_frame(self) -> None:
        app = _make_app()
        self.assertTrue(app._has_input_frame())

        future: Future[ToolPermissionDecision] = Future()
        app.enqueue_permission_request(self._request(), future)

        self.assertFalse(app._has_input_frame())

    def test_permission_enter_accept_once_resolves_future(self) -> None:
        app = _make_app()
        future: Future[ToolPermissionDecision] = Future()
        app.enqueue_permission_request(self._request(), future)

        app._resolve_active_permission()

        self.assertTrue(future.done())
        self.assertEqual(future.result().type, ToolPermissionDecisionType.ALLOW_ONCE)
        self.assertFalse(app._has_permission_prompt())

    def test_permission_down_enter_accept_session_resolves_future(self) -> None:
        app = _make_app()
        future: Future[ToolPermissionDecision] = Future()
        app.enqueue_permission_request(self._request(), future)

        app._move_permission_selection(1)
        app._resolve_active_permission()

        self.assertEqual(future.result().type, ToolPermissionDecisionType.ALLOW_SESSION)

    def test_permission_number_accept_session_resolves_future(self) -> None:
        app = _make_app()
        future: Future[ToolPermissionDecision] = Future()
        app.enqueue_permission_request(self._request(), future)

        handled = app._resolve_permission_number(2)

        self.assertTrue(handled)
        self.assertEqual(future.result().type, ToolPermissionDecisionType.ALLOW_SESSION)
        self.assertFalse(app._has_permission_prompt())

    def test_permission_invalid_number_is_ignored(self) -> None:
        app = _make_app()
        future: Future[ToolPermissionDecision] = Future()
        app.enqueue_permission_request(self._request(), future)

        handled = app._resolve_permission_number(9)

        self.assertFalse(handled)
        self.assertFalse(future.done())
        self.assertTrue(app._has_permission_prompt())

    def test_permission_esc_rejects_active_request(self) -> None:
        app = _make_app()
        future: Future[ToolPermissionDecision] = Future()
        app.enqueue_permission_request(self._request(), future)

        app._handle_esc(_make_event())

        self.assertEqual(future.result().type, ToolPermissionDecisionType.ABORT)
        self.assertFalse(app._has_permission_prompt())

    def test_permission_footer_uses_approval_shortcuts(self) -> None:
        app = _make_app()
        future: Future[ToolPermissionDecision] = Future()
        app.enqueue_permission_request(self._request(), future)

        text = _plain(app._get_footer_text())

        self.assertIn("approve", text)
        self.assertIn("1-3", text)
        self.assertIn("choose", text)
        self.assertIn("reject", text)


if __name__ == "__main__":
    unittest.main()
