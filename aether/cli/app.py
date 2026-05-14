"""Long-lived ``prompt_toolkit`` Application for the Aether REPL.

Replaces the per-turn :func:`aether.cli.input_box.prompt_box` lifecycle
with a single :class:`prompt_toolkit.application.Application` that
stays running for the entire REPL session.  The input frame remains
visible (and type-ahead works) while the engine streams its response —
additional Enter presses queue messages instead of dispatching them
immediately, mirroring claude-code's ``PromptInputQueuedCommands``.

Layout (non-fullscreen — the terminal scrollback above is regular Rich
output captured by ``patch_stdout(raw=True)``):

::

    ┌── scrollback (Rich console.print lands here) ───────────────┐
    │ › 列一下当前目录                                             │
    │                                                              │
    │ ● Searched for 1 pattern, read 1 file                        │
    │ ● 当前目录里有这些文件…                                      │
    └──────────────────────────────────────────────────────────────┘
      ● Searching for 1 pattern, reading 1 file…   ← active group
        ⎿ "foo" in src/                            ← active hint
      ⠋ Pondering (12s · ↓ 241 tokens · thinking)  ← activity bar
    ┌── input frame ───────────────────────────────────────────────┐
    │ › <buffer — editable during streaming>                       │
    └──────────────────────────────────────────────────────────────┘
      Enter send · Esc+Enter newline · /help commands · Ctrl-D exit
                                          · ▸ queued (n)            ← only when type-ahead pending

Threading model:

* ``Application.run_async()`` runs in the asyncio event loop.
* The supplied ``on_submit(text)`` coroutine is scheduled by the
  consumer task whenever the user presses Enter (or when a queued
  message becomes deliverable).
* The engine itself stays synchronous — callers wrap their blocking
  ``run_loop`` invocation with ``asyncio.to_thread`` so the event loop
  remains responsive.
* All Rich output produced inside ``on_submit`` flows through
  ``patch_stdout(raw=True)`` and lands above the prompt with ANSI
  styles preserved.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Future
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Awaitable, Callable, Optional

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.history import History
from prompt_toolkit.input.vt100_parser import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    HSplit,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame
from rich.console import Console as RichConsole
from rich.console import ConsoleRenderable

from aether.cli.activity import ActivityBar
from aether.cli.theme import (
    AETHER_ACCENT,
    AETHER_BORDER,
    AETHER_DIM,
    AETHER_PRIMARY,
    icon,
)
from aether.cli.ui import (
    CLIUI,
    _has_unclosed_command_fence,
    _looks_like_intended_tool_use,
    strip_all_command_fences,
    strip_tool_blocks,
)
from aether.runtime.tools.tool_permissions import (
    ToolPermissionDecision,
    ToolPermissionDecisionType,
    ToolPermissionRequest,
)


SubmitCallback = Callable[[str], Awaitable[None]]
InterruptCallback = Callable[[], None]
ClearHistoryCallback = Callable[[], None]


@dataclass(slots=True)
class PendingPermissionPrompt:
    request: ToolPermissionRequest
    future: Future[ToolPermissionDecision]
    selected_index: int = 0


# ---------------------------------------------------------------------------
# Shift+Enter → newline plumbing
# ---------------------------------------------------------------------------
#
# prompt_toolkit's :class:`Keys` enum doesn't model Shift+Enter natively
# because many terminals collapse it down to plain CR / LF.
# Modern terminals report it through one of two extended keyboard
# protocols, both as escape sequences we can intercept at the parser:
#
#  * kitty / CSI-u   : ``\x1b[13;2u``
#  * xterm modifyOtherKeys 2 : ``\x1b[27;2;13~``
#
# We map both to :data:`Keys.ControlJ` so the regular ``c-j`` binding
# (which inserts a newline) catches the press without us needing a
# dedicated Shift+Enter Keys enum entry.  Terminals that don't support
# either protocol fall back to plain Ctrl-J — users can still emit a
# newline by holding Ctrl-J directly.
for _seq in ("\x1b[13;2u", "\x1b[27;2;13~"):
    ANSI_SEQUENCES.setdefault(_seq, Keys.ControlJ)
del _seq


# Double-press windows for the ESC priority chain.
# - ESC: 0.8s matches claude-code's `useDoublePress` default for the
#   "press ESC again to clear" workflow.
# - Ctrl-C: 2.0s — wider window because Ctrl-C exits the REPL outright,
#   so we want to be extra forgiving against accidental double-taps.
_ESC_DOUBLE_PRESS_SEC: float = 0.8
_CTRL_C_DOUBLE_PRESS_SEC: float = 2.0
# How long a transient hint ("press ESC again to clear") stays visible
# in the footer.  Mirrors the longest double-press window so the hint
# is gone by the time the user could no longer act on it.
_HINT_DISPLAY_SEC: float = 2.0


# ---------------------------------------------------------------------------
# Style + footer fragments
# ---------------------------------------------------------------------------

_BOX_STYLE = Style.from_dict(
    {
        "frame.border": AETHER_BORDER,
        "input.icon": f"bold {AETHER_PRIMARY}",
        "input.text": "",
        "input.placeholder": f"italic {AETHER_DIM}",
        "footer": AETHER_DIM,
        "footer.kbd": f"bold {AETHER_ACCENT}",
        "footer.sep": AETHER_BORDER,
        "footer.queued": f"bold {AETHER_ACCENT}",
        "footer.busy": f"italic {AETHER_DIM}",
        "permission.border": AETHER_BORDER,
        "permission.title": f"bold {AETHER_PRIMARY}",
        "permission.subtitle": AETHER_DIM,
        "permission.question": "",
        "permission.option": "",
        "permission.option.selected": f"bold {AETHER_ACCENT}",
        "permission.diff": AETHER_DIM,
        "permission.command": f"bold {AETHER_ACCENT}",
        # Completion popup palette — copied verbatim from input_box.py so
        # the type-ahead experience stays identical to the per-turn
        # picker.
        "completion-menu":              f"bg:#1E293B {AETHER_DIM}",
        "completion-menu.completion":   f"bg:#1E293B {AETHER_DIM}",
        "completion-menu.completion.current": f"bg:{AETHER_PRIMARY} #FFFFFF bold",
        "completion-menu.meta":         f"bg:#1E293B {AETHER_DIM} italic",
        "completion-menu.meta.current": f"bg:{AETHER_PRIMARY} #FFFFFF italic",
        "completion-menu.multi-column-meta": f"bg:#1E293B {AETHER_DIM} italic",
    }
)


# ---------------------------------------------------------------------------
# Rich → ANSI bridge — used to render the Rich ``ActivityBar`` and
# ``ToolGroup`` renderables into ANSI strings that prompt_toolkit's
# FormattedTextControl can splat into the bottom region.
# ---------------------------------------------------------------------------

class _RichRenderer:
    """Render Rich renderables into ANSI strings for prompt_toolkit consumption.

    Reuses a single :class:`rich.console.Console` instance so we don't
    pay the per-call init cost; resets the underlying ``StringIO`` on
    each ``render`` call so the buffer never grows unbounded.
    """

    def __init__(self, *, width: int = 100) -> None:
        self._buf = StringIO()
        self._console = RichConsole(
            file=self._buf,
            force_terminal=True,
            color_system="truecolor",
            width=width,
            highlight=False,
            markup=False,
            soft_wrap=False,
        )

    def render(
        self,
        renderable: ConsoleRenderable,
        *,
        width: Optional[int] = None,
    ) -> str:
        if width is not None and width > 0:
            self._console.width = width
        self._buf.seek(0)
        self._buf.truncate()
        try:
            self._console.print(renderable, end="")
        except Exception:  # noqa: BLE001 — never let a render kill the app
            return ""
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# AetherApp
# ---------------------------------------------------------------------------

class AetherApp:
    """Long-lived prompt_toolkit Application — input frame stays visible.

    Public surface kept intentionally small: build, ``await run()``,
    and call :meth:`enqueue` from tests / programmatic callers if you
    want to skip the keypress path.

    Args:
        ui: shared :class:`CLIUI` (its ``managed_externally`` flag is
            flipped so ``begin_turn`` doesn't spawn a competing Rich
            ``Live`` region).
        on_submit: coroutine invoked once per user message.  Implementers
            wrap their blocking ``engine.run_loop`` call in
            ``asyncio.to_thread`` so the event loop stays responsive
            (and the activity bar keeps ticking).
        history: optional prompt_toolkit ``History`` for ↑/↓ recall.
        completer: optional ``Completer``; pass the slash-command
            completer here so ``/`` triggers the popup.
        on_interrupt: invoked when the user presses ESC or Ctrl-C while
            the engine is busy.  Should call ``engine.interrupt(...)``.
            The REPL is never killed by an interrupt key; the buffer is
            preserved so the user can re-submit.
        on_clear_history: invoked on a double-ESC from idle/empty state
            — analogous to claude-code's "Esc Esc → clear conversation".
            Implementers typically reset their in-memory message list
            while keeping the session id; ``None`` makes the chain fall
            through to a toast.
    """

    def __init__(
        self,
        ui: CLIUI,
        on_submit: SubmitCallback,
        *,
        history: History | None = None,
        completer: Completer | None = None,
        on_interrupt: Optional[InterruptCallback] = None,
        on_clear_history: Optional[ClearHistoryCallback] = None,
    ) -> None:
        self.ui = ui
        self._on_submit = on_submit
        self._on_interrupt = on_interrupt
        self._on_clear_history = on_clear_history
        self._history = history
        self._completer = completer

        self._renderer = _RichRenderer()

        # Floor on the streaming preview cap.  The actual cap is computed
        # dynamically from the terminal height (see
        # :meth:`_streaming_preview_max_rows`) so the in-flight body
        # grows as far as the terminal allows; this constant is just
        # the fallback for the rare path where ``output.get_size()``
        # can't tell us the real geometry (detached / piped stdout).
        self._streaming_preview_min_cap: int = 12

        # Rows reserved at the bottom of the terminal for the rest of
        # the layout — input frame border (2) + 1 row of input content
        # + footer (1) + activity bar (1) + reasoning excerpt (1) +
        # leading spacer (1) + a 2-row safety buffer for resize jitter
        # = 9 rows.  We subtract this from the terminal height to
        # decide how tall the streaming preview can grow before we
        # tail-crop.
        self._streaming_preview_reserved_rows: int = 9

        # Per-frame snapshot of the streaming preview — refreshed at the
        # start of every prompt_toolkit render cycle so the height
        # dimension and the FormattedTextControl always agree on what's
        # being drawn.  Without the cache the engine thread can mutate
        # ``_stream_buffer`` between the height query and the text
        # query, causing the activity bar to overlap the preview's tail
        # row.
        self._cached_preview_text: str = ""
        self._cached_preview_lines: int = 0

        # State shared between key bindings and the consumer task.
        self._busy = False
        # UI-only latch: once the user requests interrupt we hide the
        # bottom-region "live" affordances (activity bar, active tool
        # group, preview/reasoning excerpt) immediately instead of
        # waiting for the worker thread to unwind all the way out of
        # ``on_submit``.  This keeps the screen visually stopped as
        # soon as ESC lands, even if provider/tool cleanup takes a few
        # more seconds in the background.
        self._interrupt_visual_pending = False
        # We use a list rather than ``asyncio.Queue`` so the footer can
        # render the queue length without consuming it.  An ``Event``
        # gates the consumer's wait when the queue is empty.
        self._pending: list[str] = []
        self._pending_event = asyncio.Event()
        self._permission_queue: list[PendingPermissionPrompt] = []
        self._exit_requested = False
        self._consumer_task: asyncio.Task[None] | None = None
        self._refresh_task: asyncio.Task[None] | None = None

        # Double-press bookkeeping for the ESC priority chain and the
        # Ctrl-C "press again to exit" guard.  Both store a
        # ``time.monotonic()`` timestamp of the previous press (or
        # ``None`` if no press is currently in flight / the window has
        # expired).
        self._last_esc_at: float | None = None
        self._last_ctrl_c_at: float | None = None

        # Transient footer toast — set when the user presses an
        # interrupt key but no action matched (e.g. first ESC on idle +
        # empty buffer).  Cleared automatically when the display
        # deadline passes; the existing 5 Hz refresh loop drives the
        # cleanup repaint.
        self._transient_hint: str | None = None
        self._transient_hint_until: float = 0.0

        self._buffer = Buffer(
            history=history,
            multiline=True,
            completer=completer,
            complete_while_typing=True,
        )

        # Flip the UI into "managed" mode so its begin_turn / set_status
        # paths skip the Rich Live region — that would clash with our
        # bottom layout via ``patch_stdout``.
        self.ui.managed_externally = True

        self._app = self._build_app()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_app(self) -> Application[None]:
        bindings = self._build_keybindings()

        input_control = BufferControl(
            buffer=self._buffer,
            input_processors=[BeforeInput(f"{icon('user')} ", style="class:input.icon")],
            focusable=True,
        )

        input_window = Window(
            content=input_control,
            wrap_lines=True,
            # Start at one row, grow up to 12 — paste-friendly without
            # taking over the screen on very long pastes.
            height=D(min=1, max=12),
            dont_extend_height=True,
        )
        input_frame = Frame(input_window)
        input_frame_container = ConditionalContainer(
            content=input_frame,
            filter=Condition(self._has_input_frame),
        )

        # 1-row top spacer — keeps the bottom-region content visually
        # detached from the user echo / previous turn output that
        # ``patch_stdout`` lifts up into scrollback.  Without this the
        # activity bar glues directly to the previous line.  Only shown
        # while *something* below is active so idle turns don't waste a
        # row.
        spacer_window = Window(
            char=" ",
            height=1,
            dont_extend_height=True,
        )
        spacer_container = ConditionalContainer(
            content=spacer_window,
            filter=Condition(self._has_bottom_content),
        )

        # Streaming preview — the assistant body as it streams in.  Rendered
        # as plain Text inside a 2-column grid so the bullet lines up with
        # the final formatted markdown that ``end_stream`` flushes to
        # scrollback.  Capped at ~12 rows so very long responses don't push
        # the input frame off-screen mid-stream; the *full* body still
        # lands in scrollback at end_stream.
        #
        # The height is a *callable* returning the exact row count of the
        # cached preview snapshot so prompt_toolkit allocates exactly as
        # many rows as we need — a static ``D(min=1, max=12)`` defaulted
        # to ``preferred=1`` and the activity bar would overpaint our
        # second row.
        streaming_preview_window = Window(
            content=FormattedTextControl(
                text=self._get_streaming_preview_text,
                focusable=False,
                show_cursor=False,
            ),
            height=self._streaming_preview_height,
            wrap_lines=False,
        )
        streaming_preview_container = ConditionalContainer(
            content=streaming_preview_window,
            filter=Condition(self._has_streaming_preview),
        )

        permission_window = Window(
            content=FormattedTextControl(
                text=self._get_permission_text,
                focusable=False,
                show_cursor=False,
            ),
            height=self._permission_height,
            dont_extend_height=True,
            wrap_lines=False,
        )
        permission_container = ConditionalContainer(
            content=permission_window,
            filter=Condition(self._has_permission_prompt),
        )

        # Active tool group: 0-2 rows — headline + optional ⎿ hint.  The
        # ConditionalContainer hides the slot when no group is active so
        # it doesn't take up screen real-estate between turns.
        active_group_window = Window(
            content=FormattedTextControl(
                text=self._get_active_group_text,
                focusable=False,
                show_cursor=False,
            ),
            height=D(min=1, max=2),
            dont_extend_height=True,
            wrap_lines=False,
        )
        active_group_container = ConditionalContainer(
            content=active_group_window,
            filter=Condition(self._has_active_group),
        )

        # Activity bar: 1 row, only shown while a turn is running.
        activity_window = Window(
            content=FormattedTextControl(
                text=self._get_activity_text,
                focusable=False,
                show_cursor=False,
            ),
            height=1,
            dont_extend_height=True,
            wrap_lines=False,
        )
        activity_container = ConditionalContainer(
            content=activity_window,
            filter=Condition(self._has_activity),
        )

        # Reasoning excerpt: dim sub-line; same conditional.
        reasoning_window = Window(
            content=FormattedTextControl(
                text=self._get_reasoning_text,
                focusable=False,
                show_cursor=False,
            ),
            height=1,
            dont_extend_height=True,
            wrap_lines=False,
        )
        reasoning_container = ConditionalContainer(
            content=reasoning_window,
            filter=Condition(self._has_reasoning),
        )

        footer_window = Window(
            content=FormattedTextControl(
                text=self._get_footer_text,
                focusable=False,
                show_cursor=False,
            ),
            height=1,
            dont_extend_height=True,
            wrap_lines=False,
        )

        # Slash-command completion menu — placed *above* the input
        # frame as a regular HSplit child rather than floating below
        # the cursor.  Why: prompt_toolkit's ``Float(ycursor=True)``
        # tries to draw the popup below the cursor and only flips
        # upward when the *full* ``max_height`` rows fit above; in a
        # short IDE terminal pane (small Cursor split, narrow split-
        # screen layout, …) neither side has 12 free rows, so the
        # float falls back to "draw below" with whatever space is
        # left — which clipped the menu to 1-2 visible entries.
        # Inlining the menu in the HSplit gives prompt_toolkit's row
        # allocator dedicated space for the popup so it always shows
        # however many entries the terminal can fit, without
        # overlapping the input frame border.
        completions_menu = ConditionalContainer(
            content=CompletionsMenu(max_height=12, scroll_offset=1),
            filter=has_completions & Condition(self._has_input_frame),
        )

        root = HSplit(
            [
                spacer_container,
                streaming_preview_container,
                permission_container,
                active_group_container,
                activity_container,
                reasoning_container,
                completions_menu,
                input_frame_container,
                footer_window,
            ],
        )

        layout = Layout(root)

        app: Application[None] = Application(
            layout=layout,
            key_bindings=bindings,
            style=_BOX_STYLE,
            full_screen=False,
            mouse_support=False,
            erase_when_done=False,
            # Baseline 5 Hz refresh so the ``elapsed`` counter ticks
            # without us needing an explicit invalidate every 100 ms.
            # The consumer/key bindings still call ``invalidate`` on
            # state-changing events so the queued badge / busy state
            # update without animation lag.
            refresh_interval=0.2,
        )
        # Refresh the streaming preview snapshot at the start of every
        # render cycle so the height callback and the text callback
        # see the same buffer state — otherwise the engine thread can
        # append tokens between the two queries and the activity bar
        # will overpaint the preview's tail row.
        app.before_render += lambda *_: self._refresh_preview_cache()
        return app

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _build_keybindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("c-d")
        def _ctrl_d(event):  # noqa: ANN001
            if not event.current_buffer.text:
                self._exit_requested = True
                event.app.exit()
            else:
                event.current_buffer.delete()

        @bindings.add("c-c")
        def _ctrl_c(event):  # noqa: ANN001
            self._handle_ctrl_c(event)

        @bindings.add("up", filter=Condition(self._has_permission_prompt))
        def _permission_up(event):  # noqa: ANN001
            self._move_permission_selection(-1)
            event.app.invalidate()

        @bindings.add("down", filter=Condition(self._has_permission_prompt))
        def _permission_down(event):  # noqa: ANN001
            self._move_permission_selection(1)
            event.app.invalidate()

        for option_number in range(1, 10):
            @bindings.add(
                str(option_number),
                filter=Condition(self._has_permission_prompt),
                eager=True,
            )
            def _permission_number(event, option_number=option_number):  # noqa: ANN001
                self._resolve_permission_number(option_number)
                event.app.invalidate()

        # ESC single-press -> priority chain. ``eager=True``
        # bypasses prompt_toolkit's 25 ms ESC-sequence timeout so the
        # press feels instantaneous; this also displaces the
        # ``escape, enter`` newline binding (intentional — Shift+Enter
        # / Ctrl-J are the newline keys now).
        @bindings.add("escape", eager=True)
        def _esc(event):  # noqa: ANN001
            self._handle_esc(event)

        @bindings.add("c-j")
        def _ctrl_j(event):  # noqa: ANN001
            # Newline binding.  Catches three input sources:
            #   * Direct Ctrl-J press (byte 0x0a — same as ``\n``)
            #   * Shift+Enter on kitty / CSI-u terminals (the
            #     module-level ``ANSI_SEQUENCES`` extension translates
            #     ``\x1b[13;2u`` into ``Keys.ControlJ``)
            #   * Shift+Enter on xterm with ``modifyOtherKeys=2``
            #     (translates ``\x1b[27;2;13~`` the same way)
            event.current_buffer.insert_text("\n")

        @bindings.add("enter")
        def _enter(event):  # noqa: ANN001
            if self._has_permission_prompt():
                self._resolve_active_permission()
                event.app.invalidate()
                return

            buf = event.current_buffer
            # Completion popup: Enter with a highlighted candidate
            # accepts the completion, doesn't submit.
            if buf.complete_state and buf.complete_state.current_completion:
                buf.apply_completion(buf.complete_state.current_completion)
                return
            if buf.complete_state:
                buf.cancel_completion()

            text = buf.text
            if text.strip() == "":
                # Empty Enter inserts a newline (same as input_box.py).
                buf.insert_text("\n")
                return

            # Manually push to history — we're not using
            # ``session.prompt`` so prompt_toolkit's auto-history
            # path doesn't run.
            if self._history is not None:
                try:
                    self._history.append_string(text)
                except Exception:  # noqa: BLE001
                    pass

            buf.reset()
            self._pending.append(text)
            self._pending_event.set()
            event.app.invalidate()

        return bindings

    # ------------------------------------------------------------------
    # Interrupt-key handlers
    # ------------------------------------------------------------------

    def _handle_esc(self, event) -> None:  # noqa: ANN001
        """ESC priority chain — first match wins.

        Order mirrors claude-code's ``useCancelRequest`` cascade and
        ``PromptInput`` ESC dispatch.  Each branch performs its action
        then returns; the bottom branch is a no-op that arms the
        double-press window so a *second* quick ESC clears history.
        """
        if self._has_permission_prompt():
            self._reject_active_permission(source="user_abort")
            try:
                event.app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            return

        buf = event.current_buffer
        now = time.monotonic()

        # 1. Engine in flight → interrupt and drop the queue.
        if self._busy:
            self._interrupt_visual_pending = True
            if self._on_interrupt is not None:
                try:
                    self._on_interrupt()
                except Exception:  # noqa: BLE001
                    pass
            self._pending.clear()
            self._last_esc_at = None
            self._clear_transient_hint()
            try:
                event.app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            return

        # 2. Buffer has text → reset, arm the double-press window so a
        #    follow-up ESC clears history.
        if buf.text:
            buf.reset()
            self._last_esc_at = now
            self._clear_transient_hint()
            try:
                event.app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            return

        # 3. Pending queue → pop the most recent entry back for edit.
        if self._pending:
            last = self._pending.pop()
            buf.text = last
            buf.cursor_position = len(buf.text)
            # Re-clear the pending event if the queue just drained.
            if not self._pending:
                self._pending_event.clear()
            self._last_esc_at = None
            self._clear_transient_hint()
            try:
                event.app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            return

        # 4. Double-ESC within the window → clear conversation history.
        if self._is_double_press(self._last_esc_at, _ESC_DOUBLE_PRESS_SEC):
            self._last_esc_at = None
            self._clear_transient_hint()
            if self._on_clear_history is not None:
                try:
                    self._on_clear_history()
                except Exception:  # noqa: BLE001
                    pass
            try:
                event.app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            return

        # 5. Default — arm the double-press window and surface a hint.
        self._last_esc_at = now
        self._set_transient_hint("press ESC again to clear history")
        try:
            event.app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    def _handle_ctrl_c(self, event) -> None:  # noqa: ANN001
        """Ctrl-C handler with double-press exit on idle.

        Busy and "buffer has text" branches mirror the ESC chain so the
        two keys feel interchangeable.  Idle + empty buffer is the only
        place behaviour diverges: a single Ctrl-C arms a 2 s exit
        window instead of killing the REPL outright.
        """
        if self._has_permission_prompt():
            self._reject_active_permission(source="user_abort")
            try:
                event.app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            return

        buf = event.current_buffer
        now = time.monotonic()

        # 1. Engine in flight → interrupt.
        if self._busy:
            self._interrupt_visual_pending = True
            if self._on_interrupt is not None:
                try:
                    self._on_interrupt()
                except Exception:  # noqa: BLE001
                    pass
            self._pending.clear()
            self._last_ctrl_c_at = None
            self._clear_transient_hint()
            try:
                event.app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            return

        # 2. Buffer has text → reset.
        if buf.text:
            buf.reset()
            self._last_ctrl_c_at = None
            self._clear_transient_hint()
            try:
                event.app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            return

        # 3. Idle + empty → double-press exit.
        if self._is_double_press(self._last_ctrl_c_at, _CTRL_C_DOUBLE_PRESS_SEC):
            self._exit_requested = True
            try:
                event.app.exit()
            except Exception:  # noqa: BLE001
                pass
            return

        self._last_ctrl_c_at = now
        self._set_transient_hint(
            "press Ctrl-C again to exit",
            duration=_CTRL_C_DOUBLE_PRESS_SEC,
        )
        try:
            event.app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _is_double_press(last_at: float | None, window: float) -> bool:
        if last_at is None:
            return False
        return (time.monotonic() - last_at) < window

    def _set_transient_hint(
        self,
        text: str,
        *,
        duration: float = _HINT_DISPLAY_SEC,
    ) -> None:
        self._transient_hint = text
        self._transient_hint_until = time.monotonic() + duration

    def _clear_transient_hint(self) -> None:
        self._transient_hint = None
        self._transient_hint_until = 0.0

    def _active_transient_hint(self) -> str | None:
        if self._transient_hint is None:
            return None
        if time.monotonic() >= self._transient_hint_until:
            self._transient_hint = None
            self._transient_hint_until = 0.0
            return None
        return self._transient_hint

    # ------------------------------------------------------------------
    # Filters / text providers — all called on the prompt_toolkit thread
    # ------------------------------------------------------------------

    def _has_permission_prompt(self) -> bool:
        return bool(self._permission_queue)

    def _has_input_frame(self) -> bool:
        return not self._has_permission_prompt()

    def _has_active_group(self) -> bool:
        if self._has_permission_prompt():
            return False
        if self._interrupt_visual_pending:
            return False
        group = self.ui.tool_groups.active
        return group is not None and group.is_active

    def _has_activity(self) -> bool:
        if self._has_permission_prompt():
            return False
        if self._interrupt_visual_pending:
            return False
        # The streaming preview *is* the progress indicator — once the
        # model has emitted visible body the "Pondering / Thinking …"
        # verb line becomes redundant noise (the body is plainly being
        # written out).  Suppress the bar during that window.  When the
        # stream stalls between iterations / runs a tool call / waits
        # for the first token, the preview is empty and the bar comes
        # back so the user still has a heartbeat.
        if self._has_streaming_preview():
            return False
        return bool(self.ui._turn_state.verb) or self._busy

    def _has_reasoning(self) -> bool:
        if self._has_permission_prompt():
            return False
        if self._interrupt_visual_pending:
            return False
        return bool(getattr(self.ui._stream, "reasoning_excerpt", ""))

    def _has_streaming_preview(self) -> bool:
        if self._interrupt_visual_pending:
            return False
        return self._cached_preview_lines > 0

    def _has_bottom_content(self) -> bool:
        # Drives the leading 1-row spacer.  Anything that contributes
        # visible content above the input frame qualifies.
        return (
            self._has_streaming_preview()
            or self._has_permission_prompt()
            or self._has_active_group()
            or self._has_activity()
            or self._has_reasoning()
        )

    def _permission_options(
        self,
        request: ToolPermissionRequest,
    ) -> list[tuple[ToolPermissionDecisionType, str]]:
        if request.allow_session:
            if request.tool_name == "shell":
                session_label = "Yes, allow this command prefix in this session"
            elif request.preview and request.preview.path:
                session_label = "Yes, allow edits in this path during this session"
            else:
                session_label = "Yes, allow similar calls during this session"
            return [
                (ToolPermissionDecisionType.ALLOW_ONCE, "Yes"),
                (ToolPermissionDecisionType.ALLOW_SESSION, session_label),
                (ToolPermissionDecisionType.DENY, "No"),
            ]
        return [
            (ToolPermissionDecisionType.ALLOW_ONCE, "Yes"),
            (ToolPermissionDecisionType.DENY, "No"),
        ]

    def _permission_divider(self) -> str:
        # Mirrors open-claude-code's Pane: a full-width top rule followed by
        # horizontally padded content rather than a dense boxed card.
        width = max(20, self._terminal_cols() - 2)
        return "\u2500" * width

    def _permission_max_rows(self) -> int:
        # Root rows consumed while a permission modal is visible:
        # leading spacer (1) + footer shortcuts (1).  The input frame,
        # completions, activity, active group, and reasoning rows are hidden.
        return max(1, min(20, self._terminal_rows() - 2))

    def _get_permission_text(self) -> FormattedText:
        if not self._permission_queue:
            return FormattedText([])
        prompt = self._permission_queue[0]
        request = prompt.request
        preview = request.preview
        title = preview.title if preview else f"Use {request.tool_name}"
        subtitle = preview.subtitle or preview.path if preview else None
        options = self._permission_options(request)
        selected = max(0, min(prompt.selected_index, len(options) - 1))

        max_rows = self._permission_max_rows()
        critical_rows = 1 + len(options)
        optional_budget = max(0, max_rows - critical_rows)
        optional_rows = 0

        fragments: list[tuple[str, str]] = []

        def add_optional(style: str, text: str) -> bool:
            nonlocal optional_rows
            if optional_rows >= optional_budget:
                return False
            fragments.append((style, text))
            optional_rows += 1
            return True

        if optional_budget >= 4:
            add_optional("class:permission.border", f"{self._permission_divider()}\n")
            add_optional("class:permission.subtitle", "\n")
            add_optional("class:permission.title", f"  {title}\n")
            if subtitle:
                add_optional("class:permission.subtitle", f"  {subtitle}\n")
        elif optional_budget >= 1:
            add_optional("class:permission.border", f"{self._permission_divider()}\n")

        if preview and preview.command:
            command_lines = preview.command.splitlines() or [preview.command]
            if optional_rows < optional_budget and command_lines:
                add_optional("class:permission.subtitle", "\n")
            for line in command_lines:
                if not add_optional("class:permission.command", f"  $ {line}\n"):
                    break
        if preview and preview.body:
            body_lines = preview.body.splitlines()[:4]
            if optional_rows < optional_budget and body_lines:
                add_optional("class:permission.subtitle", "\n")
            for line in body_lines:
                if not add_optional("class:permission.subtitle", f"  {line}\n"):
                    break
        if preview and preview.diff:
            diff_lines = preview.diff.splitlines()
            max_lines = 8
            shown = 0
            if optional_rows < optional_budget and diff_lines:
                add_optional("class:permission.subtitle", "\n")
            for line in diff_lines[:max_lines]:
                if not add_optional("class:permission.diff", f"  {line}\n"):
                    break
                shown += 1
            if shown < len(diff_lines) and optional_rows < optional_budget:
                remaining = len(diff_lines) - shown
                add_optional(
                    "class:permission.diff",
                    f"  ... {remaining} more diff lines ...\n",
                )

        question = self._permission_question(request)
        if optional_rows < optional_budget and fragments:
            add_optional("class:permission.subtitle", "\n")
        fragments.append(("class:permission.question", f"  {question}\n"))

        for idx, (_decision, label) in enumerate(options):
            marker = "\u203a" if idx == selected else " "
            style = (
                "class:permission.option.selected"
                if idx == selected
                else "class:permission.option"
            )
            fragments.append((style, f"  {marker} {idx + 1}. {label}\n"))
        return FormattedText(fragments)

    def _permission_height(self) -> D:
        if not self._permission_queue:
            return D.exact(0)
        text = self._get_permission_text()
        rows = 0
        for _style, fragment in text:
            rows += max(1, fragment.count("\n") + (0 if fragment.endswith("\n") else 1))
        return D.exact(max(1, min(rows, self._permission_max_rows())))

    @staticmethod
    def _permission_question(request: ToolPermissionRequest) -> str:
        preview = request.preview
        if request.tool_name == "file_edit":
            target = Path(preview.path).name if preview and preview.path else "this file"
            return f"Do you want to make this edit to {target}?"
        if request.tool_name == "write_file":
            target = Path(preview.path).name if preview and preview.path else "this file"
            return f"Do you want to write {target}?"
        if request.tool_name == "shell":
            return "Do you want to run this command?"
        return f"Do you want to allow {request.tool_name}?"

    def _get_active_group_text(self) -> ANSI:
        group = self.ui.tool_groups.active
        if group is None or not group.is_active:
            return ANSI("")
        cols = self._terminal_cols()
        out = self._renderer.render(group.render_headline(active=True), width=cols)
        hint = group.render_hint()
        if hint is not None:
            out += "\n" + self._renderer.render(hint, width=cols)
        return ANSI(out.rstrip("\r\n"))

    def _get_streaming_preview_text(self) -> ANSI:
        """Return the cached streaming preview as ANSI.

        The cache is refreshed at the start of every prompt_toolkit
        render cycle by :meth:`_refresh_preview_cache` so this method
        and :meth:`_streaming_preview_height` always agree on what's
        being drawn for a given frame.
        """
        return ANSI(self._cached_preview_text)

    def _streaming_preview_max_rows(self) -> int:
        """Compute the streaming preview cap from current terminal height.

        Returns ``terminal_rows - reserved`` so the preview window
        keeps growing with the in-flight response — claude-code-style
        — instead of hitting a hard 12-row ceiling and tail-cropping
        the rest of the stream.  Falls back to
        ``_streaming_preview_min_cap`` (12 rows) when the geometry
        can't be queried (piped / detached stdout in tests).

        The cap is recomputed every call (cheap — one ``get_size``)
        so a terminal resize while a stream is in flight immediately
        adjusts the available preview area.
        """
        try:
            rows = int(self._app.output.get_size().rows)
        except Exception:  # noqa: BLE001
            return self._streaming_preview_min_cap
        return max(
            self._streaming_preview_min_cap,
            rows - self._streaming_preview_reserved_rows,
        )

    def _streaming_preview_height(self) -> D:
        """Exact-row dimension matching the cached preview snapshot."""
        if self._cached_preview_lines <= 0:
            return D.exact(0)
        return D.exact(min(self._cached_preview_lines, self._streaming_preview_max_rows()))

    def _refresh_preview_cache(self, *_args, **_kwargs) -> None:
        """Recompute the streaming preview snapshot for the next render.

        Called from prompt_toolkit's ``before_render`` event so the
        height callback and the text callback see identical state for
        the duration of one render frame.  Defensive against engine
        thread mutations: takes a list snapshot of ``_stream_buffer``
        before joining so a concurrent ``append`` can't tear the read.

        Routes through :meth:`CLIUI._render_assistant` — the *same*
        renderer that flushes the finalised body to scrollback at
        ``end_stream`` — so the in-flight preview matches the eventual
        scrollback output character-for-character (markdown headings,
        bold, code blocks, lists, …).  Partial / unclosed markdown
        constructs are rendered as Rich's parser interprets them at
        that instant; the next streamed token re-parses and the
        rendering self-corrects, mirroring claude-code's behaviour.
        """
        try:
            stream = self.ui._stream
            if not stream.active or not self.ui._stream_buffer:
                self._cached_preview_text = ""
                self._cached_preview_lines = 0
                return

            buf_snapshot = list(self.ui._stream_buffer)
            raw = "".join(buf_snapshot)
            cleaned = strip_tool_blocks(raw).rstrip()
            # Mid-stream phantom-command guard: when the body carries
            # an *unclosed* shell fence ("\u0060\u0060\u0060bash ls -la …") and the
            # leading prose has imperative verbs ("let me run / 我来"),
            # strip *every* shell fence — Kimi-class models routinely
            # emit 2-3 ``\u0060\u0060\u0060bash`` blocks interleaved with prose
            # narration ("我先看看：…  实际上让我用更系统的方式查看：…"),
            # and the previous trailing-only stripper was leaving the
            # earlier blocks on screen.  We still restrict the gate
            # to "at least one unclosed fence" so legitimate fully
            # closed code examples render normally as the user types
            # them.
            if cleaned and _has_unclosed_command_fence(cleaned) and _looks_like_intended_tool_use(cleaned):
                cleaned = strip_all_command_fences(cleaned)
            if not cleaned:
                self._cached_preview_text = ""
                self._cached_preview_lines = 0
                return

            cols = self._terminal_cols()
            rendered = self._renderer.render(
                self.ui._render_assistant(cleaned),
                width=cols,
            )
            lines = rendered.splitlines()
            # Tail-crop only when the rendered body actually exceeds
            # the (terminal-aware) preview cap so users see the latest
            # tokens; below the cap every single newline grows the
            # preview window so the visual length tracks the response
            # length all the way through streaming.
            max_rows = max(1, self._streaming_preview_max_rows())
            if len(lines) > max_rows:
                lines = lines[-max_rows:]
            stripped = [line.rstrip() for line in lines]
            self._cached_preview_text = "\n".join(stripped)
            self._cached_preview_lines = len(stripped)
        except Exception:  # noqa: BLE001 — never let preview prep kill the app
            self._cached_preview_text = ""
            self._cached_preview_lines = 0

    def _get_activity_text(self) -> ANSI:
        if not self._has_activity():
            return ANSI("")
        cols = self._terminal_cols()
        text = self._renderer.render(ActivityBar(self.ui._turn_state), width=cols)
        return ANSI(text.rstrip("\r\n"))

    def _get_reasoning_text(self) -> ANSI:
        excerpt = getattr(self.ui._stream, "reasoning_excerpt", "") or ""
        if not excerpt:
            return ANSI("")
        from aether.cli.ui import _render_reasoning_excerpt

        cols = self._terminal_cols()
        text = self._renderer.render(_render_reasoning_excerpt(excerpt), width=cols)
        return ANSI(text.rstrip("\r\n"))

    def _get_footer_text(self) -> FormattedText:
        sep = ("class:footer.sep", "  " + (icon("dot") or "·") + "  ")

        # Esc gets a context-aware label so the footer hint reflects
        # what the next ESC will actually do.  Visible at all times
        # (not gated on ``_busy``) so the user always knows the key
        # exists — claude-code does the same with its "press ESC to
        # cancel" / "ESC clear" rotation.
        if self._has_permission_prompt():
            esc_label = " reject"
        elif self._busy:
            esc_label = (
                " interrupted"
                if self._interrupt_visual_pending
                else " interrupt"
            )
        elif self._pending:
            esc_label = " pop queued"
        else:
            esc_label = " clear"

        if self._has_permission_prompt():
            option_count = len(self._permission_options(self._permission_queue[0].request))
            option_keys = "1" if option_count <= 1 else f"1-{option_count}"
            fragments: list[tuple[str, str]] = [
                ("class:footer", "  "),
                ("class:footer.kbd", "Enter"),
                ("class:footer", " approve"),
                sep,
                ("class:footer.kbd", option_keys),
                ("class:footer", " choose"),
                sep,
                ("class:footer.kbd", "Up/Down"),
                ("class:footer", " move"),
                sep,
                ("class:footer.kbd", "Esc"),
                ("class:footer", esc_label),
            ]
            return FormattedText(fragments)

        fragments = [
            ("class:footer", "  "),
            ("class:footer.kbd", "Enter"),
            ("class:footer", " send"),
            sep,
            ("class:footer.kbd", "Shift+Enter"),
            ("class:footer", " newline"),
            sep,
            ("class:footer.kbd", "Esc"),
            ("class:footer", esc_label),
            sep,
            ("class:footer.kbd", "/help"),
            ("class:footer", " commands"),
            sep,
            ("class:footer.kbd", "Ctrl-D"),
            ("class:footer", " exit"),
        ]
        if self._interrupt_visual_pending:
            fragments.append(sep)
            fragments.append(("class:footer.busy", "interrupting…"))
        elif self._busy and self._pending:
            fragments.append(sep)
            fragments.append(
                ("class:footer.queued", f"▸ queued ({len(self._pending)})"),
            )
        elif self._busy:
            fragments.append(sep)
            fragments.append(("class:footer.busy", "running…"))

        hint = self._active_transient_hint()
        if hint:
            fragments.append(sep)
            fragments.append(("class:footer.busy", hint))
        return FormattedText(fragments)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _terminal_cols(self) -> int:
        try:
            cols = self._app.output.get_size().columns
            if cols and cols > 0:
                return min(cols, 200)
        except Exception:  # noqa: BLE001
            pass
        return 100

    def _terminal_rows(self) -> int:
        try:
            rows = self._app.output.get_size().rows
            if rows and rows > 0:
                return min(rows, 80)
        except Exception:  # noqa: BLE001
            pass
        return 24

    @property
    def is_busy(self) -> bool:
        """True while an ``on_submit`` invocation is in flight."""
        return self._busy

    @property
    def pending_count(self) -> int:
        """Number of queued messages awaiting dispatch."""
        return len(self._pending)

    def enqueue(self, line: str) -> None:
        """Programmatic alternative to pressing Enter — used by tests."""
        if not line:
            return
        self._pending.append(line)
        self._pending_event.set()
        try:
            self._app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    def make_tool_permission_prompter(self, loop: asyncio.AbstractEventLoop):
        from aether.cli.tool_permission_prompter import AetherToolPermissionPrompter

        return AetherToolPermissionPrompter(self, loop)

    def enqueue_permission_request(
        self,
        request: ToolPermissionRequest,
        future: Future[ToolPermissionDecision],
    ) -> None:
        if future.done():
            return
        self._permission_queue.append(PendingPermissionPrompt(request=request, future=future))
        try:
            self._app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    def cancel_permission_request(
        self,
        future: Future[ToolPermissionDecision],
    ) -> None:
        for idx, prompt in enumerate(list(self._permission_queue)):
            if prompt.future is future:
                self._permission_queue.pop(idx)
                if not future.done():
                    future.set_result(
                        ToolPermissionDecision(
                            type=ToolPermissionDecisionType.DENY,
                            feedback="permission prompt cancelled",
                            source="cancelled",
                        )
                    )
                break
        try:
            self._app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    def _move_permission_selection(self, delta: int) -> None:
        if not self._permission_queue:
            return
        prompt = self._permission_queue[0]
        option_count = len(self._permission_options(prompt.request))
        if option_count <= 0:
            prompt.selected_index = 0
            return
        prompt.selected_index = (prompt.selected_index + delta) % option_count

    def _resolve_permission_number(self, number: int) -> bool:
        return self._resolve_permission_index(number - 1)

    def _resolve_permission_index(self, index: int) -> bool:
        if not self._permission_queue:
            return False
        prompt = self._permission_queue[0]
        options = self._permission_options(prompt.request)
        if index < 0 or index >= len(options):
            return False
        self._permission_queue.pop(0)
        decision_type, _label = options[index]
        if not prompt.future.done():
            prompt.future.set_result(
                ToolPermissionDecision(
                    type=decision_type,
                    source="user",
                )
            )
        return True

    def _resolve_active_permission(self) -> None:
        if not self._permission_queue:
            return
        prompt = self._permission_queue[0]
        options = self._permission_options(prompt.request)
        idx = max(0, min(prompt.selected_index, len(options) - 1))
        self._resolve_permission_index(idx)

    def _reject_active_permission(self, *, source: str = "user") -> None:
        if not self._permission_queue:
            return
        prompt = self._permission_queue.pop(0)
        if not prompt.future.done():
            prompt.future.set_result(
                ToolPermissionDecision(
                    type=ToolPermissionDecisionType.ABORT,
                    source=source,
                )
            )

    def _abort_all_permission_requests(self) -> None:
        while self._permission_queue:
            prompt = self._permission_queue.pop(0)
            if not prompt.future.done():
                prompt.future.set_result(
                    ToolPermissionDecision(
                        type=ToolPermissionDecisionType.ABORT,
                        source="shutdown",
                    )
                )

    # ------------------------------------------------------------------
    # Async loops
    # ------------------------------------------------------------------

    async def _consumer_loop(self) -> None:
        """Drain the pending queue: invoke ``on_submit`` once per message."""
        while not self._exit_requested:
            if not self._pending:
                self._pending_event.clear()
                try:
                    await self._pending_event.wait()
                except asyncio.CancelledError:
                    return
                if self._exit_requested:
                    return
                continue
            line = self._pending.pop(0)
            self._interrupt_visual_pending = False
            self._busy = True
            try:
                self._app.invalidate()
            except Exception:  # noqa: BLE001
                pass
            try:
                await self._on_submit(line)
            except Exception as exc:  # noqa: BLE001 — engine errors shouldn't kill the loop
                try:
                    self.ui.error(f"engine error: {exc}")
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self._busy = False
                self._interrupt_visual_pending = False
                # Clear any leftover verb so the bar collapses while we
                # wait for the next user message.
                try:
                    self.ui._turn_state.verb = ""
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self._app.invalidate()
                except Exception:  # noqa: BLE001
                    pass

    async def _refresh_loop(self) -> None:
        """Tick the activity bar's elapsed counter at ~10 Hz."""
        while not self._exit_requested:
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                return
            try:
                self._app.invalidate()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the Application until the user requests exit (Ctrl-D)."""
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        try:
            with patch_stdout(raw=True):
                await self._app.run_async()
        except EOFError:
            # stdin closed (e.g. piped input or programmatic ``request_exit``
            # after the app has already torn down its input stream) — treat
            # as a clean shutdown.
            pass
        finally:
            self._exit_requested = True
            self._abort_all_permission_requests()
            self._pending_event.set()
            for t in (self._consumer_task, self._refresh_task):
                if t is None:
                    continue
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            self.ui.managed_externally = False

    def request_exit(self) -> None:
        """Signal the application to exit on the next render cycle."""
        self._exit_requested = True
        try:
            self._app.exit()
        except Exception:  # noqa: BLE001
            pass
