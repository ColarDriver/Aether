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
from io import StringIO
from typing import Awaitable, Callable, Optional

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    Float,
    FloatContainer,
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


SubmitCallback = Callable[[str], Awaitable[None]]
InterruptCallback = Callable[[], None]


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
        on_interrupt: invoked when the user presses Ctrl-C while the
            engine is busy.  Should call ``engine.interrupt(...)``.  The
            REPL is never killed by Ctrl-C; the buffer is preserved so
            the user can re-submit.
    """

    def __init__(
        self,
        ui: CLIUI,
        on_submit: SubmitCallback,
        *,
        history: History | None = None,
        completer: Completer | None = None,
        on_interrupt: Optional[InterruptCallback] = None,
    ) -> None:
        self.ui = ui
        self._on_submit = on_submit
        self._on_interrupt = on_interrupt
        self._history = history
        self._completer = completer

        self._renderer = _RichRenderer()

        # Streaming preview cap — the bottom region shows at most this
        # many rows of the in-flight assistant body.  When the response
        # exceeds the cap the *tail* is shown so users see the latest
        # tokens; the full formatted body still lands in scrollback at
        # ``end_stream``.
        self._streaming_preview_max_rows: int = 12

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
        # We use a list rather than ``asyncio.Queue`` so the footer can
        # render the queue length without consuming it.  An ``Event``
        # gates the consumer's wait when the queue is empty.
        self._pending: list[str] = []
        self._pending_event = asyncio.Event()
        self._exit_requested = False
        self._consumer_task: asyncio.Task[None] | None = None
        self._refresh_task: asyncio.Task[None] | None = None

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

        root = HSplit(
            [
                spacer_container,
                streaming_preview_container,
                active_group_container,
                activity_container,
                reasoning_container,
                input_frame,
                footer_window,
            ],
        )

        layout = Layout(
            FloatContainer(
                content=root,
                floats=[
                    Float(
                        xcursor=True,
                        ycursor=True,
                        content=CompletionsMenu(max_height=12, scroll_offset=1),
                    ),
                ],
            )
        )

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
        # render cycle so the height callback and the text callback see
        # the same buffer state — otherwise the engine thread can append
        # tokens between the two queries and the activity bar will
        # overpaint the preview's tail row.
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
            buf = event.current_buffer
            if self._busy:
                # Engine in flight — ask it to stop, drop pending queue
                # entries, but keep the REPL alive.
                if self._on_interrupt is not None:
                    try:
                        self._on_interrupt()
                    except Exception:  # noqa: BLE001
                        pass
                self._pending.clear()
                event.app.invalidate()
                return
            if buf.text:
                buf.reset()
                event.app.invalidate()
                return
            # Ctrl-C on empty buffer while idle — exit.
            self._exit_requested = True
            event.app.exit()

        @bindings.add("escape", "enter")
        def _esc_enter(event):  # noqa: ANN001
            event.current_buffer.insert_text("\n")

        @bindings.add("c-j")
        def _ctrl_j(event):  # noqa: ANN001
            # Fallback newline for terminals that swallow Esc-Enter.
            event.current_buffer.insert_text("\n")

        @bindings.add("enter")
        def _enter(event):  # noqa: ANN001
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
    # Filters / text providers — all called on the prompt_toolkit thread
    # ------------------------------------------------------------------

    def _has_active_group(self) -> bool:
        group = self.ui.tool_groups.active
        return group is not None and group.is_active

    def _has_activity(self) -> bool:
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
        return bool(getattr(self.ui._stream, "reasoning_excerpt", ""))

    def _has_streaming_preview(self) -> bool:
        return self._cached_preview_lines > 0

    def _has_bottom_content(self) -> bool:
        # Drives the leading 1-row spacer.  Anything that contributes
        # visible content above the input frame qualifies.
        return (
            self._has_streaming_preview()
            or self._has_active_group()
            or self._has_activity()
            or self._has_reasoning()
        )

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

    def _streaming_preview_height(self) -> D:
        """Exact-row dimension matching the cached preview snapshot."""
        if self._cached_preview_lines <= 0:
            return D.exact(0)
        return D.exact(min(self._cached_preview_lines, self._streaming_preview_max_rows))

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
            max_rows = max(1, self._streaming_preview_max_rows)
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
        fragments: list[tuple[str, str]] = [
            ("class:footer", "  "),
            ("class:footer.kbd", "Enter"),
            ("class:footer", " send"),
            sep,
            ("class:footer.kbd", "Esc+Enter"),
            ("class:footer", " newline"),
            sep,
            ("class:footer.kbd", "/help"),
            ("class:footer", " commands"),
            sep,
            ("class:footer.kbd", "Ctrl-D"),
            ("class:footer", " exit"),
        ]
        if self._busy and self._pending:
            fragments.append(sep)
            fragments.append(
                ("class:footer.queued", f"▸ queued ({len(self._pending)})"),
            )
        elif self._busy:
            fragments.append(sep)
            fragments.append(("class:footer.busy", "running…  Ctrl-C to stop"))
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
