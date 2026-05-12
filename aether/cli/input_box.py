"""Bordered, multi-line input box for the Aether REPL.

Builds a small ``prompt_toolkit`` ``Application`` (non-fullscreen) that
renders a Frame around a multi-line buffer with:

  * Enter         — submit (only when the buffer is non-empty)
  * Alt/Esc+Enter — insert newline
  * Ctrl-J        — also insert newline (works on terminals that swallow
                    Esc-Enter)
  * Ctrl-C        — raise ``KeyboardInterrupt``
  * Ctrl-D        — raise ``EOFError`` (only when the buffer is empty)
  * ↑ / ↓         — history navigation when the cursor sits on the first
                    or last line and the buffer is single-line

The frame stays in scrollback after submission so the conversation reads
as a back-and-forth: the user's just-submitted message is visible above
the assistant's reply.
"""

from __future__ import annotations

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.containers import Float, FloatContainer, HSplit
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

from aether.cli.theme import (
    AETHER_ACCENT,
    AETHER_BORDER,
    AETHER_DIM,
    AETHER_PRIMARY,
    icon,
)


_BOX_STYLE = Style.from_dict(
    {
        "frame.border": AETHER_BORDER,
        "input.icon": f"bold {AETHER_PRIMARY}",
        "input.text": "",
        "input.placeholder": f"italic {AETHER_DIM}",
        "footer": AETHER_DIM,
        "footer.kbd": f"bold {AETHER_ACCENT}",
        "footer.sep": AETHER_BORDER,
        # Completion popup palette
        "completion-menu":              f"bg:#1E293B {AETHER_DIM}",
        "completion-menu.completion":   f"bg:#1E293B {AETHER_DIM}",
        "completion-menu.completion.current": f"bg:{AETHER_PRIMARY} #FFFFFF bold",
        "completion-menu.meta":         f"bg:#1E293B {AETHER_DIM} italic",
        "completion-menu.meta.current": f"bg:{AETHER_PRIMARY} #FFFFFF italic",
        "completion-menu.multi-column-meta": f"bg:#1E293B {AETHER_DIM} italic",
    }
)


def _footer_fragments() -> list[tuple[str, str]]:
    sep = ("class:footer.sep", "  " + icon("dot") + "  ")
    return [
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


def prompt_box(
    *,
    history: History | None = None,
    completer: Completer | None = None,
) -> str:
    """Show the input frame and return the user's submitted text.

    Args:
        history:   prompt_toolkit ``History`` for ↑/↓ recall.
        completer: optional ``Completer``; when provided, completions are
                   shown as a popup as the user types (e.g. the slash-command
                   palette triggered by ``/``).

    Raises:
        EOFError — when Ctrl-D is pressed on an empty buffer
        KeyboardInterrupt — when Ctrl-C is pressed
    """
    buffer = Buffer(
        history=history,
        multiline=True,
        completer=completer,
        complete_while_typing=True,
    )

    bindings = KeyBindings()

    @bindings.add("c-d")
    def _ctrl_d(event):  # noqa: ANN001
        if not event.current_buffer.text:
            event.app.exit(exception=EOFError(), style="class:exiting")
        else:
            event.current_buffer.delete()

    @bindings.add("c-c")
    def _ctrl_c(event):  # noqa: ANN001
        event.app.exit(exception=KeyboardInterrupt(), style="class:exiting")

    @bindings.add("escape", "enter")
    def _esc_enter(event):  # noqa: ANN001
        event.current_buffer.insert_text("\n")

    @bindings.add("c-j")  # ^J — fallback newline for terminals that drop Esc-Enter
    def _ctrl_j(event):  # noqa: ANN001
        event.current_buffer.insert_text("\n")

    @bindings.add("enter")
    def _enter(event):  # noqa: ANN001
        buf = event.current_buffer
        # If the completion popup is open AND the user has highlighted a
        # candidate (Tab / ↓ pressed), Enter accepts that completion
        # instead of submitting the message.
        if buf.complete_state and buf.complete_state.current_completion:
            buf.apply_completion(buf.complete_state.current_completion)
            return
        # Close any "ghost" popup that hasn't been navigated yet.
        if buf.complete_state:
            buf.cancel_completion()
        text = buf.text
        if text.strip() == "":
            buf.insert_text("\n")
            return
        event.app.exit(result=text)

    control = BufferControl(
        buffer=buffer,
        input_processors=[BeforeInput(f"{icon('user')} ", style="class:input.icon")],
        focusable=True,
    )

    input_window = Window(
        content=control,
        wrap_lines=True,
        # Start at one row, grow as the user adds newlines, and cap at
        # 12 rows so a very long paste stays scrollable inside the box
        # instead of taking over the screen.
        height=D(min=1, max=12),
        dont_extend_height=True,
    )

    frame = Frame(input_window)

    root = HSplit(
        [
            frame,
            Window(
                FormattedTextControl(_footer_fragments(), focusable=False),
                height=1,
                dont_extend_height=True,
            ),
        ],
        # Without this, the HSplit asks the terminal to reserve enough
        # rows for the frame's *max* height (12) before the user types
        # anything, which renders as a huge empty box on first paint.
        height=D(min=4),
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

    app: Application[str] = Application(
        layout=layout,
        key_bindings=bindings,
        style=_BOX_STYLE,
        full_screen=False,
        mouse_support=False,
        # Wipe the box (and footer) after submit so the REPL becomes a
        # clean log: caller is expected to echo the entered line itself.
        erase_when_done=True,
    )

    return app.run()
