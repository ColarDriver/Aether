"""Bordered list-picker UI used by ``/model`` and similar selectors.

A small ``prompt_toolkit`` ``Application`` (non-fullscreen, erased on exit)
that renders:

  * a header line  (the picker title)
  * a list of items, one per row, with the active row highlighted
  * a footer line  (key hints)

Returns the selected value, or ``None`` if the user cancels (Esc/Ctrl-C).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Sequence

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

from aether.cli.theme import (
    AETHER_ACCENT,
    AETHER_BORDER,
    AETHER_DIM,
    AETHER_PRIMARY,
    AETHER_TEXT,
    icon,
)


# ---------------------------------------------------------------------------
# Item type & style
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PickerItem:
    value: str
    label: str
    description: str = ""


_PICKER_STYLE = Style.from_dict(
    {
        "frame.border": AETHER_BORDER,
        "picker.title": f"bold {AETHER_PRIMARY}",
        "picker.row.cursor": f"bold {AETHER_PRIMARY}",
        "picker.row.label": AETHER_TEXT,
        "picker.row.label.current": f"bold {AETHER_ACCENT}",
        "picker.row.desc": f"italic {AETHER_DIM}",
        "picker.row.check": f"bold {AETHER_ACCENT}",
        "picker.row.highlight": f"reverse {AETHER_PRIMARY}",
        "picker.footer": AETHER_DIM,
        "picker.footer.kbd": f"bold {AETHER_ACCENT}",
        "picker.footer.sep": AETHER_BORDER,
        "picker.empty": f"italic {AETHER_DIM}",
    }
)


# ---------------------------------------------------------------------------
# pick()
# ---------------------------------------------------------------------------

def pick(
    items: Sequence[PickerItem],
    *,
    title: str = "Select",
    current: str | None = None,
    page_size: int = 10,
) -> str | None:
    """Render the picker; return the selected value or ``None``."""
    if not items:
        return None

    # Initial cursor position: prefer the row that matches *current*.
    state = {"index": 0, "offset": 0}
    if current:
        for i, item in enumerate(items):
            if item.value == current:
                state["index"] = i
                break

    # We render the picker as a full_screen modal (alternate buffer)
    # rather than non-fullscreen.  In non-fullscreen mode prompt_toolkit
    # only gets the rows below the current cursor; if there isn't enough
    # space, fixed-height windows get silently clipped from the bottom
    # and the highlighted row appears to "vanish" until the user pages
    # back into the visible region.  Full-screen guarantees we have the
    # whole terminal to work with and the previous REPL state is
    # restored when the picker exits.
    term_lines = shutil.get_terminal_size((80, 24)).lines
    # chrome: frame(2) + title(1) + footer(1) + indicators(2) = 6
    fit_page = max(3, term_lines - 6)
    page = max(3, min(page_size, len(items), fit_page))

    # Compute the longest label so descriptions align as a column.
    label_width = min(40, max((len(it.label) for it in items), default=0))

    def _ensure_visible() -> None:
        # Clamp offset into a legal range first — callers may have moved
        # ``index`` past the end (wrap-around) and we want the visible
        # window to track that without going past the last item.
        last_offset = max(0, len(items) - page)
        if state["index"] < state["offset"]:
            state["offset"] = state["index"]
        elif state["index"] >= state["offset"] + page:
            state["offset"] = state["index"] - page + 1
        if state["offset"] > last_offset:
            state["offset"] = last_offset
        if state["offset"] < 0:
            state["offset"] = 0

    def _render() -> list[tuple[str, str]]:
        _ensure_visible()
        fragments: list[tuple[str, str]] = []
        visible = items[state["offset"] : state["offset"] + page]

        # Up indicator
        if state["offset"] > 0:
            fragments.append(("class:picker.row.desc", f"   {icon('arrow')} more above\n"))
        else:
            fragments.append(("", "\n"))

        for local_idx, item in enumerate(visible):
            absolute = state["offset"] + local_idx
            is_cursor = absolute == state["index"]
            is_current = item.value == current

            cursor = icon("arrow") if is_cursor else " "
            check = " " + icon("success") if is_current else "  "

            cursor_style = "class:picker.row.cursor" if is_cursor else ""
            label_style = (
                "class:picker.row.label.current"
                if is_current
                else "class:picker.row.label"
            )
            desc_style = "class:picker.row.desc"
            check_style = "class:picker.row.check" if is_current else ""

            if is_cursor:
                fragments.append(("class:picker.row.highlight", f" {cursor} "))
            else:
                fragments.append((cursor_style, f" {cursor} "))

            label = item.label.ljust(label_width)
            fragments.append((label_style, label))
            if item.description:
                fragments.append((desc_style, f"   {item.description}"))
            fragments.append((check_style, check))
            fragments.append(("", "\n"))

        # Down indicator
        if state["offset"] + page < len(items):
            fragments.append(
                ("class:picker.row.desc", f"   {icon('arrow')} more below\n")
            )

        return fragments

    bindings = KeyBindings()

    @bindings.add("up")
    @bindings.add("c-p")
    def _up(event):  # noqa: ANN001
        state["index"] = (state["index"] - 1) % len(items)

    @bindings.add("down")
    @bindings.add("c-n")
    def _down(event):  # noqa: ANN001
        state["index"] = (state["index"] + 1) % len(items)

    @bindings.add("pageup")
    def _pageup(event):  # noqa: ANN001
        state["index"] = max(0, state["index"] - page)

    @bindings.add("pagedown")
    def _pagedown(event):  # noqa: ANN001
        state["index"] = min(len(items) - 1, state["index"] + page)

    @bindings.add("home")
    def _home(event):  # noqa: ANN001
        state["index"] = 0

    @bindings.add("end")
    def _end(event):  # noqa: ANN001
        state["index"] = len(items) - 1

    @bindings.add("enter")
    def _enter(event):  # noqa: ANN001
        event.app.exit(result=items[state["index"]].value)

    @bindings.add("escape")
    @bindings.add("c-c")
    def _cancel(event):  # noqa: ANN001
        event.app.exit(result=None)

    title_text = [
        ("class:picker.title", f" {icon('logo')} {title} "),
    ]
    title_window = Window(
        FormattedTextControl(title_text, focusable=False),
        height=1,
        dont_extend_height=True,
    )

    # +2 leaves room for the "more above"/"more below" indicator rows.
    list_height = page + 2
    list_window = Window(
        FormattedTextControl(_render, focusable=True, show_cursor=False),
        wrap_lines=False,
        always_hide_cursor=True,
        height=list_height,
    )

    footer_fragments: list[tuple[str, str]] = []
    sep = ("class:picker.footer.sep", "  " + icon("dot") + "  ")
    for kbd, label in (
        ("↑↓", "navigate"),
        ("Enter", "select"),
        ("Esc", "cancel"),
    ):
        footer_fragments.append(("class:picker.footer.kbd", kbd))
        footer_fragments.append(("class:picker.footer", f" {label}"))
        footer_fragments.append(sep)
    footer_fragments = footer_fragments[:-1]
    footer_fragments.insert(0, ("class:picker.footer", "  "))

    footer_window = Window(
        FormattedTextControl(footer_fragments, focusable=False),
        height=1,
        dont_extend_height=True,
    )

    body = HSplit(
        [
            title_window,
            list_window,
            footer_window,
        ]
    )

    layout = Layout(Frame(body))

    app: Application[str | None] = Application(
        layout=layout,
        key_bindings=bindings,
        style=_PICKER_STYLE,
        full_screen=True,
        mouse_support=False,
    )

    return app.run()
