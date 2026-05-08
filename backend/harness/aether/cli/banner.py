"""Welcome banner: wordmark + mascot panel for the Aether CLI."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aether.cli.theme import (
    AETHER_ACCENT,
    AETHER_BORDER,
    AETHER_DIM,
    AETHER_PRIMARY,
    AETHER_PRIMARY_DIM,
    AETHER_TEXT,
    icon,
)


# ---------------------------------------------------------------------------
# Terminal mascot — a soft round companion inspired by the product icon.
# The face is drawn with light shell fills, dark eyes, and pink cheeks so
# the banner feels friendlier than the previous hard-edged robot glyph.
# ---------------------------------------------------------------------------

_MASCOT_SHELL = "#F8FAFC"
_MASCOT_SHELL_EDGE = "#CBD5E1"
_MASCOT_FEATURE = "#111827"
_MASCOT_BLUSH = "#F9A8D4"

# ``w`` means "shell fill cell" and is rendered as a white background
# space.  All other visible characters are outline/features.
_MASCOT_TEMPLATE_LINES: list[str] = []

_LOGO_LINES: list[str] = [
    " █████╗ ███████╗████████╗██╗  ██╗███████╗██████╗ ",
    "██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗",
    "███████║█████╗     ██║   ███████║█████╗  ██████╔╝",
    "██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗",
    "██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║",
    "╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝",
]

# Compact form for narrow terminals — keep the same rounded character but
# collapse it into three rows so the wordmark still fits.
_COMPACT_MASCOT_TEMPLATE_LINES: list[str] = []

_COMPACT_LOGO_LINES: list[str] = [
    "  ▄▀█ █▀▀ ▀█▀ █░█ █▀▀ █▀█  ",
    "  █▀█ ██▄ ░█░ █▀█ ██▄ █▀▄  ",
    "                           ",
]

# Padding inserted between mascot and wordmark.
_LOGO_GUTTER = "  "


def _gradient_color(step: int, total: int) -> str:
    """Return a hex colour interpolating primary → accent across *total* steps."""
    # crude 2-stop gradient; good enough for 6-row block art
    p_r, p_g, p_b = 0x7C, 0x5C, 0xFF
    a_r, a_g, a_b = 0x22, 0xD3, 0xEE
    t = 0.0 if total <= 1 else step / max(1, total - 1)
    r = int(p_r + (a_r - p_r) * t)
    g = int(p_g + (a_g - p_g) * t)
    b = int(p_b + (a_b - p_b) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def _mascot_color(step: int, total: int) -> str:
    """Reverse gradient (accent → primary) so the mascot reads cyan-on-top."""
    return _gradient_color(total - 1 - step, total)


def _render_mascot_line(template: str, *, compact: bool = False) -> Text:
    """Render one mascot row from a shell-fill template."""
    text = Text(no_wrap=True)
    for ch in template:
        if ch == "w":
            text.append(" ", style=f"on {_MASCOT_SHELL}")
        elif ch in {"●", "◉"}:
            text.append(ch, style=f"bold {_MASCOT_FEATURE} on {_MASCOT_SHELL}")
        elif ch in {"◌", "◍"}:
            text.append(ch, style=f"bold {_MASCOT_BLUSH} on {_MASCOT_SHELL}")
        elif ch in {"◡", "︶", "⌣"}:
            text.append(ch, style=f"bold {_MASCOT_FEATURE} on {_MASCOT_SHELL}")
        elif ch in {"╭", "╮", "╰", "╯", "◖", "◗"}:
            text.append(ch, style=f"bold {_MASCOT_SHELL_EDGE}")
        else:
            text.append(ch, style=f"{AETHER_TEXT}")
    return text


def _render_logo(*, compact: bool = False) -> Text:
    """Render the mascot + AETHER wordmark side-by-side as a single Text block."""
    mascot_lines = _COMPACT_MASCOT_TEMPLATE_LINES if compact else _MASCOT_TEMPLATE_LINES
    word_lines = _COMPACT_LOGO_LINES if compact else _LOGO_LINES

    # Pair lines defensively in case future edits change the row counts.
    rows = max(len(mascot_lines), len(word_lines))
    mascot_width = max((len(l) for l in mascot_lines), default=0)
    word_width = max((len(l) for l in word_lines), default=0)

    text = Text(no_wrap=True)
    for i in range(rows):
        m = (mascot_lines[i] if i < len(mascot_lines) else "").ljust(mascot_width)
        w = (word_lines[i] if i < len(word_lines) else "").ljust(word_width)
        word_color = _gradient_color(i, rows)
        text.append_text(_render_mascot_line(m, compact=compact))
        text.append(_LOGO_GUTTER)
        text.append(w + "\n", style=f"bold {word_color}")
    return text


def _logo_width(*, compact: bool) -> int:
    mascot_lines = _COMPACT_MASCOT_TEMPLATE_LINES if compact else _MASCOT_TEMPLATE_LINES
    word_lines = _COMPACT_LOGO_LINES if compact else _LOGO_LINES
    mascot_width = max((len(l) for l in mascot_lines), default=0)
    word_width = max((len(l) for l in word_lines), default=0)
    return mascot_width + len(_LOGO_GUTTER) + word_width


# ---------------------------------------------------------------------------
# Welcome banner
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BannerInfo:
    version: str
    provider: str
    model: str
    base_url: str | None
    session_id: str
    cwd: str
    tool_count: int = 0
    system_prompt_set: bool = False


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _info_table(info: BannerInfo) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(style=f"{AETHER_DIM}", justify="right", no_wrap=True)
    table.add_column(style=f"{AETHER_TEXT}")

    sess_short = info.session_id[:8] + "…" if len(info.session_id) > 9 else info.session_id

    def row(label_icon: str, label: str, value: Text | str) -> None:
        label_text = Text()
        label_text.append(label_icon + " ", style=AETHER_PRIMARY)
        label_text.append(label, style=AETHER_DIM)
        table.add_row(label_text, value)

    model_text = Text()
    model_text.append(info.model, style=f"bold {AETHER_TEXT}")
    if info.base_url:
        model_text.append("  ")
        model_text.append(_truncate(info.base_url, 48), style=f"dim {AETHER_DIM}")

    row(icon("provider"), "provider", Text(info.provider, style=f"bold {AETHER_ACCENT}"))
    row(icon("model"), "model", model_text)
    row(icon("session"), "session", Text(sess_short, style=AETHER_TEXT))
    row(icon("dot"), "cwd", Text(_truncate(info.cwd, 60), style=f"dim {AETHER_DIM}"))
    if info.tool_count:
        row(icon("tool"), "tools", Text(str(info.tool_count), style=AETHER_TEXT))
    if info.system_prompt_set:
        row(icon("info"), "system", Text("custom system prompt loaded", style=AETHER_TEXT))

    return table


def _hint_line() -> Text:
    txt = Text(justify="center", overflow="ellipsis")
    txt.append("Type your message  ", style=AETHER_DIM)
    txt.append(icon("dot") + "  ", style=AETHER_BORDER)
    txt.append("/help", style=AETHER_ACCENT)
    txt.append(" for commands  ", style=AETHER_DIM)
    txt.append(icon("dot") + "  ", style=AETHER_BORDER)
    txt.append("Esc+Enter", style=AETHER_ACCENT)
    txt.append(" newline  ", style=AETHER_DIM)
    txt.append(icon("dot") + "  ", style=AETHER_BORDER)
    txt.append("Ctrl-D", style=AETHER_ACCENT)
    txt.append(" exit", style=AETHER_DIM)
    return txt


def render_banner(console: Console, info: BannerInfo) -> None:
    """Print the Aether welcome banner: logo + info panel + key hints."""
    term_width = shutil.get_terminal_size((100, 24)).columns
    compact = term_width < (_logo_width(compact=False) + 4)

    logo = _render_logo(compact=compact)
    tagline = Text(
        f"  industrial agent harness  v{info.version}",
        style=f"italic {AETHER_DIM}",
        justify="left",
    )

    body = Group(
        Align.left(logo),
        tagline,
        Text(),
        _info_table(info),
    )

    panel = Panel(
        body,
        border_style=f"{AETHER_PRIMARY_DIM}",
        padding=(1, 2),
        title=Text(f"{icon('logo')} Aether", style=f"bold {AETHER_PRIMARY}"),
        title_align="left",
        subtitle=_hint_line(),
        subtitle_align="center",
    )

    console.print()
    console.print(panel)
    console.print()


def render_compact_session_line(console: Console, info: BannerInfo) -> None:
    """One-line session summary used after `/new` and similar resets."""
    line = Text()
    line.append(icon("logo") + " ", style=f"bold {AETHER_PRIMARY}")
    line.append("Aether", style=f"bold {AETHER_PRIMARY}")
    line.append("  ")
    line.append(icon("provider") + " ", style=AETHER_DIM)
    line.append(info.provider, style=f"bold {AETHER_ACCENT}")
    line.append("  ")
    line.append(icon("model") + " ", style=AETHER_DIM)
    line.append(info.model, style=AETHER_TEXT)
    line.append("  ")
    line.append(icon("session") + " ", style=AETHER_DIM)
    line.append(info.session_id[:8], style=f"dim {AETHER_DIM}")
    console.print(line)
