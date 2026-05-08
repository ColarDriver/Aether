"""Color palette, Rich theme, and glyph table for the Aether CLI.

The whole CLI pulls its colours and icons from this file so that themes
can be swapped without touching individual UI modules.
"""

from __future__ import annotations

import os
import sys

from rich.theme import Theme


# ---------------------------------------------------------------------------
# Palette  (true-color hex; degrades gracefully on 256-colour terminals)
# ---------------------------------------------------------------------------

AETHER_PRIMARY = "#7C5CFF"      # ultraviolet — brand colour
AETHER_PRIMARY_DIM = "#5B3FE0"
AETHER_ACCENT = "#22D3EE"       # cyan — highlights, user input
AETHER_ACCENT_DIM = "#0E7490"
AETHER_TEXT = "#E5E7EB"
AETHER_DIM = "#64748B"
AETHER_BORDER = "#334155"
AETHER_PANEL = "#0F172A"

AETHER_SUCCESS = "#22C55E"
AETHER_WARNING = "#F59E0B"
AETHER_ERROR = "#EF4444"
AETHER_INFO = "#38BDF8"

# Tool-call panel accents (warm copper so they read distinctly from the
# brand purple used for the assistant)
TOOL_ACCENT = "#F97316"
TOOL_DIM = "#9A3412"


# ---------------------------------------------------------------------------
# Rich theme
# ---------------------------------------------------------------------------

THEME = Theme(
    {
        "aether.brand": f"bold {AETHER_PRIMARY}",
        "aether.accent": AETHER_ACCENT,
        "aether.dim": AETHER_DIM,
        "aether.text": AETHER_TEXT,
        "aether.border": AETHER_BORDER,

        "aether.user": f"bold {AETHER_ACCENT}",
        "aether.assistant": f"bold {AETHER_PRIMARY}",
        "aether.system": f"italic {AETHER_DIM}",

        "aether.tool": f"bold {TOOL_ACCENT}",
        "aether.tool.dim": TOOL_DIM,
        "aether.tool.args": f"italic {AETHER_DIM}",
        "aether.tool.result": AETHER_TEXT,
        "aether.tool.error": f"bold {AETHER_ERROR}",

        "aether.success": f"bold {AETHER_SUCCESS}",
        "aether.warning": f"bold {AETHER_WARNING}",
        "aether.error": f"bold {AETHER_ERROR}",
        "aether.info": AETHER_INFO,

        "aether.status": f"italic {AETHER_PRIMARY}",
        "aether.kbd": f"reverse {AETHER_DIM}",
    }
)


# ---------------------------------------------------------------------------
# Icons / glyphs
# ---------------------------------------------------------------------------

# Unicode set — used when stdout is a UTF-8 TTY.
ICONS_UNICODE: dict[str, str] = {
    "logo": "✦",
    "user": "›",
    "assistant": "●",     # filled bullet for assistant turns (Claude-Code style)
    "tool": "⚙",
    "tool_done": "✓",
    "tool_error": "✗",
    "info": "ℹ",
    "warn": "⚠",
    "error": "✗",
    "success": "✓",
    "thinking": "✷",
    "arrow": "→",
    "bullet": "•",
    "dot": "·",
    "session": "◈",
    "model": "◇",
    "provider": "◆",
    "iter": "↻",
    "interrupt": "⏹",
    "spark": "⚡",
}

# ASCII fallback for terminals/locales that can't render the above.
ICONS_ASCII: dict[str, str] = {
    "logo": "*",
    "user": ">",
    "assistant": "*",
    "tool": "T",
    "tool_done": "v",
    "tool_error": "x",
    "info": "i",
    "warn": "!",
    "error": "x",
    "success": "v",
    "thinking": "*",
    "arrow": "->",
    "bullet": "-",
    "dot": ".",
    "session": "#",
    "model": "M",
    "provider": "P",
    "iter": "@",
    "interrupt": "[]",
    "spark": "!",
}


def _supports_unicode() -> bool:
    if os.environ.get("AETHER_ASCII") == "1":
        return False
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


def icon(name: str) -> str:
    """Return the glyph for *name* (Unicode if supported, else ASCII)."""
    table = ICONS_UNICODE if _supports_unicode() else ICONS_ASCII
    return table.get(name, "")


# ---------------------------------------------------------------------------
# Color toggle
# ---------------------------------------------------------------------------

def color_enabled() -> bool:
    """Honour ``NO_COLOR``/``TERM=dumb``/non-tty stdout for colour decisions."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    return True
