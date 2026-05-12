"""Live activity bar shown at the bottom of the REPL during a turn.

Mirrors the Claude-Code transcript style:

    · Deciphering… (12s · ↓ 241 tokens · thought for 4s)

The bar is driven by a 20 Hz ``rich.live.Live`` refresh.  Each refresh
re-evaluates :class:`ActivityBar` which reads :class:`TurnState` and
recomputes elapsed time + an approximate token count.  The state object
itself is mutated by :class:`aether.cli.ui.CLIUI` and the engine
middleware bridge — never touched directly from the bar.

Token estimation: we use ``response_chars // 4`` as an approximation —
matches open-claude-code's heuristic and avoids running ``tiktoken`` on
the hot animation path.  The counter appears as soon as we have at
least :data:`MIN_DISPLAY_TOKENS` accumulated (≈ a handful of words),
mirroring claude-code's ``SpinnerAnimationRow`` which shows the value
the moment streaming produces output — no time gate, just a small
character floor to stop sub-second replies from flashing
``↓ 1 tokens`` for one frame.

Width-aware: when the terminal is narrow we drop fields in order
``thinking → tokens → timer`` so the verb is always visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Union

from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text

from aether.cli.theme import AETHER_DIM, TOOL_ACCENT, TOOL_SHIMMER, icon


# Minimum estimated tokens before we show the ``↓ N tokens`` field.
# claude-code shows the live value the instant any output exists; we
# add a tiny floor (≈ a few words / one short sentence) so a turn that
# replies with "ok\n" doesn't flash ``↓ 1 tokens`` for a single frame
# before disappearing.  Below the floor the suffix simply omits the
# counter — no time gate, so a 3 s turn with 200 chars of streamed
# output already shows ``↓ 50 tokens``.
MIN_DISPLAY_TOKENS = 3

# Legacy time gate retained for backwards compatibility with callers /
# tests that still reference it (e.g. external integrations that
# tweaked the threshold).  No longer consulted by ``ActivityBar`` —
# kept at ``0`` so any code that *does* still apply it as a guard
# becomes a no-op rather than silently keeping the counter hidden.
SHOW_TOKENS_AFTER_MS = 0

# Animated leading glyph — cycled via a wall-clock derived frame index so
# the bar feels alive ("processing…") even when no other field is changing.
# Braille spinner glyphs render at the same width across every monospace
# font and degrade gracefully on terminals that lack the codepoints.
_SPINNER_FRAMES_UNICODE: tuple[str, ...] = (
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
)
_SPINNER_FRAMES_ASCII: tuple[str, ...] = ("|", "/", "-", "\\")
# 10 Hz rotation — slow enough not to be distracting, fast enough to read
# as motion.  Lower than the bar's 20 Hz refresh so we don't double-step.
_SPINNER_TICK_HZ: float = 10.0

# Rough chars→tokens approximation.  Good enough for a live counter where
# precision doesn't matter; running a real tokenizer at 20 Hz on every
# refresh would be wasted CPU.
TOKEN_CHAR_RATIO = 4

# Don't show "thought for Ns" until thinking actually took >0.5 s — the
# first-iteration latency to a local OpenAI-compatible endpoint can be
# under 200 ms and showing "thought for 0s" looks broken.
MIN_THINKING_DISPLAY_MS = 500

# Shimmer animation parameters — match claude-code's ``useShimmerAnimation``
# (`useShimmerAnimation.ts`):  the glimmer index advances one column every
# ``SHIMMER_GLIMMER_SPEED_MS`` ms, then wraps after ``messageWidth + 20``
# steps so the highlight sweeps fully past the verb before re-entering.
# Direction is right→left for thinking/responding modes (matches the React
# branch ``messageWidth + 10 - (cyclePosition % cycleLength)``).
SHIMMER_GLIMMER_SPEED_MS = 200
SHIMMER_TAIL_PADDING = 20

# Mode flags driven by the middleware bridge.  We don't enforce these as
# an Enum on the hot path so a bad value never breaks rendering — the
# bar just shows the verb regardless.
MODE_THINKING = "thinking"
MODE_RESPONDING = "responding"
MODE_TOOL_USE = "tool-use"
MODE_REQUESTING = "requesting"


ThinkingStatus = Union[str, int, None]  # 'thinking' | <elapsed_ms> | None


@dataclass(slots=True)
class TurnState:
    """Mutable per-turn state read by :class:`ActivityBar` on every refresh.

    Lives on :class:`CLIUI` for the duration of a turn.  Reset at
    ``begin_turn``; mutated by the UI surface and the middleware bridge;
    read (never mutated) by ``ActivityBar.__rich_console__``.
    """

    # ----- presentation -----
    verb: str = ""                          # "Thinking", "Reading file", …
    mode: str = MODE_THINKING

    # ----- counters -----
    response_chars: int = 0                 # accumulated streamed chars
    tool_use_count: int = 0                 # tool dispatches this turn
    has_active_tools: bool = False          # currently waiting on a tool

    # ----- thinking state machine -----
    # 'thinking' while the model hasn't produced its first token yet, or
    # the elapsed ms once it did, or None when not applicable.  The bar
    # renders this as ``thinking`` / ``thought for Ns``.
    thinking_status: ThinkingStatus = None
    _thinking_started_at: Optional[float] = None

    # ----- clock -----
    started_at: float = 0.0                 # monotonic seconds at begin_turn
    paused_ms: int = 0
    _pause_started_at: Optional[float] = None

    # -------------------------------------------------------------------
    # Lifecycle hooks called by CLIUI / middleware
    # -------------------------------------------------------------------
    def reset(self) -> None:
        self.verb = ""
        self.mode = MODE_THINKING
        self.response_chars = 0
        self.tool_use_count = 0
        self.has_active_tools = False
        self.thinking_status = None
        self._thinking_started_at = None
        self.started_at = time.monotonic()
        self.paused_ms = 0
        self._pause_started_at = None

    def mark_thinking_start(self) -> None:
        self.thinking_status = "thinking"
        self._thinking_started_at = time.monotonic()

    def mark_first_response_token(self) -> None:
        """Called when the first stream delta arrives.

        Flips the bar from ``thinking`` to ``thought for Ns`` and switches
        the mode to ``responding`` so the verb / glyph follow the action.
        """
        if self._thinking_started_at is not None:
            elapsed_ms = int((time.monotonic() - self._thinking_started_at) * 1000)
            self.thinking_status = elapsed_ms
        else:
            self.thinking_status = None
        self.mode = MODE_RESPONDING

    # -------------------------------------------------------------------
    # Clock math
    # -------------------------------------------------------------------
    def elapsed_ms(self) -> int:
        if self.started_at == 0.0:
            return 0
        if self._pause_started_at is not None:
            wall = self._pause_started_at - self.started_at
        else:
            wall = time.monotonic() - self.started_at
        return max(0, int(wall * 1000) - self.paused_ms)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_duration_ms(ms: int) -> str:
    """``12s`` / ``2m 14s`` / ``1h 03m`` — terse, single-line friendly."""
    seconds = max(0, ms // 1000)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def format_tokens(n: int) -> str:
    """``241`` / ``1.2k`` / ``3.4M`` — keeps the bar narrow."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


# ---------------------------------------------------------------------------
# ActivityBar
# ---------------------------------------------------------------------------

class ActivityBar:
    """Single-line bottom-pinned activity indicator.

    Implements Rich's renderable protocol so a long-lived ``Live`` can
    auto-refresh us at 20 Hz; we recompute everything from
    :attr:`state` each call so the elapsed-time / token counters tick.
    """

    def __init__(self, state: TurnState) -> None:
        self.state = state

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        st = self.state
        if not st.verb:
            # Nothing useful to show yet — emit a blank line so the Live
            # region doesn't collapse and snap upward on the next frame.
            yield Text("")
            return

        cols = options.max_width or 80
        elapsed_ms = st.elapsed_ms()

        # Build candidate suffix fields in priority order.  Each entry
        # is (width, text) — we add as many as fit.
        suffix_fields: list[str] = []

        # Elapsed time appears as soon as we hit ~1 s so the bar feels
        # alive on the first paint without showing a noisy "0s" flicker.
        if elapsed_ms >= 1000:
            suffix_fields.append(format_duration_ms(elapsed_ms))

        # Token counter — appears as soon as ``≥ MIN_DISPLAY_TOKENS``
        # have been streamed.  Mirrors claude-code's
        # ``SpinnerAnimationRow``: no time gate, just a small char
        # floor so trivial replies don't flash ``↓ 1 tokens``.
        # Down-arrow convention matches Claude Code's "tokens flowing
        # out of the model" visual; theme's ``arrow`` icon (→) reads as
        # navigation, not direction-of-flow.
        approx_tokens = max(0, st.response_chars // TOKEN_CHAR_RATIO)
        if approx_tokens >= MIN_DISPLAY_TOKENS:
            suffix_fields.append(f"↓ {format_tokens(approx_tokens)} tokens")

        # Thinking state — either "thinking" while waiting for the first
        # token, or "thought for Ns" once we've started receiving output.
        ts = st.thinking_status
        if ts == "thinking":
            suffix_fields.append("thinking")
        elif isinstance(ts, int) and ts >= MIN_THINKING_DISPLAY_MS:
            suffix_fields.append(f"thought for {max(1, round(ts / 1000))}s")

        # Compose the line.  If width is tight, drop suffix fields from
        # the right (least informative first → preserve the verb).
        line = Text()
        bullet = _spinner_frame()
        line.append(f"{bullet} ", style=f"bold {TOOL_ACCENT}")
        # Shimmer the verb: a soft highlight sweeps right→left through
        # the word, mirroring claude-code's GlimmerMessage / shimmer
        # animation.  Falls back to flat colour when the row is too
        # narrow for the suffix budget calc to keep its math simple.
        line.append_text(_shimmer_verb(st.verb))

        if suffix_fields:
            # Allow up to ~2/3 of the row for the suffix; the verb owns
            # the remainder.  This is a soft budget — Rich will hard-wrap
            # if a single field is wider than the row, which is fine.
            verb_width = len(bullet) + 1 + len(st.verb)
            budget = max(10, cols - verb_width - 3)  # 3 = " ()" padding

            kept: list[str] = []
            running = 0
            for field in suffix_fields:
                addition = len(field) + (3 if kept else 0)  # " · " between
                if running + addition > budget and kept:
                    break
                kept.append(field)
                running += addition

            if kept:
                line.append(" (", style=AETHER_DIM)
                for i, field in enumerate(kept):
                    if i:
                        line.append(" · ", style=AETHER_DIM)
                    line.append(field, style=AETHER_DIM)
                line.append(")", style=AETHER_DIM)

        yield line


def _spinner_frame() -> str:
    """Pick the current animated bullet glyph.

    Driven by wall-clock so consecutive ``ActivityBar`` renders from the
    same Live cycle through frames smoothly.  Falls back to an ASCII
    spinner when the terminal can't render Braille glyphs (matches the
    rest of the icon table's behaviour).
    """
    frames = _SPINNER_FRAMES_UNICODE if (icon("dot") or "") == "·" else _SPINNER_FRAMES_ASCII
    idx = int(time.monotonic() * _SPINNER_TICK_HZ) % len(frames)
    return frames[idx]


def _glimmer_index(message_width: int) -> int:
    """Wall-clock-driven shimmer column position.

    Mirrors claude-code's ``useShimmerAnimation``:

    * One column per :data:`SHIMMER_GLIMMER_SPEED_MS` ms.
    * Cycle length = ``message_width + SHIMMER_TAIL_PADDING`` so the
      highlight sweeps fully past the verb before re-entering.
    * Right→left direction (``message_width + 10 - cyclePosition``) for
      the default (non-requesting) modes — gives the "draining left"
      feel that makes thinking read as deliberate.
    """
    if message_width <= 0:
        return -100  # off-screen — every column reads as "not shimmering"
    cycle_length = message_width + SHIMMER_TAIL_PADDING
    cycle_position = int(time.monotonic() * 1000 / SHIMMER_GLIMMER_SPEED_MS)
    return message_width + 10 - (cycle_position % cycle_length)


def _shimmer_verb(message: str) -> Text:
    """Render *message* with a 3-character-wide highlight at ``_glimmer_index``.

    Mirrors claude-code's ``GlimmerMessage`` — characters at
    ``glimmer_index`` and its two immediate neighbours get the brighter
    :data:`TOOL_SHIMMER` colour; the rest stay on :data:`TOOL_ACCENT`.
    The result is a single :class:`rich.text.Text` so the caller can
    splice it into a longer line.

    Wide-char (CJK) verbs render correctly in terms of *display* — the
    shimmer indexes by code point rather than visual column, which is
    a tiny visual approximation but acceptable for spinner verbs that
    are almost always ASCII ("Thinking", "Pondering", "Channelling", …).
    """
    out = Text()
    if not message:
        return out
    width = len(message)
    glimmer = _glimmer_index(width)
    base = f"bold {TOOL_ACCENT}"
    shim = f"bold {TOOL_SHIMMER}"
    if glimmer < -1 or glimmer > width:  # whole verb off-shimmer this frame
        out.append(message, style=base)
        return out
    for i, ch in enumerate(message):
        out.append(ch, style=shim if abs(i - glimmer) <= 1 else base)
    return out
