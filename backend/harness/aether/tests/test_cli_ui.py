from __future__ import annotations

import time
import unittest

from rich.console import Console

from aether.cli.activity import (
    MIN_DISPLAY_TOKENS,
    SHOW_TOKENS_AFTER_MS,
    TOKEN_CHAR_RATIO,
    ActivityBar,
    TurnState,
    format_duration_ms,
    format_tokens,
)
from aether.cli.ui import (
    CLIUI,
    _StreamState,
    _TurnSurface,
    _extract_all_command_fences,
    _extract_inline_tool_intent,
    _extract_trailing_command_fence,
    _has_unclosed_command_fence,
    _inline_activity_from_text,
    _looks_like_intended_tool_use,
    extract_thinking_blocks,
    strip_all_command_fences,
    strip_tool_blocks,
    strip_trailing_command_fence,
)


def _raw_inline_tool_output() -> str:
    body = (
        '{"name":"exec_command","arguments":{"cmd":"find /workspace/Aether -maxdepth 2 -type f"}}'
    )
    return (
        "让我看看你的项目结构。\n"
        "<function_calls>\n"
        f"{body}\n"
        f"{body}\n"
        "</function_calls>"
    )


def _raw_bracket_tool_output() -> str:
    return (
        "让我看看你的项目结构。\n\n"
        "[Tool: list]\n\n"
        "[Tool: read]\n\n"
        "[Tool: glob]\n\n"
        "这是一个 CLI agent harness 项目。"
    )


class CLIUITests(unittest.TestCase):
    def test_render_assistant_block_hides_preview_by_default_for_inline_tool_tags(self) -> None:
        console = Console(record=True, width=100)
        ui = CLIUI(console)

        ui.render_assistant_block(_raw_inline_tool_output())
        # Sprint 1.5 / P0-9: phantom warning is now deferred from
        # render time to ``end_turn`` so synthesis can suppress it.
        # ``end_turn`` is what materialises the diagnostic.
        ui.end_turn(status="COMPLETED", exit_reason="text_response", iterations=1)

        output = console.export_text()
        self.assertIn("让我看看你的项目结构。", output)
        self.assertIn("model emitted inline tool tags", output)
        self.assertNotIn("raw model output preview", output)
        self.assertNotIn("<function_calls>", output)

    def test_render_assistant_block_shows_preview_in_verbose_mode_for_inline_tool_tags(self) -> None:
        console = Console(record=True, width=100)
        ui = CLIUI(console, verbose=True)

        ui.render_assistant_block(_raw_inline_tool_output())
        ui.end_turn(status="COMPLETED", exit_reason="text_response", iterations=1)

        output = console.export_text()
        self.assertIn("raw model output preview", output)
        self.assertIn("<function_calls>", output)
        self.assertIn("exec_command", output)

    def test_end_stream_hides_preview_by_default_when_streamed_output_was_only_inline_tool_tags(self) -> None:
        console = Console(record=True, width=100)
        ui = CLIUI(console)
        raw = _raw_inline_tool_output()

        ui._stream = _StreamState(
            active=True,
            stripped_chars=len(raw),
            tool_calls_at_start=0,
        )
        ui._stream_buffer = [raw]
        ui.end_stream()
        ui.end_turn(status="COMPLETED", exit_reason="text_response", iterations=1)

        output = console.export_text()
        self.assertIn("model emitted inline tool tags", output)
        self.assertNotIn("raw model output preview", output)
        self.assertNotIn("<function_calls>", output)

    def test_strip_tool_blocks_removes_bracket_style_tool_lines(self) -> None:
        cleaned = strip_tool_blocks(_raw_bracket_tool_output())

        self.assertNotIn("[Tool: list]", cleaned)
        self.assertNotIn("[Tool: read]", cleaned)
        self.assertNotIn("[Tool: glob]", cleaned)
        self.assertIn("让我看看你的项目结构。", cleaned)
        self.assertIn("这是一个 CLI agent harness 项目。", cleaned)

    def test_render_assistant_block_hides_bracket_style_tool_lines_without_preview(self) -> None:
        console = Console(record=True, width=100)
        ui = CLIUI(console)

        ui.render_assistant_block(_raw_bracket_tool_output())

        output = console.export_text()
        self.assertIn("让我看看你的项目结构。", output)
        self.assertIn("这是一个 CLI agent harness 项目。", output)
        self.assertNotIn("[Tool: list]", output)
        self.assertNotIn("[Tool: read]", output)
        self.assertNotIn("[Tool: glob]", output)
        self.assertNotIn("raw model output preview", output)

    def test_inline_activity_from_bracket_style_tool_lines_uses_last_tool(self) -> None:
        activity = _inline_activity_from_text(_raw_bracket_tool_output())
        self.assertEqual(activity, "Finding files…")

    def test_inline_activity_from_xml_invoke_uses_name_and_parameter(self) -> None:
        raw = (
            '<function_calls>\n'
            '<invoke name="mcp__filesystem__read_file">\n'
            '<parameter name="path">README.md</parameter>\n'
            '</invoke>\n'
            '</function_calls>'
        )
        activity = _inline_activity_from_text(raw)
        self.assertEqual(activity, "Reading file…  README.md")

    def test_strip_tool_blocks_hides_partial_inline_xml_tag(self) -> None:
        cleaned = strip_tool_blocks("先看看。<funct")
        self.assertEqual(cleaned, "先看看。")

    def test_render_stream_surface_places_activity_above_assistant_text(self) -> None:
        console = Console(record=True, width=100)
        ui = CLIUI(console)

        console.print(ui._render_stream_surface("这是最终回答。", activity="Reading file…"))

        output = console.export_text()
        self.assertIn("Reading file…", output)
        self.assertIn("这是最终回答。", output)

    def test_turn_surface_tail_crops_long_streaming_buffer(self) -> None:
        """Regression: live region must only render the last ~N lines.

        When the streamed buffer grows past the terminal viewport,
        rendering the full markdown each frame caused Rich's Live
        region to leak "frames" into scrollback (cursor clear-and-redraw
        only reaches still-visible rows).  The fix in
        ``_TurnSurface.__rich_console__`` tail-crops the buffer to
        ``options.height - 1`` lines so the rendered surface stays
        bounded; the full content is still flushed to scrollback by
        ``end_stream``.

        This test instantiates the surface with a small terminal and
        asserts that only the tail content is rendered while the stream
        is active.
        """
        console = Console(record=True, width=80, height=8)
        ui = CLIUI(console)
        # Mark the stream as active and feed it 50 distinct lines.  Each
        # line is short and well-formed markdown so we can verify that
        # only the last few survive into the rendered output.
        ui._stream = _StreamState(active=True)
        ui._stream_buffer = ["\n".join(f"line-{i:02d}" for i in range(50))]

        console.print(_TurnSurface(ui))
        output = console.export_text()

        # The very last lines must be present (latest content takes
        # priority — that's what the user is actively reading).
        self.assertIn("line-49", output)
        self.assertIn("line-48", output)
        # And the very first lines must NOT be present — those were
        # cropped off the top so the live region fits the viewport.
        self.assertNotIn("line-00", output)
        self.assertNotIn("line-01", output)

    def test_turn_surface_renders_full_buffer_when_under_viewport(self) -> None:
        """No tail-crop when the buffer fits — the user sees everything."""
        console = Console(record=True, width=80, height=20)
        ui = CLIUI(console)
        ui._stream = _StreamState(active=True)
        ui._stream_buffer = ["line-A\nline-B\nline-C"]

        console.print(_TurnSurface(ui))
        output = console.export_text()

        self.assertIn("line-A", output)
        self.assertIn("line-B", output)
        self.assertIn("line-C", output)


class SpacingPolicyTests(unittest.TestCase):
    """Verify the centralised single-blank-line spacer between heavy blocks."""

    def test_user_echo_followed_by_assistant_block_has_one_blank_line(self) -> None:
        console = Console(record=True, width=80, force_terminal=False)
        ui = CLIUI(console)

        ui.render_input_echo("帮我列一下当前目录")
        ui.render_assistant_block("当前目录里有很多文件。")

        output = console.export_text()
        # Expect: <user line>\n\n<assistant line>\n
        # i.e. exactly one blank line between user and assistant blocks.
        # Count consecutive blank lines around the assistant block.
        lines = [ln for ln in output.split("\n")]
        # Find the user line and the next assistant line.
        user_idx = next(
            (i for i, ln in enumerate(lines) if "帮我列一下当前目录" in ln),
            None,
        )
        assistant_idx = next(
            (i for i, ln in enumerate(lines) if "当前目录里有很多文件" in ln),
            None,
        )
        self.assertIsNotNone(user_idx)
        self.assertIsNotNone(assistant_idx)
        assert user_idx is not None and assistant_idx is not None
        self.assertGreater(assistant_idx, user_idx)
        # Lines between them should be exactly one blank row.
        between = lines[user_idx + 1 : assistant_idx]
        self.assertEqual(
            [ln.strip() for ln in between],
            [""],
            f"expected one blank line between user/assistant, got {between!r}",
        )

    def test_two_assistant_blocks_in_a_row_keep_one_blank_separator(self) -> None:
        console = Console(record=True, width=80, force_terminal=False)
        ui = CLIUI(console)

        ui.render_assistant_block("first paragraph")
        ui.render_assistant_block("second paragraph")

        output = console.export_text()
        # Both bodies must be present.
        self.assertIn("first paragraph", output)
        self.assertIn("second paragraph", output)
        # They must be separated by at least one blank.
        first_idx = output.index("first paragraph")
        second_idx = output.index("second paragraph")
        between = output[first_idx:second_idx]
        # Split into non-empty rows; both blocks should produce
        # individual lines, separated by an empty row.
        self.assertIn("\n\n", between)


class ReasoningChannelTests(unittest.TestCase):
    """Ensure reasoning deltas surface as the dim ``thinking: …`` sub-line."""

    def test_extract_thinking_blocks_returns_concatenated_text(self) -> None:
        raw = (
            "前言。\n"
            "<thinking>第一段思考</thinking>\n"
            "中间。\n"
            "<thinking>第二段思考</thinking>\n"
            "结尾。"
        )
        excerpt = extract_thinking_blocks(raw)
        self.assertIn("第一段思考", excerpt)
        self.assertIn("第二段思考", excerpt)

    def test_extract_thinking_blocks_returns_empty_for_plain_text(self) -> None:
        self.assertEqual(extract_thinking_blocks("just plain prose"), "")
        self.assertEqual(extract_thinking_blocks(""), "")

    def test_stream_reasoning_delta_populates_excerpt(self) -> None:
        console = Console(record=True, width=80, force_terminal=False)
        ui = CLIUI(console)
        ui.begin_stream()
        ui.stream_reasoning_delta("planning the next step")
        ui.stream_reasoning_delta(" then verifying")

        self.assertGreater(ui._stream.reasoning_chars, 0)
        self.assertIn("planning the next step", ui._stream.reasoning_excerpt)
        self.assertIn("then verifying", ui._stream.reasoning_excerpt)

    def test_turn_surface_renders_reasoning_excerpt_when_no_visible_body(self) -> None:
        """When the stream has only reasoning content, the bar still shows it."""
        console = Console(record=True, width=80, height=10, force_terminal=False)
        ui = CLIUI(console)
        ui.begin_stream()
        ui._turn_state.verb = "Pondering"
        ui._stream.reasoning_excerpt = "weighing the options"

        console.print(_TurnSurface(ui))
        output = console.export_text()
        self.assertIn("Pondering", output)
        self.assertIn("thinking:", output)
        self.assertIn("weighing the options", output)


class ToolGroupIntegrationTests(unittest.TestCase):
    """Integration tests for the active-group renderer inside _TurnSurface."""

    def test_in_flight_tool_group_renders_headline_and_hint(self) -> None:
        console = Console(record=True, width=100, height=10, force_terminal=False)
        ui = CLIUI(console)
        ui._turn_state.verb = "Searching"
        # Simulate the middleware path: start a tool call, but don't
        # finish it yet.  The active group should render in _TurnSurface.
        ui.tool_groups.start_call("Bash", {"command": "ls -la"})

        console.print(_TurnSurface(ui))
        output = console.export_text()
        self.assertIn("Running", output)
        self.assertIn("1 command", output)
        self.assertIn("⎿", output)
        self.assertIn("$ ls -la", output)

    def test_resolved_group_flushes_past_tense_to_scrollback(self) -> None:
        console = Console(record=True, width=100, force_terminal=False)
        ui = CLIUI(console)
        ui.tool_groups.start_call("Read", {"file_path": "README.md"})
        ui.tool_groups.finish_call()
        ui.tool_groups.flush_active()

        output = console.export_text()
        self.assertIn("Read", output)
        self.assertIn("1 file", output)
        # Past tense must not carry the trailing ellipsis used by active.
        self.assertNotIn("Reading 1 file…", output)

    def test_tool_group_renders_two_categories_in_one_headline(self) -> None:
        console = Console(record=True, width=120, height=10, force_terminal=False)
        ui = CLIUI(console)
        ui._turn_state.verb = "Searching"
        ui.tool_groups.start_call("Grep", {"pattern": "foo"})
        ui.tool_groups.start_call("Read", {"file_path": "x"})

        console.print(_TurnSurface(ui))
        output = console.export_text()
        # Both categories on one line.
        self.assertIn("Searching for 1 pattern", output)
        self.assertIn("reading 1 file", output)


class ActivityBarTests(unittest.TestCase):
    """Bottom-of-screen activity bar — verb / elapsed / tokens / thinking."""

    @staticmethod
    def _render(state: TurnState, *, width: int = 100) -> str:
        console = Console(record=True, width=width)
        console.print(ActivityBar(state))
        return console.export_text()

    def test_activity_bar_blank_when_no_verb(self) -> None:
        state = TurnState()
        state.started_at = time.monotonic()
        out = self._render(state)
        # Without a verb the bar collapses to a blank line.
        self.assertNotIn("·", out)

    def test_activity_bar_shows_verb_and_elapsed_after_one_second(self) -> None:
        state = TurnState()
        state.verb = "Deciphering"
        # Pretend the turn has been running for ~12 s.
        state.started_at = time.monotonic() - 12.0
        out = self._render(state)
        self.assertIn("Deciphering", out)
        self.assertIn("12s", out)

    def test_activity_bar_hides_tokens_below_char_floor(self) -> None:
        # Tokens are gated by ``MIN_DISPLAY_TOKENS`` (chars / 4) — below
        # the floor we omit the field entirely so trivial replies
        # ("ok\n", an emoji) don't flash a "↓ 1 tokens" frame.
        state = TurnState()
        state.verb = "Thinking"
        state.started_at = time.monotonic() - 5.0
        state.response_chars = TOKEN_CHAR_RATIO * (MIN_DISPLAY_TOKENS - 1)
        out = self._render(state)
        self.assertNotIn("tokens", out)

    def test_activity_bar_shows_tokens_immediately_above_floor(self) -> None:
        # claude-code parity: the moment streaming produces enough
        # output to clear the floor, the counter appears — no time
        # gate, even on a 2-second turn.  This test is the regression
        # guard for the legacy 30s ``SHOW_TOKENS_AFTER_MS`` lockout.
        state = TurnState()
        state.verb = "Thinking"
        state.started_at = time.monotonic() - 2.0  # well under any time gate
        state.response_chars = TOKEN_CHAR_RATIO * 69  # 69 tokens, like the screenshot
        out = self._render(state)
        self.assertIn("69 tokens", out)
        self.assertIn("↓", out)

    def test_activity_bar_shows_large_token_count_after_long_turn(self) -> None:
        state = TurnState()
        state.verb = "Thinking"
        state.started_at = time.monotonic() - 35.0
        state.response_chars = TOKEN_CHAR_RATIO * 241
        out = self._render(state)
        self.assertIn("241 tokens", out)

    def test_legacy_show_tokens_after_ms_constant_is_zero(self) -> None:
        # Backwards compat: keep the symbol importable but ensure it's
        # a no-op so any external integration that still consults it
        # (e.g. as a min-elapsed gate) does not silently re-suppress
        # the counter.
        self.assertEqual(SHOW_TOKENS_AFTER_MS, 0)

    def test_activity_bar_renders_thinking_state(self) -> None:
        state = TurnState()
        state.verb = "Pondering"
        state.started_at = time.monotonic() - 2.0
        state.thinking_status = "thinking"
        out = self._render(state)
        self.assertIn("thinking", out)

    def test_activity_bar_renders_thought_for_state(self) -> None:
        state = TurnState()
        state.verb = "Reading file"
        state.started_at = time.monotonic() - 5.0
        state.thinking_status = 4200  # 4.2 s elapsed thinking
        out = self._render(state)
        self.assertIn("thought for 4s", out)


class FormatHelpersTests(unittest.TestCase):
    def test_format_duration_seconds(self) -> None:
        self.assertEqual(format_duration_ms(12_000), "12s")

    def test_format_duration_minutes(self) -> None:
        self.assertEqual(format_duration_ms(134_000), "2m 14s")

    def test_format_duration_hours(self) -> None:
        self.assertEqual(format_duration_ms(3_780_000), "1h 03m")

    def test_format_tokens_under_1k(self) -> None:
        self.assertEqual(format_tokens(241), "241")

    def test_format_tokens_above_1k(self) -> None:
        self.assertEqual(format_tokens(1234), "1.2k")


class IdleTurnHeuristicTests(unittest.TestCase):
    """Tighter conditions on the "no tools were dispatched" warning."""

    def test_short_greeting_does_not_count_as_intent(self) -> None:
        self.assertFalse(_looks_like_intended_tool_use("你好！有什么我可以帮你的吗？"))
        self.assertFalse(_looks_like_intended_tool_use("Hi! How can I help?"))

    def test_fenced_shell_block_counts_as_intent(self) -> None:
        text = "Let me check the file:\n```bash\nls /tmp\n```"
        self.assertTrue(_looks_like_intended_tool_use(text))

    def test_shell_prompt_line_counts_as_intent(self) -> None:
        text = "我来运行一下:\n$ ls /tmp"
        self.assertTrue(_looks_like_intended_tool_use(text))

    def test_chinese_imperative_counts_as_intent(self) -> None:
        self.assertTrue(_looks_like_intended_tool_use("我来运行 ls 看看"))

    def test_english_imperative_counts_as_intent(self) -> None:
        self.assertTrue(_looks_like_intended_tool_use("I'll run ls to inspect."))


class InlineToolIntentTests(unittest.TestCase):
    """Pulling (name, args) out of stripped inline XML / JSON tool tags."""

    def test_extracts_invoke_with_parameters(self) -> None:
        raw = (
            '<function_calls>\n'
            '<invoke name="read_file">\n'
            '<parameter name="path">README.md</parameter>\n'
            '</invoke>\n'
            '</function_calls>'
        )
        intent = _extract_inline_tool_intent(raw)
        self.assertIsNotNone(intent)
        assert intent is not None  # type-narrow for mypy
        self.assertEqual(intent.name, "read_file")
        self.assertEqual(intent.args.get("path"), "README.md")

    def test_extracts_tool_call_json(self) -> None:
        raw = (
            '<tool_call>{"name":"exec_command",'
            '"arguments":{"cmd":"ls /tmp"}}</tool_call>'
        )
        intent = _extract_inline_tool_intent(raw)
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.name, "exec_command")
        self.assertEqual(intent.args.get("cmd"), "ls /tmp")

    def test_returns_none_for_plain_prose(self) -> None:
        self.assertIsNone(_extract_inline_tool_intent("hello world"))

    def test_extracts_function_equals_inline_tag(self) -> None:
        raw = (
            "我来看看这个项目的结构。"
            "<function=execute_command> ls -la /workspace/Aether/"
        )
        intent = _extract_inline_tool_intent(raw)
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.name, "execute_command")
        self.assertEqual(intent.args.get("command"), "ls -la /workspace/Aether/")


class FunctionEqualsTagTests(unittest.TestCase):
    """The non-standard ``<function=NAME> <body>`` syntax.

    Some Kimi-class models emit this when they fail to populate the
    structured ``tool_calls`` field — the tag has no closing pair, so
    each ``<function=…>`` consumes everything up to the next
    ``<function=`` or end-of-text.  We pin the strip + intent
    extraction here so the diagnostic surfaces the attempted command
    instead of leaving a wall of inline tags in the response body.
    """

    def test_strip_tool_blocks_removes_single_function_equals_block(self) -> None:
        raw = (
            "我来看看这个项目的结构。"
            "<function=execute_command> ls -la /workspace/Aether/"
        )
        cleaned = strip_tool_blocks(raw)
        self.assertNotIn("<function=", cleaned)
        self.assertNotIn("ls -la", cleaned)
        self.assertIn("我来看看这个项目的结构。", cleaned)

    def test_strip_tool_blocks_removes_multiple_function_equals_blocks(self) -> None:
        raw = (
            "我来看看这个项目的结构。\n"
            "<function=execute_command> ls -la /workspace/Aether/  "
            "<function=execute_command> cat /workspace/Aether/README.md  "
            "<function=execute_command> find /workspace/Aether -type f"
        )
        cleaned = strip_tool_blocks(raw)
        self.assertNotIn("<function=", cleaned)
        self.assertNotIn("ls -la", cleaned)
        self.assertNotIn("cat", cleaned)
        self.assertNotIn("find", cleaned)
        self.assertIn("我来看看这个项目的结构。", cleaned)

    def test_strip_tool_blocks_drops_partial_trailing_function_equals_open_tag(self) -> None:
        cleaned = strip_tool_blocks("先看看。<function=")
        self.assertEqual(cleaned, "先看看。")

    def test_inline_activity_falls_back_to_function_equals_command(self) -> None:
        raw = (
            "我来看看这个项目的结构。"
            "<function=execute_command> ls -la /workspace/Aether/"
        )
        activity = _inline_activity_from_text(raw)
        self.assertIn("ls -la /workspace/Aether/", activity)

    def test_end_stream_strips_function_equals_block_and_surfaces_attempted(self) -> None:
        console = Console(record=True, width=140)
        ui = CLIUI(console)
        raw = (
            "我来看看这个项目的结构和相关文档，帮你分析一下它是做什么的。让我先查看项目"
            "根目录和关键文件。"
            "<function=execute_command> ls -la /workspace/Aether/  "
            "<function=execute_command> cat /workspace/Aether/README.md  "
            "<function=execute_command> find /workspace/Aether -type f"
        )

        ui.begin_turn()
        ui._stream = _StreamState(active=True, tool_calls_at_start=0)
        ui._stream_buffer = [raw]
        ui.stats.streamed_chars = len(raw)
        ui.end_stream()
        ui.end_turn(status="COMPLETED", exit_reason="natural_completion", iterations=1)

        output = console.export_text()
        self.assertNotIn("<function=", output)
        self.assertNotIn("ls -la", output[: output.find("attempted")] if "attempted" in output else output)
        self.assertIn("我来看看这个项目的结构", output)
        self.assertIn("attempted:", output)


class TrailingCommandFenceTests(unittest.TestCase):
    """Phantom-bash-command intent extraction.

    When the model writes ``\u0060\u0060\u0060bash <cmd>\u0060\u0060\u0060`` (or, more
    commonly, leaves the closing fence off because it bailed at the
    moment it tried to invoke a tool), we want to:

    1. Pull the parsed command body out for the diagnostic.
    2. Strip the trailing fence from the visible body so the user
       doesn't see a dangling code block.
    3. Surface the parsed command on a ``\u2514 attempted: $ <cmd>`` line.

    These tests pin all three behaviours so the regression the user
    reported ("我们这个项目是做什么的 → \u0060\u0060\u0060bash ls -la 就断了") doesn't
    come back the next time we touch the CLI rendering pipeline.
    """

    def test_extracts_trailing_unclosed_bash_fence(self) -> None:
        text = (
            "我来帮你看看这个项目是做什么的。让我先浏览一下项目结构和关键文件。"
            "```bash ls -la /workspace/Aether"
        )
        cmd = _extract_trailing_command_fence(text)
        self.assertEqual(cmd, "ls -la /workspace/Aether")

    def test_extracts_trailing_closed_bash_fence(self) -> None:
        text = (
            "Let me check the project structure for you.\n"
            "```bash\nls -la /workspace/Aether\n```"
        )
        cmd = _extract_trailing_command_fence(text)
        self.assertEqual(cmd, "ls -la /workspace/Aether")

    def test_returns_none_when_fence_is_followed_by_substantial_prose(self) -> None:
        text = (
            "Here's how to list files:\n"
            "```bash\nls -la\n```\n"
            "This will print everything in the directory, including "
            "permissions, owner, group, size, and modified time.  Useful "
            "for inspecting what's in a folder before changing anything."
        )
        self.assertIsNone(_extract_trailing_command_fence(text))

    def test_returns_none_for_text_without_fence(self) -> None:
        self.assertIsNone(_extract_trailing_command_fence("just prose"))

    def test_strip_trailing_command_fence_removes_dangling_block(self) -> None:
        text = (
            "我来帮你看看这个项目是做什么的。让我先浏览一下项目结构和关键文件。"
            "```bash ls -la /workspace/Aether"
        )
        cleaned = strip_trailing_command_fence(text)
        self.assertNotIn("```", cleaned)
        self.assertNotIn("ls -la", cleaned)
        self.assertIn("我来帮你看看这个项目", cleaned)

    def test_strip_trailing_command_fence_preserves_embedded_examples(self) -> None:
        text = (
            "Here's how to list files:\n"
            "```bash\nls -la\n```\n"
            "This will print everything in the directory, including "
            "permissions, owner, group, size, and modified time.  Useful "
            "for inspecting what's in a folder before changing anything."
        )
        self.assertEqual(strip_trailing_command_fence(text), text)

    def test_unclosed_fence_detected_when_model_truncated_mid_command(self) -> None:
        self.assertTrue(
            _has_unclosed_command_fence("我来运行 ```bash ls -la /workspace")
        )

    def test_closed_fence_not_detected_as_unclosed(self) -> None:
        self.assertFalse(
            _has_unclosed_command_fence("Try ```bash\nls\n``` to list files.")
        )

    def test_end_stream_strips_dangling_bash_fence_and_surfaces_attempted_command(self) -> None:
        console = Console(record=True, width=120)
        ui = CLIUI(console)
        raw = (
            "我来帮你看看这个项目是做什么的。让我先浏览一下项目结构和关键文件。"
            "```bash ls -la /workspace/Aether"
        )

        ui.begin_turn()
        ui._stream = _StreamState(active=True, tool_calls_at_start=0)
        ui._stream_buffer = [raw]
        ui.stats.streamed_chars = len(raw)
        ui.end_stream()
        ui.end_turn(status="COMPLETED", exit_reason="natural_completion", iterations=1)

        output = console.export_text()
        self.assertIn("我来帮你看看这个项目", output)
        self.assertNotIn("```bash", output)
        self.assertIn("no tools were dispatched this turn", output)
        self.assertIn("attempted:", output)
        self.assertIn("ls -la /workspace/Aether", output)

    def test_extract_all_command_fences_returns_each_block_in_source_order(self) -> None:
        text = (
            "你好！让我先探索一下项目结构。\n\n"
            "我先看看项目的整体结构和关键文件：```bash cd /workspace/Aether && ls -la "
            "find /workspace/Aether -maxdepth 2 -type f -name '*.md' | head -30\n\n"
            "实际上让我用更系统的方式查看：```bash\n"
            "ls -la /workspace/Aether/\n"
            "cat /workspace/Aether/README.md 2>/dev/null || echo \"No README\"\n"
            "```\n\n"
            "让我执行这些命令来获取项目信息。"
        )
        commands = _extract_all_command_fences(text)
        self.assertEqual(len(commands), 2)
        self.assertIn("cd /workspace/Aether && ls -la", commands[0])
        self.assertIn("ls -la /workspace/Aether/", commands[1])
        self.assertIn("cat /workspace/Aether/README.md", commands[1])

    def test_strip_all_command_fences_removes_every_block_and_dangling_colon(self) -> None:
        text = (
            "你好！让我先探索一下项目结构。\n\n"
            "我先看看项目的整体结构：```bash cd /workspace/Aether && ls -la\n\n"
            "实际上让我用更系统的方式查看：```bash\n"
            "ls -la /workspace/Aether/\n"
            "```\n\n"
            "让我执行这些命令。"
        )
        cleaned = strip_all_command_fences(text)
        self.assertNotIn("```", cleaned)
        self.assertNotIn("ls -la", cleaned)
        self.assertNotIn("cd /workspace/Aether", cleaned)
        self.assertNotIn("整体结构：", cleaned)
        self.assertIn("你好", cleaned)
        self.assertIn("让我执行这些命令", cleaned)

    def test_end_stream_strips_multiple_bash_fences_and_lists_each_attempt(self) -> None:
        console = Console(record=True, width=140)
        ui = CLIUI(console)
        raw = (
            "你好！让我先探索一下项目结构，了解这个项目是做什么的。\n\n"
            "我先看看项目的整体结构和关键文件：```bash cd /workspace/Aether && ls -la "
            "find /workspace/Aether -maxdepth 2 -type f | head -30\n\n"
            "实际上让我用更系统的方式查看：```bash\n"
            "ls -la /workspace/Aether/\n"
            "cat /workspace/Aether/README.md 2>/dev/null || echo 'no readme'\n"
            "```\n\n"
            "让我执行这些命令来获取项目信息。"
        )

        ui.begin_turn()
        ui._stream = _StreamState(active=True, tool_calls_at_start=0)
        ui._stream_buffer = [raw]
        ui.stats.streamed_chars = len(raw)
        ui.end_stream()
        ui.end_turn(status="COMPLETED", exit_reason="natural_completion", iterations=1)

        output = console.export_text()
        self.assertNotIn("```", output)
        self.assertNotIn("cd /workspace/Aether && ls -la find", output[: output.find("attempted")])
        self.assertIn("attempted:", output)
        self.assertIn("$ cd /workspace/Aether && ls -la", output)
        self.assertIn("$ ls -la /workspace/Aether/", output)
        self.assertIn("你好", output)
        self.assertIn("让我执行这些命令", output)

    def test_render_assistant_block_strips_dangling_bash_fence_in_fallback_path(self) -> None:
        console = Console(record=True, width=120)
        ui = CLIUI(console)
        raw = (
            "我来帮你看看这个项目是做什么的。让我先浏览一下项目结构和关键文件。"
            "```bash ls -la /workspace/Aether"
        )

        ui.begin_turn()
        ui.stats.streamed_chars = len(raw)
        ui.render_assistant_block(raw)
        ui.end_turn(status="COMPLETED", exit_reason="natural_completion", iterations=1)

        output = console.export_text()
        self.assertNotIn("```bash", output)
        self.assertIn("attempted:", output)
        self.assertIn("ls -la /workspace/Aether", output)


if __name__ == "__main__":
    unittest.main()
