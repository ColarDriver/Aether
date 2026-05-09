"""Slash-command dispatch tests — paths-vs-commands disambiguation.

The REPL routes ``/foo`` lines through ``aether.cli.commands.is_slash``
to decide whether to invoke the slash-command dispatcher or forward the
input to the model.  Naïve ``startswith('/')`` was eating absolute file
paths (``/workspace/hermes-agent``) and surfacing a misleading
``Unknown command`` warning instead of letting the LLM see them.

These tests pin the three-gate disambiguation modelled on
open-claude-code's ``looksLikeCommand`` + ``fs.stat`` guard:

  1. starts with '/',
  2. head token is ``[a-zA-Z0-9:_-]+`` (no slashes / dots / CJK / spaces),
  3. ``"/" + head`` does not exist on disk.

If ANY gate fails the line is treated as a regular prompt.  Genuine
typos like ``/halp`` still pass all three gates so the dispatcher
keeps surfacing ``Unknown command`` for them — that's the desired
behaviour, we only want to stop swallowing real paths.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from aether.cli.commands import REGISTRY, is_slash


class IsSlashRegisteredCommandsTests(unittest.TestCase):
    """Every name in REGISTRY must keep being routed to the dispatcher."""

    def test_all_registered_commands_pass_gate(self) -> None:
        for name in REGISTRY:
            with self.subTest(command=name):
                self.assertTrue(is_slash(name), f"{name!r} should route as a command")

    def test_registered_command_with_args_passes(self) -> None:
        self.assertTrue(is_slash("/model gpt-4"))
        self.assertTrue(is_slash("/resume abc1234"))
        self.assertTrue(is_slash("/system you are a helpful assistant"))

    def test_unknown_but_command_shaped_passes(self) -> None:
        # Typos like /halp still route to the dispatcher so it can show
        # the helpful "Unknown command: /halp (try /help)" warning.
        # Pre-condition: the path /halp must not exist on the host.
        if os.path.exists("/halp"):  # pragma: no cover — host pollution
            self.skipTest("/halp exists on this host; can't pin typo behaviour")
        self.assertTrue(is_slash("/halp"))
        self.assertTrue(is_slash("/notacommand foo bar"))


class IsSlashPathsAreNotCommandsTests(unittest.TestCase):
    """Absolute file-path inputs must NOT be treated as slash commands."""

    def test_path_with_subdirectory_segments_rejected(self) -> None:
        # Multi-segment path → head contains '/', fails the regex gate
        # regardless of whether the path exists on disk.
        self.assertFalse(is_slash("/workspace/hermes-agent"))
        self.assertFalse(is_slash("/workspace/hermes-agent 帮我看下这个项目"))
        self.assertFalse(is_slash("/Users/me/file.py"))
        self.assertFalse(is_slash("/etc/passwd"))
        self.assertFalse(is_slash("/var/log/syslog 看下日志"))

    def test_existing_top_level_directory_rejected(self) -> None:
        # ``/tmp`` is virtually guaranteed on POSIX hosts and is the
        # canonical example of a single-segment identifier-shaped path
        # that must NOT be treated as a command.
        if not os.path.isdir("/tmp"):  # pragma: no cover — odd hosts
            self.skipTest("/tmp not present; can't exercise fs gate")
        self.assertFalse(is_slash("/tmp"))
        self.assertFalse(is_slash("/tmp 帮我看下这个目录"))

    def test_existing_directory_via_tempdir(self) -> None:
        # Build a fresh single-segment identifier-shaped path that
        # actually exists, to pin the fs.stat gate independently of the
        # ambient /tmp / /var presence.
        with tempfile.TemporaryDirectory(dir="/", prefix="aether_slash_test_") as path:
            head = os.path.basename(path)
            self.assertTrue(head.replace("_", "").isalnum(), f"unexpected basename: {head!r}")
            self.assertFalse(is_slash(f"/{head}"))
            self.assertFalse(is_slash(f"/{head} look at this"))


class IsSlashShapeRejectionsTests(unittest.TestCase):
    """Inputs that aren't even shaped like ``/cmd`` must be rejected."""

    def test_empty_and_bare_slash_rejected(self) -> None:
        self.assertFalse(is_slash(""))
        self.assertFalse(is_slash("   "))
        self.assertFalse(is_slash("/"))
        self.assertFalse(is_slash(" / "))

    def test_no_leading_slash_rejected(self) -> None:
        self.assertFalse(is_slash("hello"))
        self.assertFalse(is_slash("help me"))
        self.assertFalse(is_slash("./relative/path"))
        self.assertFalse(is_slash("../parent/path"))

    def test_head_with_disallowed_chars_rejected(self) -> None:
        # The regex only accepts [a-zA-Z0-9:_-] so any other punctuation
        # in the head token is treated as prose / path / fragment.
        self.assertFalse(is_slash("/hello.world"))
        self.assertFalse(is_slash("/hello@world"))
        self.assertFalse(is_slash("/中文命令"))
        self.assertFalse(is_slash("/hello!"))

    def test_leading_whitespace_is_trimmed(self) -> None:
        # is_slash strips before checking — preserve that contract.
        self.assertTrue(is_slash("   /help"))
        self.assertFalse(is_slash("   /workspace/hermes-agent"))


class IsSlashFsErrorIsNonFatalTests(unittest.TestCase):
    """``os.path.exists`` raising must not crash the REPL."""

    def test_oserror_falls_through_to_command_path(self) -> None:
        # If exists() somehow raises (broken symlink, permissions),
        # we keep the command-attempt interpretation — the dispatcher
        # will then surface a friendly "Unknown command" instead of
        # bubbling an OSError up to the prompt loop.
        with mock.patch("aether.cli.commands.os.path.exists", side_effect=OSError("boom")):
            self.assertTrue(is_slash("/help"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
