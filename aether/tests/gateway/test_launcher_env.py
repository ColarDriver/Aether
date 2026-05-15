"""Unit tests for the TS-TUI launcher env translation.

Lives next to the gateway tests because the launcher only matters as a
bridge between the Python CLI surface and the gateway-backed TS TUI.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest import mock

from aether.cli.launcher import _build_env, _resolve_tui_entry, main as launcher_main
from aether.cli.parser import build_parser


class LauncherEnvTranslation(unittest.TestCase):
    def _parse(self, *flags: str):
        parser = build_parser()
        return parser.parse_args(list(flags))

    def test_forwards_provider_model_temperature(self) -> None:
        args = self._parse(
            "--provider", "claude",
            "--model", "claude-sonnet-4-6",
            "--temperature", "0.3",
            "--max-tokens", "512",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            env = _build_env(args)
        self.assertEqual(env["AETHER_PROVIDER"], "claude")
        self.assertEqual(env["AETHER_MODEL"], "claude-sonnet-4-6")
        self.assertEqual(env["AETHER_TEMPERATURE"], "0.3")
        self.assertEqual(env["AETHER_MAX_TOKENS"], "512")

    def test_forwards_workspace_cwd(self) -> None:
        args = self._parse()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("aether.cli.launcher.os.getcwd", return_value="/work/project"),
        ):
            env = _build_env(args)
        self.assertEqual(env["AETHER_WORKSPACE_CWD"], "/work/project")

    def test_forwards_boolean_flags(self) -> None:
        args = self._parse("--verbose", "--no-banner", "--no-builtin-tools")
        with mock.patch.dict(os.environ, {}, clear=True):
            env = _build_env(args)
        self.assertEqual(env["AETHER_VERBOSE"], "1")
        self.assertEqual(env["AETHER_NO_BANNER"], "1")
        self.assertEqual(env["AETHER_NO_BUILTIN_TOOLS"], "1")

    def test_omits_unset_string_flags(self) -> None:
        args = self._parse()
        with mock.patch.dict(os.environ, {}, clear=True):
            env = _build_env(args)
        self.assertNotIn("AETHER_API_KEY", env)
        self.assertNotIn("AETHER_BASE_URL", env)
        self.assertNotIn("AETHER_SYSTEM", env)
        self.assertNotIn("AETHER_VERBOSE", env)

    def test_resume_pick_translates_to_sentinel(self) -> None:
        args = self._parse("--resume")
        with mock.patch.dict(os.environ, {}, clear=True):
            env = _build_env(args)
        self.assertEqual(env["AETHER_RESUME"], "1")

    def test_resume_id_passed_through(self) -> None:
        args = self._parse("--resume", "ses_abc")
        with mock.patch.dict(os.environ, {}, clear=True):
            env = _build_env(args)
        self.assertEqual(env["AETHER_RESUME"], "ses_abc")

    def test_session_id_uses_session_var(self) -> None:
        args = self._parse("--session", "ses_xyz")
        with mock.patch.dict(os.environ, {}, clear=True):
            env = _build_env(args)
        self.assertEqual(env["AETHER_SESSION_ID"], "ses_xyz")

    def test_api_key_passed_to_gateway_env(self) -> None:
        args = self._parse("--api-key", "sk-test")
        with mock.patch.dict(os.environ, {}, clear=True):
            env = _build_env(args)
        self.assertEqual(env["AETHER_API_KEY"], "sk-test")

    def test_existing_env_wins_over_argparse_default(self) -> None:
        # Caller already set AETHER_MAX_ITERATIONS — don't clobber it with
        # the argparse default of 32.
        args = self._parse()
        with mock.patch.dict(
            os.environ, {"AETHER_MAX_ITERATIONS": "9"}, clear=True
        ):
            env = _build_env(args)
        self.assertEqual(env["AETHER_MAX_ITERATIONS"], "9")

    def test_resolve_tui_entry_returns_something_or_none(self) -> None:
        # Smoke-test: the function should never raise.
        resolved = _resolve_tui_entry()
        # Either we found tui/ + node/npx, or we did not — both are valid.
        if resolved is not None:
            cmd, cwd = resolved
            self.assertIsInstance(cmd, list)
            self.assertTrue(cwd.is_dir())

    def test_non_chat_subcommand_dispatches_locally(self) -> None:
        with (
            mock.patch("aether.cli.launcher.cmd_version") as cmd_version,
            mock.patch("aether.cli.launcher._resolve_tui_entry") as resolve_tui,
        ):
            exit_code = launcher_main(["version"])
        self.assertEqual(exit_code, 0)
        cmd_version.assert_called_once()
        resolve_tui.assert_not_called()

    def test_chat_command_launches_tui_process(self) -> None:
        proc = mock.Mock()
        proc.wait.return_value = 0
        with (
            mock.patch(
                "aether.cli.launcher._resolve_tui_entry",
                return_value=(["node", "dist/entry.js"], Path("/tmp/tui")),
            ),
            mock.patch("aether.cli.launcher.subprocess.Popen", return_value=proc) as popen,
        ):
            exit_code = launcher_main(["chat", "--model", "gpt-5.5"])
        self.assertEqual(exit_code, 0)
        popen.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
