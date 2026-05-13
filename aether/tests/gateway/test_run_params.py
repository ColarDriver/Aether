"""Unit tests for ``agent.run`` parameter parsing.

Exercises the ``_parse_run_params`` helper directly so we get fast
coverage of the validation rules without spinning up an engine.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from aether.cli.sessions import SessionRecord
from aether.gateway.handlers.agent_methods import (
    _build_provider_for_record,
    _parse_run_params,
)
from aether.gateway.protocol import GatewayError


class ParseRunParams(unittest.TestCase):
    def _base(self, **extra):
        return {"session_id": "ses_x", "user_message": "hello", **extra}

    def test_accepts_minimum_required(self) -> None:
        parsed = _parse_run_params(self._base())
        self.assertEqual(parsed["session_id"], "ses_x")
        self.assertEqual(parsed["user_message"], "hello")
        self.assertIsNone(parsed["max_iterations"])
        self.assertIsNone(parsed["temperature"])
        self.assertIsNone(parsed["max_tokens"])
        self.assertIsNone(parsed["disable_builtin_tools"])
        self.assertIsNone(parsed["system_override"])

    def test_accepts_all_optional_params(self) -> None:
        parsed = _parse_run_params(
            self._base(
                max_iterations=8,
                temperature=0.2,
                max_tokens=512,
                disable_builtin_tools=True,
                system_override="be terse",
            )
        )
        self.assertEqual(parsed["max_iterations"], 8)
        self.assertEqual(parsed["temperature"], 0.2)
        self.assertEqual(parsed["max_tokens"], 512)
        self.assertIs(parsed["disable_builtin_tools"], True)
        self.assertEqual(parsed["system_override"], "be terse")

    def test_rejects_non_positive_max_iterations(self) -> None:
        with self.assertRaises(GatewayError):
            _parse_run_params(self._base(max_iterations=0))
        with self.assertRaises(GatewayError):
            _parse_run_params(self._base(max_iterations=-1))
        with self.assertRaises(GatewayError):
            _parse_run_params(self._base(max_iterations="four"))

    def test_rejects_non_positive_max_tokens(self) -> None:
        with self.assertRaises(GatewayError):
            _parse_run_params(self._base(max_tokens=0))
        with self.assertRaises(GatewayError):
            _parse_run_params(self._base(max_tokens=-100))
        with self.assertRaises(GatewayError):
            _parse_run_params(self._base(max_tokens="big"))

    def test_rejects_non_numeric_temperature(self) -> None:
        with self.assertRaises(GatewayError):
            _parse_run_params(self._base(temperature="hot"))

    def test_rejects_non_bool_disable_builtin_tools(self) -> None:
        with self.assertRaises(GatewayError):
            _parse_run_params(self._base(disable_builtin_tools="yes"))
        with self.assertRaises(GatewayError):
            _parse_run_params(self._base(disable_builtin_tools=1))

    def test_rejects_non_string_system_override(self) -> None:
        with self.assertRaises(GatewayError):
            _parse_run_params(self._base(system_override=123))


class BuildProviderForRecord(unittest.TestCase):
    def test_forwards_aether_api_key_to_provider_factory(self) -> None:
        record = SessionRecord.new(
            session_id="ses_x",
            provider="openai",
            model="gpt-4o",
            base_url="https://example.test/v1",
        )
        with (
            mock.patch.dict(os.environ, {"AETHER_API_KEY": "sk-test"}, clear=True),
            mock.patch("aether.cli.providers.build_provider") as build_provider,
        ):
            _build_provider_for_record(record)
        build_provider.assert_called_once_with(
            "openai",
            model="gpt-4o",
            api_key="sk-test",
            base_url="https://example.test/v1",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
