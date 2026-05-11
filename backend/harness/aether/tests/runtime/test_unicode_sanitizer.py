from __future__ import annotations

import unittest

from aether.runtime.unicode_sanitizer import (
    sanitize_provider_credentials_non_ascii,
    sanitize_structure_non_ascii,
    sanitize_structure_surrogates,
    strip_non_ascii,
    strip_surrogates,
)
from aether.tools.base import ToolDescriptor


class FakeClient:
    def __init__(self) -> None:
        self.api_key = "sk-ʋalue"
        self.auth_token = "tok-é"


class FakeProvider:
    def __init__(self) -> None:
        self.api_key = "sk-ʋalue"
        self._api_key = "anthropic-é"
        self._access_token = "access-é"
        self.default_headers = {"Authorization": "Bearer é", "x-note": "ok"}
        self._client = FakeClient()


class UnicodeSanitizerTests(unittest.TestCase):
    def test_strip_surrogates_preserves_valid_unicode(self) -> None:
        self.assertEqual(strip_surrogates("hello\udcff世界"), "hello世界")

    def test_strip_non_ascii_removes_all_non_ascii(self) -> None:
        self.assertEqual(strip_non_ascii("hello 世界 café"), "hello  caf")

    def test_sanitize_structure_surrogates_mutates_nested_values(self) -> None:
        payload = {
            "bad\udcff": ["ok", {"nested": "x\udcffy"}],
            "tuple": ("a\udcff", "b"),
        }

        changed = sanitize_structure_surrogates(payload)

        self.assertTrue(changed)
        self.assertIn("bad", payload)
        self.assertEqual(payload["bad"][1]["nested"], "xy")
        self.assertEqual(payload["tuple"], ("a", "b"))

    def test_sanitize_structure_non_ascii_mutates_tool_descriptor(self) -> None:
        descriptor = ToolDescriptor(
            name="tool",
            description="desc é",
            parameters={
                "properties": {
                    "city": {"description": "São Paulo"},
                }
            },
            required=["cityé"],
        )

        changed = sanitize_structure_non_ascii(descriptor)

        self.assertTrue(changed)
        self.assertEqual(descriptor.description, "desc ")
        self.assertEqual(
            descriptor.parameters["properties"]["city"]["description"],
            "So Paulo",
        )
        self.assertEqual(descriptor.required, ["city"])

    def test_sanitize_provider_credentials_non_ascii(self) -> None:
        provider = FakeProvider()

        changed = sanitize_provider_credentials_non_ascii(provider)

        self.assertTrue(changed)
        self.assertEqual(provider.api_key, "sk-alue")
        self.assertEqual(provider._api_key, "anthropic-")
        self.assertEqual(provider._access_token, "access-")
        self.assertEqual(provider.default_headers["Authorization"], "Bearer ")
        self.assertEqual(provider._client.api_key, "sk-alue")
        self.assertEqual(provider._client.auth_token, "tok-")


if __name__ == "__main__":
    unittest.main()
