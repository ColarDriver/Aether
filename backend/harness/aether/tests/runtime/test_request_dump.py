from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aether.runtime.observability.request_dump import dump_api_request_debug, redact_for_dump


class RequestDumpTests(unittest.TestCase):
    def test_redacts_sensitive_keys_recursively(self) -> None:
        redacted = redact_for_dump(
            {
                "api_key": "sk-secret",
                "headers": {
                    "Authorization": "Bearer token",
                    "cookie": "session=value",
                },
                "nested": [{"token": "abc"}, {"password": "pw"}],
            }
        )

        self.assertEqual(redacted["api_key"], "<redacted>")
        self.assertEqual(redacted["headers"]["Authorization"], "<redacted>")
        self.assertEqual(redacted["headers"]["cookie"], "<redacted>")
        self.assertEqual(redacted["nested"][0]["token"], "<redacted>")
        self.assertEqual(redacted["nested"][1]["password"], "<redacted>")

    def test_truncates_long_content_values(self) -> None:
        redacted = redact_for_dump(
            {"messages": [{"role": "user", "content": "x" * 20}]},
            max_content_chars=8,
        )

        content = redacted["messages"][0]["content"]
        self.assertEqual(content["preview"], "x" * 8)
        self.assertEqual(content["original_length"], 20)
        self.assertTrue(content["_truncated"])

    def test_dump_api_request_debug_writes_redacted_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = dump_api_request_debug(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "headers": {"Authorization": "Bearer secret"},
                    "api_key": "sk-secret",
                },
                model="model-a",
                provider="provider-a",
                base_url="https://example.test",
                reason="non_retryable_client_error",
                error=RuntimeError("boom"),
                dump_dir=Path(tmp),
                session_id="sess-1",
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "model-a")
            self.assertEqual(payload["provider"], "provider-a")
            self.assertEqual(payload["base_url"], "https://example.test")
            self.assertEqual(payload["reason"], "non_retryable_client_error")
            self.assertIn("boom", payload["error"])
            self.assertEqual(payload["kwargs"]["headers"]["Authorization"], "<redacted>")
            self.assertEqual(payload["kwargs"]["api_key"], "<redacted>")


if __name__ == "__main__":
    unittest.main()
