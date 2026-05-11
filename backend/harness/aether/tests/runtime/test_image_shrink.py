from __future__ import annotations

import base64
import copy
import unittest
from io import BytesIO
from typing import Any

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime import image_shrink as image_shrink_module
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.runtime.error_classifier import FailoverReason, classify_api_error
from aether.runtime.image_shrink import (
    DEFAULT_MAX_BASE64_BYTES,
    DEFAULT_TARGET_BASE64_BYTES,
    shrink_image_parts_in_messages,
)
from aether.runtime.provider_errors import ProviderInvocationError
from aether.runtime.recovery import AttemptState, ClassifiedRecoveryStrategy
from aether.tools.base import ToolDescriptor

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    Image = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False


def _large_bmp_base64(size: int = 1300) -> str:
    if Image is None:  # pragma: no cover - guarded by skipUnless
        raise unittest.SkipTest("Pillow required to build image fixture")
    image = Image.new("RGB", (size, size), (23, 97, 181))
    out = BytesIO()
    image.save(out, format="BMP")
    return base64.b64encode(out.getvalue()).decode("ascii")


def _small_png_data_url() -> str:
    if Image is None:  # pragma: no cover - guarded by skipUnless
        raise unittest.SkipTest("Pillow required to build image fixture")
    image = Image.new("RGB", (16, 16), (255, 0, 0))
    out = BytesIO()
    image.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _data_url_base64(url: str) -> str:
    return url.split(",", 1)[1]


class _CapturingImageProvider(ModelProvider):
    def __init__(
        self,
        *,
        failures_before_success: int,
        body_summary: str | None = None,
    ) -> None:
        self.failures_before_success = failures_before_success
        self.body_summary = body_summary or "image_too_large: base64 image exceeds 5 MB"
        self.calls: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del tools, config, context, stream_callback, stream_silent_callback
        self.calls.append(copy.deepcopy(messages))
        if len(self.calls) <= self.failures_before_success:
            raise ProviderInvocationError(
                status_code=400,
                body_summary=self.body_summary,
            )
        return NormalizedResponse(content="ok")


class ImageShrinkUnitTests(unittest.TestCase):
    @unittest.skipUnless(_PIL_AVAILABLE, "Pillow required for image shrink test")
    def test_three_supported_image_shapes_are_shrunk(self) -> None:
        large = _large_bmp_base64()
        self.assertGreater(len(large), DEFAULT_MAX_BASE64_BYTES)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/bmp;base64,{large}"},
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/bmp;base64,{large}",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/bmp",
                            "data": large,
                        },
                    },
                ],
            }
        ]
        original = copy.deepcopy(messages)

        shrunk, stats = shrink_image_parts_in_messages(messages)

        self.assertTrue(stats.changed)
        self.assertEqual(stats.changed_count, 3)
        self.assertEqual(messages, original)
        content = shrunk[0]["content"]
        chat_url = content[0]["image_url"]["url"]
        response_url = content[1]["image_url"]
        anthropic_source = content[2]["source"]
        self.assertTrue(chat_url.startswith("data:image/jpeg;base64,"))
        self.assertTrue(response_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(anthropic_source["media_type"], "image/jpeg")
        self.assertLessEqual(len(_data_url_base64(chat_url)), DEFAULT_TARGET_BASE64_BYTES)
        self.assertLessEqual(len(_data_url_base64(response_url)), DEFAULT_TARGET_BASE64_BYTES)
        self.assertLessEqual(len(anthropic_source["data"]), DEFAULT_TARGET_BASE64_BYTES)
        self.assertGreater(stats.original_base64_bytes_max, DEFAULT_MAX_BASE64_BYTES)
        self.assertLessEqual(stats.shrunk_base64_bytes_max, DEFAULT_TARGET_BASE64_BYTES)

    @unittest.skipUnless(_PIL_AVAILABLE, "Pillow required for image shrink test")
    def test_small_image_and_http_url_are_unchanged(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _small_png_data_url()}},
                    {"type": "input_image", "image_url": "https://example.com/image.png"},
                ],
            }
        ]
        original = copy.deepcopy(messages)

        shrunk, stats = shrink_image_parts_in_messages(messages)

        self.assertFalse(stats.changed)
        self.assertEqual(shrunk, original)
        self.assertEqual(messages, original)

    def test_pillow_unavailable_degrades_without_raising(self) -> None:
        data = base64.b64encode(b"x" * (DEFAULT_MAX_BASE64_BYTES + 10)).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}
                ],
            }
        ]
        original_loader = image_shrink_module._load_pillow_image
        image_shrink_module._load_pillow_image = lambda: None
        try:
            shrunk, stats = shrink_image_parts_in_messages(messages)
        finally:
            image_shrink_module._load_pillow_image = original_loader

        self.assertFalse(stats.changed)
        self.assertEqual(shrunk, messages)
        self.assertIn("pillow_unavailable", stats.error_reasons)


class ImageTooLargeClassifierTests(unittest.TestCase):
    def test_classifier_identifies_image_too_large(self) -> None:
        classified = classify_api_error(
            ProviderInvocationError(
                status_code=400,
                body_summary="image_too_large: base64 image exceeds 5 MB limit",
            )
        )

        self.assertEqual(classified.reason, FailoverReason.image_too_large)
        self.assertTrue(classified.retryable)
        self.assertFalse(classified.should_fallback)

    def test_generic_413_stays_payload_too_large(self) -> None:
        classified = classify_api_error(
            ProviderInvocationError(status_code=413, body_summary="payload too large")
        )

        self.assertEqual(classified.reason, FailoverReason.payload_too_large)

    def test_recovery_strategy_requests_image_shrink_retry(self) -> None:
        decision = ClassifiedRecoveryStrategy(base_wait_seconds=0.0).decide(
            ProviderInvocationError(
                status_code=400,
                body_summary="base64 image exceeds 5 MB",
            ),
            AttemptState(attempt=1),
            TurnContext(session_id="image", iteration=1, metadata={}),
        )

        self.assertTrue(decision.retry)
        self.assertEqual(decision.wait_seconds, 0.0)
        self.assertEqual(decision.classified_reason, FailoverReason.image_too_large.value)
        self.assertEqual(decision.reason, "image-too-large:shrink-and-retry")


class ImageShrinkEngineTests(unittest.TestCase):
    @unittest.skipUnless(_PIL_AVAILABLE, "Pillow required for image shrink test")
    def test_provider_image_too_large_error_shrinks_and_retries_once(self) -> None:
        large = _large_bmp_base64()
        provider = _CapturingImageProvider(failures_before_success=1)
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False, max_iterations=2),
            recovery_strategy=ClassifiedRecoveryStrategy(
                max_attempts=3,
                base_wait_seconds=0.0,
            ),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="image-shrink-ok",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/bmp;base64,{large}"},
                            }
                        ],
                    }
                ],
            )
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "ok")
        self.assertEqual(len(provider.calls), 2)
        first_url = provider.calls[0][0]["content"][0]["image_url"]["url"]
        second_url = provider.calls[1][0]["content"][0]["image_url"]["url"]
        self.assertGreater(len(_data_url_base64(first_url)), DEFAULT_MAX_BASE64_BYTES)
        self.assertLessEqual(len(_data_url_base64(second_url)), DEFAULT_TARGET_BASE64_BYTES)
        self.assertTrue(result.metadata["turn"]["image_shrink_applied"])
        self.assertEqual(result.metadata["turn"]["image_shrink_count"], 1)
        self.assertEqual(
            result.metadata["recovery"]["cascade_log"],
            ["image_shrink(count=1)"],
        )

    def test_image_too_large_without_shrink_does_not_retry_forever(self) -> None:
        provider = _CapturingImageProvider(failures_before_success=10)
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False, max_iterations=2),
            recovery_strategy=ClassifiedRecoveryStrategy(
                max_attempts=3,
                base_wait_seconds=0.0,
            ),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="image-shrink-unavailable",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.png"},
                            }
                        ],
                    }
                ],
            )
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue(result.metadata["turn"]["image_shrink_attempted"])
        self.assertFalse(result.metadata["turn"]["image_shrink"]["changed"])
        decision = result.metadata["turn"]["recovery_decisions"][0]
        self.assertFalse(decision["retry"])
        self.assertEqual(decision["classified_reason"], FailoverReason.image_too_large.value)


if __name__ == "__main__":
    unittest.main()
