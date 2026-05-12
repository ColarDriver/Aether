"""Sprint 5 / PR 5.9 — cross-session rate guard tests."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, List
from unittest.mock import patch

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.runtime.recovery.fallback_chain import FallbackChain, ProviderSlot
from aether.runtime.recovery.provider_errors import ProviderInvocationError
from aether.runtime.recovery.rate_guard import (
    RateGuard,
    RateGuardCheck,
    RateGuardLock,
    provider_rate_guard_key,
)
from aether.runtime.recovery.strategies import ClassifiedRecoveryStrategy, GenericBackoffStrategy
from aether.tools.base import ToolDescriptor


class _ScriptedProvider(ModelProvider):
    def __init__(
        self,
        name: str,
        script: Iterable[Any],
        *,
        base_url: str | None = None,
        model: str = "test-model",
    ) -> None:
        self.provider_name = name
        self.base_url = base_url
        self.model = model
        self._script = list(script)
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        self.calls += 1
        if not self._script:
            raise RuntimeError(f"{self.provider_name}: script exhausted")
        outcome = self._script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _slot(name: str, provider: ModelProvider) -> ProviderSlot:
    return ProviderSlot(name=name, factory=lambda: provider)


class RateGuardUnitTests(unittest.TestCase):
    def test_lock_write_and_active_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard_dir = Path(tmp)
            provider = _ScriptedProvider(
                "openai-compatible",
                [],
                base_url="https://api.example.test/v1",
            )
            guard = RateGuard(guard_dir)

            lock = guard.block(
                provider,
                until_unix=time.time() + 60,
                source_session_id="session-a",
            )
            check = guard.check(provider)

            self.assertIsNotNone(lock)
            self.assertTrue(check.checked)
            self.assertTrue(check.blocked)
            self.assertIsNotNone(check.lock)
            self.assertEqual(check.lock.source_session_id, "session-a")

            key = provider_rate_guard_key(provider, guard_dir)
            self.assertTrue(key.path.exists())
            self.assertNotIn("api.example.test", key.filename)
            self.assertIn(key.namespace_hash, key.filename)

    def test_expired_lock_is_ignored_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard_dir = Path(tmp)
            provider = _ScriptedProvider("openai-compatible", [], base_url="https://old.test/v1")
            guard = RateGuard(guard_dir)

            guard.block(provider, until_unix=time.time() - 1, source_session_id="expired")
            key = provider_rate_guard_key(provider, guard_dir)

            check = guard.check(provider)

            self.assertFalse(check.blocked)
            self.assertFalse(key.path.exists())

    def test_clear_removes_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard_dir = Path(tmp)
            provider = _ScriptedProvider("openai-compatible", [], base_url="https://clear.test/v1")
            guard = RateGuard(guard_dir)
            guard.block(provider, until_unix=time.time() + 60, source_session_id="session-a")
            key = provider_rate_guard_key(provider, guard_dir)
            self.assertTrue(key.path.exists())

            self.assertTrue(guard.clear(provider))

            self.assertFalse(key.path.exists())
            self.assertFalse(guard.check(provider).blocked)

    def test_concurrent_writes_leave_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard_dir = Path(tmp)
            provider = _ScriptedProvider("openai-compatible", [], base_url="https://race.test/v1")
            guard = RateGuard(guard_dir)

            def write_lock(index: int) -> bool:
                return guard.block(
                    provider,
                    until_unix=time.time() + 60 + index,
                    source_session_id=f"session-{index}",
                ) is not None

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(write_lock, range(24)))

            self.assertTrue(all(results))
            key = provider_rate_guard_key(provider, guard_dir)
            with key.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            lock = RateGuardLock.from_mapping(payload)
            self.assertIsNotNone(lock)
            self.assertEqual(lock.provider, "openai-compatible")
            self.assertEqual(lock.base_url_hash, key.namespace_hash)

    def test_filesystem_failure_degrades_to_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard_file = Path(tmp) / "not-a-dir"
            guard_file.write_text("x", encoding="utf-8")
            provider = _ScriptedProvider("openai-compatible", [], base_url="https://fs.test/v1")
            guard = RateGuard(guard_file)

            self.assertIsNone(
                guard.block(provider, until_unix=time.time() + 60, source_session_id="session-a")
            )
            self.assertFalse(guard.check(provider).blocked)
            self.assertFalse(guard.clear(provider))


class RateGuardEngineIntegrationTests(unittest.TestCase):
    def test_active_lock_uses_fallback_without_calling_current_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard_dir = Path(tmp)
            primary = _ScriptedProvider(
                "openai-compatible",
                [NormalizedResponse(content="must-not-run")],
                base_url="https://primary.test/v1",
            )
            fallback = _ScriptedProvider(
                "openai-compatible",
                [NormalizedResponse(content="from fallback")],
                base_url="https://fallback.test/v1",
            )
            RateGuard(guard_dir).block(
                primary,
                until_unix=time.time() + 60,
                source_session_id="other-session",
            )
            chain = FallbackChain([
                _slot("primary", primary),
                _slot("fallback", fallback),
            ])
            engine = AgentEngine(
                primary,
                fallback_chain=chain,
                config=EngineConfig(
                    use_builtin_tools=False,
                    fallback_chain_enabled=True,
                    rate_guard_dir=guard_dir,
                    max_iterations=2,
                ),
            )

            result = engine.run_turn(EngineRequest(session_id="session-b", user_message="hi"))

            self.assertEqual(result.status, EngineStatus.COMPLETED)
            self.assertEqual(result.final_response, "from fallback")
            self.assertEqual(primary.calls, 0)
            self.assertEqual(fallback.calls, 1)
            rate_guard = result.metadata["turn"]["rate_guard"]
            self.assertTrue(rate_guard["checked"])
            self.assertTrue(rate_guard["blocked"])
            self.assertTrue(rate_guard["fallback_activated"])
            self.assertEqual(rate_guard["fallback_slot"], "fallback")
            self.assertEqual(rate_guard["source_session_id"], "other-session")

    def test_active_lock_without_fallback_blocks_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard_dir = Path(tmp)
            provider = _ScriptedProvider(
                "openai-compatible",
                [NormalizedResponse(content="must-not-run")],
                base_url="https://solo.test/v1",
            )
            until = time.time() + 60
            RateGuard(guard_dir).block(
                provider,
                until_unix=until,
                source_session_id="other-session",
            )
            engine = AgentEngine(
                provider,
                config=EngineConfig(
                    use_builtin_tools=False,
                    rate_guard_dir=guard_dir,
                    max_iterations=2,
                ),
            )

            result = engine.run_turn(EngineRequest(session_id="session-c", user_message="hi"))

            self.assertEqual(result.status, EngineStatus.FAILED)
            self.assertEqual(result.exit_reason, ExitReason.RATE_LIMITED)
            self.assertIn("rate guard blocked provider", result.error or "")
            self.assertEqual(provider.calls, 0)
            rate_guard = result.metadata["turn"]["rate_guard"]
            self.assertTrue(rate_guard["checked"])
            self.assertTrue(rate_guard["blocked"])
            self.assertFalse(rate_guard["fallback_activated"])
            self.assertAlmostEqual(rate_guard["until_unix"], until, delta=1.0)

    def test_rate_limit_error_writes_lock(self) -> None:
        cases = [
            (
                "http-429",
                ProviderInvocationError(
                    status_code=429,
                    retry_after_seconds=45,
                    body_summary="too many requests",
                ),
                GenericBackoffStrategy(max_attempts=1, base_wait_seconds=0.0),
            ),
            (
                "classified-rate-limit",
                ProviderInvocationError(
                    status_code=400,
                    retry_after_seconds=45,
                    body_summary="rate limit exceeded for this workspace",
                ),
                ClassifiedRecoveryStrategy(max_attempts=1, base_wait_seconds=0.0),
            ),
        ]

        for name, error, strategy in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                guard_dir = Path(tmp)
                provider = _ScriptedProvider(
                    "openai-compatible",
                    [error],
                    base_url=f"https://{name}.test/v1",
                )
                engine = AgentEngine(
                    provider,
                    recovery_strategy=strategy,
                    config=EngineConfig(
                        use_builtin_tools=False,
                        rate_guard_dir=guard_dir,
                        max_iterations=2,
                    ),
                )

                result = engine.run_turn(EngineRequest(session_id=f"session-{name}", user_message="hi"))
                check = RateGuard(guard_dir).check(provider)

                self.assertEqual(result.status, EngineStatus.FAILED)
                self.assertTrue(check.blocked)
                self.assertIsNotNone(check.until_unix)
                self.assertGreater(check.until_unix or 0.0, time.time() + 30)
                rate_guard = result.metadata["turn"]["rate_guard"]
                self.assertTrue(rate_guard["write_attempted"])
                self.assertTrue(rate_guard["lock_written"])

    def test_successful_provider_response_clears_rate_guard_key(self) -> None:
        class _SpyRateGuard:
            clear_calls: list[ModelProvider] = []

            def __init__(self, guard_dir: Path | None = None) -> None:
                self.guard_dir = guard_dir

            def check(self, provider: ModelProvider) -> RateGuardCheck:
                return RateGuardCheck(
                    checked=True,
                    blocked=False,
                    key=provider_rate_guard_key(provider, self.guard_dir),
                )

            def clear(self, provider: ModelProvider) -> bool:
                self.clear_calls.append(provider)
                return True

        provider = _ScriptedProvider(
            "openai-compatible",
            [NormalizedResponse(content="ok")],
            base_url="https://success.test/v1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            _SpyRateGuard.clear_calls = []
            with patch("aether.agents.core.agent.RateGuard", _SpyRateGuard):
                engine = AgentEngine(
                    provider,
                    config=EngineConfig(
                        use_builtin_tools=False,
                        rate_guard_dir=Path(tmp),
                        max_iterations=2,
                    ),
                )
                result = engine.run_turn(EngineRequest(session_id="session-ok", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(_SpyRateGuard.clear_calls, [provider])
        self.assertTrue(result.metadata["turn"]["rate_guard"]["cleared"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
