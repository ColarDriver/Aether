from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineResult,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
)


class _FacadeEngine(AgentEngine):
    def __init__(self) -> None:
        super().__init__(ScriptedProvider([NormalizedResponse(content="unused")]))
        self.requests: list[EngineRequest] = []

    def run_loop(self, request: EngineRequest) -> EngineResult:
        self.requests.append(request)
        return EngineResult(
            session_id=request.session_id,
            status=EngineStatus.COMPLETED,
            exit_reason=ExitReason.TEXT_RESPONSE,
            messages=[],
            iterations=0,
            final_response="ok",
        )


class AgentEngineFacadeCompatTests(unittest.TestCase):
    def test_run_turn_delegates_to_run_loop(self) -> None:
        engine = _FacadeEngine()
        request = EngineRequest(session_id="facade-turn")

        result = engine.run_turn(request)

        self.assertEqual(result.final_response, "ok")
        self.assertEqual(engine.requests, [request])

    def test_resume_delegates_to_run_loop(self) -> None:
        engine = _FacadeEngine()
        request = EngineRequest(session_id="facade-resume")

        result = engine.resume(request)

        self.assertEqual(result.final_response, "ok")
        self.assertEqual(engine.requests, [request])


if __name__ == "__main__":
    unittest.main()

