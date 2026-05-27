from __future__ import annotations

import threading

import pytest

from aether.cli.sessions import SessionRecord, load_session, save_session
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.recovery.provider_errors import ProviderInvocationError
from aether.runtime.session.session_state import clear_cwd, set_cwd
from aether.services.common import (
    ServiceConflictError,
    ServiceNotFoundError,
    ServiceValidationError,
)
from aether.services.runs import (
    AgentRunCancelRequest,
    AgentRunRequest,
    AgentRunService,
    AssistantDelta,
    ReasoningDelta,
    RunEvent,
    RunFailed,
    RunFinished,
    RunStarted,
    SilentProgress,
    ToolFinished,
    ToolStarted,
)
from aether.services.runs.builder import RunDependencyBuilder
from aether.services.runs.handles import RunHandle
from aether.services.sessions import SessionService
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _Sink:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


def _service(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    provider: ModelProvider,
    *,
    registry: ToolRegistry | None = None,
) -> tuple[AgentRunService, SessionService, _Sink]:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    sessions = SessionService(session_dir=tmp_path / "sessions")
    sink = _Sink()
    builder = RunDependencyBuilder(
        provider_factory=lambda record: provider,
        config_factory=lambda options: _test_config(options.max_iterations),
        tool_registry_factory=lambda: registry,
    )
    return AgentRunService(session_service=sessions, builder=builder), sessions, sink


def _save_session(tmp_path, *, session_id: str = "ses_run", provider: str = "mock", model: str = "mock-model"):
    record = SessionRecord.new(session_id=session_id, provider=provider, model=model)
    save_session(record, base=tmp_path / "sessions")
    return record


def test_agent_run_service_runs_provider_emits_events_and_persists_result(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_session(tmp_path)
    service, _sessions, sink = _service(tmp_path, monkeypatch, _StreamingProvider(["hel", "lo"], reasoning="why"))

    result = service.start(
        AgentRunRequest(session_id="ses_run", user_message="say hello", run_id="run-1"),
        sink=sink,
    )

    assert result.final_text == "hello"
    assert result.exit_reason == "done"
    assert result.usage["prompt_tokens"] == 4
    assert isinstance(sink.events[0], RunStarted)
    assert isinstance(sink.events[-1], RunFinished)
    assert [event.text for event in sink.events if isinstance(event, AssistantDelta)] == ["hel", "lo"]
    assert [event.sequence for event in sink.events if isinstance(event, AssistantDelta)] == [0, 1]
    assert [event.text for event in sink.events if isinstance(event, ReasoningDelta)] == ["why"]
    saved = load_session("ses_run", base=tmp_path / "sessions")
    assert saved is not None
    assert len(saved.messages) >= 2
    assert service.final_result("run-1") == result
    assert service.status("ses_run") is not None


def test_run_dependency_builder_sets_engine_request_cwd_from_request_or_session(tmp_path) -> None:
    request_root = tmp_path / "request-root"
    request_root.mkdir()
    tracked_root = tmp_path / "tracked-root"
    tracked_root.mkdir()
    record = SessionRecord.new(session_id="cwd_ses", provider="mock", model="mock-model")
    builder = RunDependencyBuilder()

    explicit = builder.build_engine_request(
        record=record,
        request=AgentRunRequest(session_id="cwd_ses", user_message="hello", cwd=str(request_root)),
        run_id="run-cwd",
        handle=RunHandle(session_id="cwd_ses", run_id="run-cwd"),
        stream_callback=lambda _text: None,
        stream_silent_callback=lambda _text: None,
        loop_state_callback=lambda _state: None,
    )
    assert explicit.cwd == str(request_root)

    set_cwd("cwd_ses", tracked_root)
    try:
        tracked = builder.build_engine_request(
            record=record,
            request=AgentRunRequest(session_id="cwd_ses", user_message="hello"),
            run_id="run-cwd-tracked",
            handle=RunHandle(session_id="cwd_ses", run_id="run-cwd-tracked"),
            stream_callback=lambda _text: None,
            stream_silent_callback=lambda _text: None,
            loop_state_callback=lambda _state: None,
        )
        assert tracked.cwd == str(tracked_root.resolve())
    finally:
        clear_cwd("cwd_ses")


def test_agent_run_service_persists_run_observability_metadata(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_session(tmp_path)
    service, _sessions, sink = _service(tmp_path, monkeypatch, _StreamingProvider(["done"]))

    result = service.start(
        AgentRunRequest(session_id="ses_run", user_message="track context", run_id="run-meta"),
        sink=sink,
    )

    assert result.usage["total_tokens"] == 6
    saved = load_session("ses_run", base=tmp_path / "sessions")
    assert saved is not None
    assistant_messages = [message for message in saved.messages if message.get("role") == "assistant"]
    assert assistant_messages
    metadata = assistant_messages[-1]["metadata"]
    assert metadata["usage"]["total_tokens"] == 6
    assert metadata["context_engine"]["name"] == "default"
    assert metadata["context_engine"]["compression"]["status"] == "skipped"
    assert "turn" not in metadata


def test_agent_run_service_promotes_and_persists_workspace_checkpoint_metadata(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_session(tmp_path)
    service, _sessions, sink = _service(tmp_path, monkeypatch, _StreamingProvider(["done"]))

    result = service.start(
        AgentRunRequest(
            session_id="ses_run",
            user_message="track checkpoint",
            run_id="run-checkpoint",
            metadata={
                "workspace_checkpoint": {
                    "checkpoint_id": "20260527010101-abcdef12",
                    "label": "Before web run",
                    "files": [{"path": "app.py", "exists": True}],
                }
            },
        ),
        sink=sink,
    )

    assert result.metadata["workspace_checkpoint"]["checkpoint_id"] == "20260527010101-abcdef12"
    saved = load_session("ses_run", base=tmp_path / "sessions")
    assert saved is not None
    assistant_messages = [message for message in saved.messages if message.get("role") == "assistant"]
    assert assistant_messages[-1]["metadata"]["workspace_checkpoint"]["checkpoint_id"] == "20260527010101-abcdef12"


def test_agent_run_service_persists_user_display_attachments(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_session(tmp_path)
    service, _sessions, sink = _service(tmp_path, monkeypatch, _StreamingProvider(["ok"]))

    service.start(
        AgentRunRequest(
            session_id="ses_run",
            user_message="inspect this",
            attachments=[
                {
                    "type": "image",
                    "name": "plot.png",
                    "mimeType": "image/png",
                    "data": "data:image/png;base64,abc",
                }
            ],
            run_id="run-attachments",
        ),
        sink=sink,
    )

    saved = load_session("ses_run", base=tmp_path / "sessions")
    assert saved is not None
    user_message = next(message for message in saved.messages if message["role"] == "user")
    assert user_message["metadata"]["displayAttachments"] == [
        {
            "type": "image",
            "name": "plot.png",
            "mimeType": "image/png",
            "data": "data:image/png;base64,abc",
        }
    ]


def test_agent_run_service_injects_text_attachment_context_for_provider(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_session(tmp_path)
    provider = _RecordingProvider()
    service, _sessions, _sink = _service(tmp_path, monkeypatch, provider)

    service.start(
        AgentRunRequest(
            session_id="ses_run",
            user_message="inspect this",
            attachments=[
                {
                    "type": "text",
                    "name": "app.ts",
                    "path": "src/app.ts",
                    "note": "workspace reference",
                    "_llm_language": "typescript",
                    "_llm_content": "export const app = 1",
                }
            ],
            run_id="run-attachment-context",
        )
    )

    assert provider.messages_by_call
    user_message = [message for message in provider.messages_by_call[0] if message["role"] == "user"][-1]
    assert "inspect this" in user_message["content"]
    assert "<attached_context>" in user_message["content"]
    assert "```typescript" in user_message["content"]
    assert "export const app = 1" in user_message["content"]

    saved = load_session("ses_run", base=tmp_path / "sessions")
    assert saved is not None
    persisted_user = next(message for message in saved.messages if message["role"] == "user")
    assert persisted_user["content"] == "inspect this"
    assert persisted_user["metadata"]["displayAttachments"] == [
        {
            "type": "text",
            "name": "app.ts",
            "path": "src/app.ts",
            "note": "workspace reference",
        }
    ]


def test_agent_run_service_injects_native_image_attachment_parts_for_provider(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_session(tmp_path)
    provider = _RecordingProvider()
    service, _sessions, _sink = _service(tmp_path, monkeypatch, provider)

    service.start(
        AgentRunRequest(
            session_id="ses_run",
            user_message="describe this",
            attachments=[
                {
                    "type": "image",
                    "name": "plot.png",
                    "mimeType": "image/png",
                    "data": "data:image/png;base64,abc",
                }
            ],
            run_id="run-native-image-attachment",
        )
    )

    assert provider.messages_by_call
    user_message = [message for message in provider.messages_by_call[0] if message["role"] == "user"][-1]
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0]["type"] == "text"
    assert "describe this" in user_message["content"][0]["text"]
    assert any(
        part.get("type") == "image_url"
        and part.get("image_url", {}).get("url") == "data:image/png;base64,abc"
        for part in user_message["content"]
    )

    saved = load_session("ses_run", base=tmp_path / "sessions")
    assert saved is not None
    persisted_user = next(message for message in saved.messages if message["role"] == "user")
    assert persisted_user["content"] == "describe this"


def test_agent_run_service_missing_and_invalid_session_errors(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service, _sessions, sink = _service(tmp_path, monkeypatch, _StreamingProvider(["x"]))

    with pytest.raises(ServiceNotFoundError):
        service.start(AgentRunRequest(session_id="missing", user_message="x"), sink=sink)

    _save_session(tmp_path, session_id="bad", provider="", model="model")
    with pytest.raises(ServiceValidationError):
        service.start(AgentRunRequest(session_id="bad", user_message="x"), sink=sink)


def test_agent_run_service_tool_and_silent_events(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_session(tmp_path)
    registry = ToolRegistry()
    registry.register(_EchoTool())
    service, _sessions, sink = _service(tmp_path, monkeypatch, _SilentToolProvider(), registry=registry)

    result = service.start(
        AgentRunRequest(session_id="ses_run", user_message="use echo", run_id="run-tool"),
        sink=sink,
    )

    assert result.exit_reason == "done"
    assert [event.chars for event in sink.events if isinstance(event, SilentProgress)] == [8, 9]
    tool_starts = [event for event in sink.events if isinstance(event, ToolStarted)]
    tool_finishes = [event for event in sink.events if isinstance(event, ToolFinished)]
    assert tool_starts[0].tool_call_id == "tc_silent"
    assert tool_finishes[0].tool_call_id == "tc_silent"
    assert tool_finishes[0].content == "silent"


def test_agent_run_service_cancel_interrupts_active_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_session(tmp_path)
    provider = _CancellableProvider()
    service, _sessions, sink = _service(tmp_path, monkeypatch, provider)
    result_box: dict[str, object] = {}

    thread = threading.Thread(
        target=lambda: result_box.setdefault(
            "result",
            service.start(
                AgentRunRequest(session_id="ses_run", user_message="stream", run_id="run-cancel"),
                sink=sink,
            ),
        )
    )
    thread.start()
    assert provider.first_delta.wait(timeout=2.0)
    assert service.cancel(AgentRunCancelRequest(session_id="ses_run", reason="test-cancel")) is True
    provider.allow_next_delta.set()
    thread.join(timeout=3.0)

    result = result_box["result"]
    assert getattr(result, "exit_reason") == "cancelled"
    assert getattr(result, "final_text") == "a"
    assert service.cancel(AgentRunCancelRequest(session_id="ses_run")) is False
    assert service.final_result("run-cancel") == result


def test_agent_run_service_active_guard_and_prompt_disconnect_failure(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_session(tmp_path)
    provider = _BlockingProvider()
    service, _sessions, sink = _service(tmp_path, monkeypatch, provider)

    thread = threading.Thread(
        target=lambda: service.start(
            AgentRunRequest(session_id="ses_run", user_message="first", run_id="run-block"),
            sink=sink,
        )
    )
    thread.start()
    assert provider.started.wait(timeout=2.0)
    with pytest.raises(ServiceConflictError):
        service.start(AgentRunRequest(session_id="ses_run", user_message="second"), sink=sink)
    provider.release.set()
    thread.join(timeout=3.0)

    service2, _sessions2, sink2 = _service(tmp_path, monkeypatch, _FailingProvider(RuntimeError("peer disconnected")))
    result = service2.start(
        AgentRunRequest(session_id="ses_run", user_message="fail", run_id="run-fail"),
        sink=sink2,
    )
    assert result.exit_reason == "error"
    assert isinstance(sink2.events[-1], RunFailed)


def test_agent_run_service_exposes_provider_error_metadata(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_session(tmp_path)
    exc = ProviderInvocationError(
        status_code=401,
        body_summary="invalid api key",
        metadata={"url": "https://provider.test/chat/completions", "method": "POST"},
    )
    service, _sessions, sink = _service(tmp_path, monkeypatch, _FailingProvider(exc))

    result = service.start(
        AgentRunRequest(session_id="ses_run", user_message="fail", run_id="run-provider-error"),
        sink=sink,
    )

    assert result.exit_reason == "error"
    assert result.metadata["error"] == {
        "type": "ProviderInvocationError",
        "message": "provider HTTP 401: invalid api key",
        "status_code": 401,
        "body_summary": "invalid api key",
        "metadata": {"url": "https://provider.test/chat/completions", "method": "POST"},
    }
    assert isinstance(sink.events[-1], RunFailed)


class _RecordingProvider(ModelProvider):
    def __init__(self) -> None:
        self.messages_by_call: list[list[dict]] = []

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
        self.messages_by_call.append(messages)
        return NormalizedResponse(content="ok")


class _StreamingProvider(ModelProvider):
    def __init__(self, chunks: list[str], *, reasoning: str | None = None) -> None:
        self._chunks = chunks
        self._reasoning = reasoning

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del messages, tools, config, context, stream_silent_callback
        for chunk in self._chunks:
            if stream_callback is not None:
                stream_callback(chunk)
        metadata: dict[str, object] = {
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            }
        }
        if self._reasoning:
            metadata["reasoning_content"] = self._reasoning
        return NormalizedResponse(content="".join(self._chunks), metadata=metadata)


class _SilentToolProvider(ModelProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del messages, tools, config, context
        self.calls += 1
        if self.calls > 1:
            if stream_callback is not None:
                stream_callback("done")
            return NormalizedResponse(content="done")
        if stream_silent_callback is not None:
            stream_silent_callback('{"path":')
            stream_silent_callback('"/tmp/a"}')
        return NormalizedResponse(
            tool_calls=[ToolCall(id="tc_silent", name="echo", arguments={"text": "silent"})]
        )


class _CancellableProvider(ModelProvider):
    def __init__(self) -> None:
        self.first_delta = threading.Event()
        self.allow_next_delta = threading.Event()

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del messages, tools, config, context, stream_silent_callback
        if stream_callback is not None:
            stream_callback("a")
        self.first_delta.set()
        self.allow_next_delta.wait(timeout=2.0)
        if stream_callback is not None:
            stream_callback("b")
        return NormalizedResponse(content="ab")


class _BlockingProvider(ModelProvider):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del messages, tools, config, context, stream_callback, stream_silent_callback
        self.started.set()
        self.release.wait(timeout=2.0)
        return NormalizedResponse(content="done")


class _FailingProvider(ModelProvider):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del messages, tools, config, context, stream_callback, stream_silent_callback
        raise self._exc


class _EchoTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="echo",
            description="echo text",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
            required=["text"],
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=str(call.arguments.get("text", "")),
        )


def _test_config(max_iterations: int | None = None) -> EngineConfig:
    config = EngineConfig()
    config.max_iterations = max_iterations or 4
    config.use_builtin_tools = False
    config.tool_use_contract_enabled = False
    config.memory_enabled = False
    config.summary_on_budget_exhausted = False
    config.task_store_enabled = False
    config.allow_subagent_dispatch = False
    return config
