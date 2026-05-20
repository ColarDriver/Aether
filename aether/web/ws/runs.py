"""Structured agent-run WebSocket endpoint."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, cast

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from aether.services.common import ServiceConflictError, ServiceError
from aether.services.runs import (
    AgentRunOptions,
    AgentRunRequest,
    RunEvent,
    RunEventSink,
)
from aether.web.security import websocket_has_valid_session_token
from aether.web.ws.events import run_event_to_frame
from aether.web.ws.prompts import (
    WebApprovalPrompter,
    WebPromptBroker,
    WebToolPermissionPrompter,
)

router = APIRouter()


class _SocketEventSink:
    def __init__(self, outbound: "_OutboundQueue") -> None:
        self._outbound = outbound

    def emit(self, event: RunEvent) -> None:
        self._outbound.send(run_event_to_frame(event))


class _OutboundQueue:
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        self._loop = loop
        self._queue = queue
        self._sequence = 0

    def send(self, frame: dict[str, Any]) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, self._with_sequence(frame))

    async def send_async(self, frame: dict[str, Any]) -> None:
        await self._queue.put(self._with_sequence(frame))

    async def close(self) -> None:
        await self._queue.put(None)

    def _with_sequence(self, frame: dict[str, Any]) -> dict[str, Any]:
        payload = dict(frame)
        payload.setdefault("transport_sequence", self._sequence)
        self._sequence += 1
        return payload


@router.websocket("/api/runs/ws")
async def run_websocket(websocket: WebSocket) -> None:
    if not _authorized(websocket):
        await websocket.close(code=4401)
        return

    await websocket.accept()

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    outbound = _OutboundQueue(loop, queue)
    broker = WebPromptBroker(send_frame=outbound.send)
    sink = _SocketEventSink(outbound)
    run_tasks: set[asyncio.Task[Any]] = set()

    sender = asyncio.create_task(_send_loop(websocket, queue))
    await outbound.send_async({"type": "ready", "payload": {"protocol": "aether.run.v1"}})

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            await _handle_message(
                raw,
                websocket=websocket,
                outbound=outbound,
                broker=broker,
                sink=sink,
                run_tasks=run_tasks,
            )
    finally:
        broker.reject_all()
        for task in list(run_tasks):
            if task.done():
                continue
            task.cancel()
        await outbound.close()
        await sender


def _authorized(websocket: WebSocket) -> bool:
    app = websocket.app
    if not bool(getattr(app.state, "aether_auth_enabled", True)):
        return True
    token = str(getattr(app.state, "aether_session_token", ""))
    return bool(token) and websocket_has_valid_session_token(websocket, token)


async def _send_loop(
    websocket: WebSocket,
    queue: asyncio.Queue[dict[str, Any] | None],
) -> None:
    while True:
        frame = await queue.get()
        if frame is None:
            return
        await websocket.send_text(json.dumps(frame, ensure_ascii=False))


async def _handle_message(
    raw: str,
    *,
    websocket: WebSocket,
    outbound: _OutboundQueue,
    broker: WebPromptBroker,
    sink: RunEventSink,
    run_tasks: set[asyncio.Task[Any]],
) -> None:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        await outbound.send_async(_error_frame("invalid_json", "Message must be valid JSON."))
        return
    if not isinstance(message, dict):
        await outbound.send_async(_error_frame("invalid_message", "Message must be a JSON object."))
        return

    message_type = str(message.get("type") or "")
    raw_payload = message.get("payload")
    payload = cast(dict[str, Any], raw_payload) if isinstance(raw_payload, dict) else {}
    if message_type == "ping":
        await outbound.send_async({"type": "pong", "payload": {}})
    elif message_type == "run.start":
        await _start_run(
            payload,
            client_id=message.get("id"),
            websocket=websocket,
            outbound=outbound,
            broker=broker,
            sink=sink,
            run_tasks=run_tasks,
        )
    elif message_type == "run.cancel":
        await _cancel_run(payload, websocket=websocket, outbound=outbound)
    elif message_type in {"permission.respond", "approval.respond"}:
        await _resolve_prompt(payload, broker=broker, outbound=outbound)
    else:
        await outbound.send_async(_error_frame("unknown_message_type", f"Unknown message type: {message_type}"))


async def _start_run(
    payload: dict[str, Any],
    *,
    client_id: Any,
    websocket: WebSocket,
    outbound: _OutboundQueue,
    broker: WebPromptBroker,
    sink: RunEventSink,
    run_tasks: set[asyncio.Task[Any]],
) -> None:
    session_id = _required_str(payload, "session_id")
    user_message = _required_str(payload, "user_message")
    if not session_id or not user_message:
        await outbound.send_async(_error_frame("invalid_run_start", "run.start requires session_id and user_message."))
        return
    run_id = str(payload.get("run_id") or client_id or uuid.uuid4())
    options = _run_options(payload.get("options"))
    await outbound.send_async(
        {
            "type": "run.accepted",
            "id": client_id,
            "payload": {"session_id": session_id, "run_id": run_id},
        }
    )
    task = asyncio.create_task(
        asyncio.to_thread(
            _run_start_sync,
            websocket.app.state.aether_services.runs,
            AgentRunRequest(
                session_id=session_id,
                user_message=user_message,
                run_id=run_id,
                options=options,
                approval_prompter=WebApprovalPrompter(
                    broker=broker,
                    session_id=session_id,
                    run_id=run_id,
                ),
                tool_permission_prompter=WebToolPermissionPrompter(
                    broker=broker,
                    run_id=run_id,
                ),
            ),
            sink,
            outbound,
        )
    )
    run_tasks.add(task)
    task.add_done_callback(run_tasks.discard)


def _run_start_sync(
    service: Any,
    request: AgentRunRequest,
    sink: RunEventSink,
    outbound: _OutboundQueue,
) -> None:
    try:
        result = service.start(request, sink=sink)
    except ServiceConflictError as exc:
        outbound.send(_error_frame(exc.code, exc.message, details=exc.details))
    except ServiceError as exc:
        outbound.send(_error_frame(exc.code, exc.message, details=exc.details))
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        outbound.send(_error_frame("run_failed", str(exc) or type(exc).__name__))
    else:
        outbound.send(
            {
                "type": "run.result",
                "payload": {
                    "session_id": result.session_id,
                    "run_id": result.run_id,
                    "final_text": result.final_text,
                    "exit_reason": result.exit_reason,
                    "usage": dict(result.usage or {}),
                    "metadata": dict(result.metadata or {}),
                },
            }
        )


async def _cancel_run(
    payload: dict[str, Any],
    *,
    websocket: WebSocket,
    outbound: _OutboundQueue,
) -> None:
    from aether.services.runs import AgentRunCancelRequest

    session_id = _required_str(payload, "session_id")
    run_id = _optional_str(payload, "run_id")
    if not session_id:
        await outbound.send_async(_error_frame("invalid_cancel", "run.cancel requires session_id."))
        return
    cancelled = websocket.app.state.aether_services.runs.cancel(
        AgentRunCancelRequest(
            session_id=session_id,
            run_id=run_id,
            reason=_optional_str(payload, "reason"),
        )
    )
    await outbound.send_async(
        {
            "type": "run.cancel.accepted",
            "payload": {"session_id": session_id, "run_id": run_id, "cancelled": cancelled},
        }
    )


async def _resolve_prompt(
    payload: dict[str, Any],
    *,
    broker: WebPromptBroker,
    outbound: _OutboundQueue,
) -> None:
    prompt_id = _required_str(payload, "prompt_id")
    if not prompt_id:
        await outbound.send_async(_error_frame("invalid_prompt_response", "prompt response requires prompt_id."))
        return
    resolved = broker.resolve(prompt_id, payload)
    await outbound.send_async(
        {
            "type": "prompt.resolved" if resolved else "prompt.missing",
            "payload": {"prompt_id": prompt_id},
        }
    )


def _run_options(raw: Any) -> AgentRunOptions:
    if not isinstance(raw, dict):
        raw = {}
    return AgentRunOptions(
        max_iterations=_optional_int(raw, "max_iterations"),
        temperature=_optional_float(raw, "temperature"),
        max_tokens=_optional_int(raw, "max_tokens"),
        disable_builtin_tools=_optional_bool(raw, "disable_builtin_tools"),
        system_override=_optional_str(raw, "system_override"),
    )


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) and value > 0 else None


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _error_frame(code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "error",
        "payload": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }


__all__ = ["router"]
