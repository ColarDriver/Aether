"""Structured agent-run WebSocket endpoint."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from typing import Any, Callable, cast

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

FrameSender = Callable[[dict[str, Any]], None]


class _SocketEventSink:
    def __init__(self, send_frame: "FrameSender") -> None:
        self._send_frame = send_frame

    def emit(self, event: RunEvent) -> None:
        self._send_frame(run_event_to_frame(event))


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
    run_hub = websocket.app.state.aether_run_socket_hub
    broker = websocket.app.state.aether_web_prompt_broker
    run_tasks = websocket.app.state.aether_run_tasks
    socket_sender = outbound.send
    run_hub.attach(socket_sender)
    sink = _SocketEventSink(run_hub.broadcast)

    sender = asyncio.create_task(_send_loop(websocket, queue))
    await outbound.send_async({"type": "ready", "payload": {"protocol": "aether.run.v1"}})
    broker.replay_pending(socket_sender)

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
        run_hub.detach(socket_sender)
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
    run_tasks: set[Any],
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
    run_tasks: set[Any],
) -> None:
    session_id = _required_str(payload, "session_id")
    user_message = _optional_str(payload, "user_message") or ""
    attachments = _run_attachments(payload.get("attachments"))
    attachments = _enrich_workspace_reference_attachments(
        attachments,
        getattr(websocket.app.state.aether_services, "workspace", None),
    )
    if not session_id or (not user_message and not attachments):
        await outbound.send_async(_error_frame("invalid_run_start", "run.start requires session_id and user_message or attachments."))
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
    request = AgentRunRequest(
        session_id=session_id,
        user_message=user_message,
        attachments=attachments,
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
    )
    thread = threading.Thread(
        target=_run_start_thread,
        args=(
            websocket.app.state.aether_services.runs,
            request,
            sink,
            websocket.app.state.aether_run_socket_hub.broadcast,
            run_tasks,
        ),
        name=f"aether-web-run-{run_id}",
        daemon=True,
    )
    run_tasks.add(thread)
    thread.start()


def _run_start_thread(
    service: Any,
    request: AgentRunRequest,
    sink: RunEventSink,
    send_frame: FrameSender,
    run_tasks: set[Any],
) -> None:
    try:
        _run_start_sync(service, request, sink, send_frame)
    finally:
        run_tasks.discard(threading.current_thread())


def _run_start_sync(
    service: Any,
    request: AgentRunRequest,
    sink: RunEventSink,
    send_frame: FrameSender,
) -> None:
    try:
        result = service.start(request, sink=sink)
    except ServiceConflictError as exc:
        send_frame(_error_frame(exc.code, exc.message, details=exc.details))
    except ServiceError as exc:
        send_frame(_error_frame(exc.code, exc.message, details=exc.details))
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        send_frame(_error_frame("run_failed", str(exc) or type(exc).__name__))
    else:
        send_frame(
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
    if cancelled and run_id:
        websocket.app.state.aether_web_prompt_broker.reject_run(run_id)
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
    resolution = _prompt_resolution_payload(payload)
    resolved = broker.resolve(prompt_id, resolution)
    await outbound.send_async(
        {
            "type": "prompt.resolved" if resolved else "prompt.missing",
            "payload": {"prompt_id": prompt_id, "result": resolution},
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


def _enrich_workspace_reference_attachments(
    attachments: list[dict[str, Any]],
    workspace: Any,
) -> list[dict[str, Any]]:
    if workspace is None or not attachments:
        return attachments
    enriched: list[dict[str, Any]] = []
    for attachment in attachments:
        if attachment.get("note") != "workspace reference":
            enriched.append(attachment)
            continue
        path = attachment.get("path")
        if not isinstance(path, str) or not path or attachment.get("isDirectory") is True:
            enriched.append(attachment)
            continue
        next_attachment = dict(attachment)
        try:
            workspace_file = workspace.read_file(path)
        except Exception as exc:  # noqa: BLE001 - surface context miss to the model
            next_attachment["_llm_error"] = str(exc) or type(exc).__name__
        else:
            next_attachment.setdefault("name", workspace_file.name)
            next_attachment["path"] = workspace_file.path
            next_attachment["_llm_language"] = workspace_file.language
            next_attachment["_llm_truncated"] = bool(workspace_file.truncated)
            next_attachment["_llm_binary"] = bool(workspace_file.binary)
            if not workspace_file.binary:
                next_attachment["_llm_content"] = workspace_file.content
        enriched.append(next_attachment)
    return enriched


def _run_attachments(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    attachments: list[dict[str, Any]] = []
    for item in raw[:20]:
        attachment = _run_attachment(item)
        if attachment:
            attachments.append(attachment)
    return attachments


def _run_attachment(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    attachment_type = _attachment_type(raw.get("type") or raw.get("kind"))
    attachment: dict[str, Any] = {"type": attachment_type}
    for target, keys in {
        "name": ("name", "filename"),
        "path": ("path", "file_path", "filePath"),
        "url": ("url", "previewUrl", "preview_url"),
        "mimeType": ("mimeType", "mime_type", "media_type"),
        "note": ("note",),
        "quote": ("quote",),
    }.items():
        value = _first_optional_str(raw, keys)
        if value:
            attachment[target] = value
    data = _raw_optional_str(raw, "data")
    if data:
        attachment["data"] = data
    if isinstance(raw.get("isDirectory"), bool):
        attachment["isDirectory"] = raw["isDirectory"]
    elif isinstance(raw.get("is_directory"), bool):
        attachment["isDirectory"] = raw["is_directory"]
    line_start = _first_optional_int(raw, ("lineStart", "line_start"))
    line_end = _first_optional_int(raw, ("lineEnd", "line_end"))
    if line_start is not None:
        attachment["lineStart"] = line_start
    if line_end is not None:
        attachment["lineEnd"] = line_end
    return attachment if len(attachment) > 1 else None


def _attachment_type(value: Any) -> str:
    return value if value in {"file", "image", "text"} else "file"


def _first_optional_str(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _optional_str(payload, key)
        if value:
            return value
    return None


def _first_optional_int(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _optional_int(payload, key)
        if value is not None:
            return value
    return None


def _raw_optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _prompt_resolution_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    if isinstance(result, dict):
        merged = dict(payload)
        merged.update(result)
        return merged
    return payload


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
