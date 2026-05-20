"""Local web-console security helpers."""

from __future__ import annotations

import hmac
import secrets

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


SESSION_HEADER_NAME = "X-Aether-Session-Token"
PUBLIC_API_PATHS: frozenset[str] = frozenset({"/api/status", "/api/health"})
_LOOPBACK_HOST_VALUES: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


def create_session_token() -> str:
    return secrets.token_urlsafe(32)


def install_security(
    app: FastAPI,
    *,
    auth_enabled: bool,
    session_token: str,
    bound_host: str | None,
) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def host_header_middleware(request: Request, call_next):
        if bound_host and not is_accepted_host(request.headers.get("host", ""), bound_host):
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "invalid_host", "message": "Invalid Host header."}},
            )
        return await call_next(request)

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if not auth_enabled:
            return await call_next(request)
        path = request.url.path
        if path.startswith("/api/") and path not in PUBLIC_API_PATHS:
            if not has_valid_session_token(request, session_token):
                return JSONResponse(
                    status_code=401,
                    content={"error": {"code": "unauthorized", "message": "Unauthorized"}},
                )
        return await call_next(request)


def has_valid_session_token(request: Request, session_token: str) -> bool:
    session_header = request.headers.get(SESSION_HEADER_NAME, "")
    if session_header and hmac.compare_digest(session_header.encode(), session_token.encode()):
        return True

    auth = request.headers.get("authorization", "")
    expected = f"Bearer {session_token}"
    return hmac.compare_digest(auth.encode(), expected.encode())


def is_accepted_host(host_header: str, bound_host: str) -> bool:
    if not host_header:
        return False
    host_only = _host_without_port(host_header)
    bound_lc = bound_host.lower()
    if bound_lc in {"0.0.0.0", "::"}:
        return True
    if bound_lc in _LOOPBACK_HOST_VALUES:
        return host_only in _LOOPBACK_HOST_VALUES
    return host_only == bound_lc


def _host_without_port(host_header: str) -> str:
    host = host_header.strip()
    if host.startswith("["):
        close = host.find("]")
        if close != -1:
            return host[1:close].lower()
        return host.strip("[]").lower()
    if ":" in host:
        return host.rsplit(":", 1)[0].lower()
    return host.lower()


__all__ = [
    "PUBLIC_API_PATHS",
    "SESSION_HEADER_NAME",
    "create_session_token",
    "has_valid_session_token",
    "install_security",
    "is_accepted_host",
]
