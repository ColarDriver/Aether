"""Static SPA mounting helpers."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles


def mount_spa(app: FastAPI, web_dist: Path, *, session_token: str) -> None:
    dist = web_dist.expanduser().resolve()
    assets_dir = dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="web-assets")

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    async def spa_fallback(request: Request, path: str) -> Response:
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "not_found", "message": "API route not found"}},
            )
        index_path = dist / "index.html"
        if not index_path.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "web_dist_missing",
                        "message": f"web build not found at {dist}",
                    }
                },
            )
        html = index_path.read_text(encoding="utf-8")
        return HTMLResponse(inject_bootstrap(html, session_token=session_token))


def inject_bootstrap(html: str, *, session_token: str, base_path: str = "") -> str:
    script = (
        "<script>"
        f"window.__AETHER_BASE_PATH__={base_path!r};"
        f"window.__AETHER_SESSION_TOKEN__={session_token!r};"
        "</script>"
    )
    module_index = html.find("<script type=\"module\"")
    if module_index >= 0:
        return html[:module_index] + script + html[module_index:]
    if "</head>" in html:
        return html.replace("</head>", f"{script}</head>", 1)
    return f"{script}{html}"


__all__ = ["inject_bootstrap", "mount_spa"]
