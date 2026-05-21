# Aether Web Console

Standalone React/TypeScript console for the local Aether backend. The browser app talks to the Python service over REST and the structured run WebSocket; it does not use the TUI gateway handlers directly.

## Backend plus Vite

Run the Python backend:

~~~bash
uv run aether-web --port 9120
~~~

Run the frontend dev server:

~~~bash
cd web
npm run dev
~~~

Vite proxies /api and /api/runs/ws to http://127.0.0.1:9120 by default. Override the backend target when needed:

~~~bash
AETHER_WEB_BACKEND=http://127.0.0.1:9130 npm run dev
~~~

## Built SPA served by Python

Build the frontend:

~~~bash
cd web
npm run build
~~~

Serve it from the Python web entrypoint:

~~~bash
uv run aether-web --web-dist web/dist
~~~

The Python static mount injects window.__AETHER_BASE_PATH__ and window.__AETHER_SESSION_TOKEN__ into index.html. Unknown non-API routes fall back to the SPA index; unknown API routes return a structured JSON 404.

## Checks

~~~bash
npm test
npm run build
~~~

Backend checks live at the repository root:

~~~bash
python -m pytest aether/tests/web
python -m pytest aether/tests/services
python -m pytest aether/tests/gateway
~~~

## Manual Smoke Path

1. Start the backend and Vite dev server.
2. Open the Vite URL.
3. Create or select a session.
4. Send a message and confirm assistant deltas stream into one response.
5. Trigger a permission or plan approval and confirm the modal can approve/reject.
6. Reload the page and confirm transcript messages plus historical tool/diff blocks still render.
7. Visit Models, Tools, Skills, Diagnostics, and Settings.
