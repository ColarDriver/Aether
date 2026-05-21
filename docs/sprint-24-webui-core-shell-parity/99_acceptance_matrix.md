# Sprint 24 Acceptance Matrix

| Area | Required Evidence |
| --- | --- |
| TypeScript implementation | New UI is React/TypeScript, no copied Hermes vanilla JS. |
| Three-panel shell | Chat route renders sidebar, chat column, and workspace rail on desktop. |
| Scroll stability | `body` and outer workspace stay fixed-height; chat history scrolls inside `.chat-scroll`. |
| Composer visibility | Composer remains visible after sending messages and on narrow widths. |
| Workspace rail | Tree, search, directory navigation, file preview, loading, and error states are covered. |
| Composer footer | Provider/model, workspace affordance, context ring, attach, stop, and send are visible and wrap safely. |
| Sidebar | Sessions, nav, new chat, search, appearance/settings controls are accessible. |
| Regression tests | `cd web && npm test`, `npm run typecheck`, and `npm run build` pass. |
| Diff hygiene | `git diff --check` is clean. |
