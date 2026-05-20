# Sprint 20 - Web Console Migration

This sprint migrates useful web-console patterns from Hermes Agent and cc-haha
into Aether with a TypeScript browser frontend and Python backend.

## Documents

- [Overview](00_overview.md)
- [PR20.1 - Web Backend Foundation](01_pr20_1_web_backend_foundation.md)
- [PR20.2 - REST API Services](02_pr20_2_rest_api_services.md)
- [PR20.3 - Run Streaming and Approvals](03_pr20_3_run_streaming_and_approvals.md)
- [PR20.4 - Web Frontend Shell](04_pr20_4_web_frontend_shell.md)
- [PR20.5 - Chat Transcript, Tools, Diff, and Permissions](05_pr20_5_chat_transcript_tools_diff.md)
- [PR20.6 - Settings, Models, Skills, Tools, and Health Views](06_pr20_6_settings_models_skills_health.md)
- [PR20.7 - Tests, Dev Server, Packaging, and Acceptance](07_pr20_7_tests_dev_server_acceptance.md)
- [Acceptance Matrix](99_acceptance_matrix.md)

## Implementation Principle

The web backend is an adapter over `aether/services/*`. It must not become a
second business layer, and it must not import gateway handlers. The browser app
is a real console surface, not a terminal emulator and not a marketing page.
