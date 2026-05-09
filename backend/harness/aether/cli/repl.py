"""Interactive REPL loop for the Aether CLI.

Wires together:
    * :class:`aether.cli.app.AetherApp` — long-lived ``prompt_toolkit``
      Application that keeps the input frame visible across turns.
    * ``rich`` for output (streaming, panels, colour) — captured by
      :func:`prompt_toolkit.patch_stdout.patch_stdout` so prints land
      above the prompt.
    * :class:`aether.cli.ui_middleware.CLIUIMiddleware` for in-loop
      tool-call rendering.

Engine threading model: the Aether ``AgentEngine`` is synchronous, so
each user message is dispatched via ``asyncio.to_thread`` from the
asyncio event loop.  This keeps the bottom region (activity bar,
active tool group, queued badge) responsive while the engine runs.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from prompt_toolkit.history import FileHistory
from rich.console import Console

from aether.agents.core.agent import AgentEngine
from aether.cli import commands as slash
from aether.cli import sessions as session_store
from aether.cli.app import AetherApp
from aether.cli.banner import BannerInfo, render_banner
from aether.cli.commands import SlashCompleter
from aether.cli.sessions import SessionRecord
from aether.cli.theme import THEME, color_enabled
from aether.cli.ui import CLIUI
from aether.cli.ui_middleware import CLIUIHooks, CLIUIMiddleware
from aether.config.schema import ModelCallConfig
from aether.runtime.contracts import EngineRequest, EngineStatus

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# REPL state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ReplState:
    engine: AgentEngine
    ui: CLIUI
    provider_name: str
    session_id: str
    system_prompt: str | None
    model_config: ModelCallConfig
    messages: list[dict] = field(default_factory=list)
    verbose: bool = False
    # Live, in-memory mirror of the on-disk session record.  ``None``
    # until the first turn (or until a successful ``/resume``).  We keep
    # the dataclass instance around so ``created_at`` survives across
    # turns and we don't re-compute ``first_user_message`` on every save.
    session_record: SessionRecord | None = None


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------

def _attach_ui_middleware(engine: AgentEngine, ui: CLIUI) -> None:
    """Install the UI middleware/hooks on the engine.

    Idempotent: re-attaching is a no-op so resuming a session inside the
    same process doesn't double-render tool panels.
    """
    pipeline = engine.services.middleware_pipeline
    already = any(isinstance(m, CLIUIMiddleware) for m in pipeline.middlewares)
    if not already:
        pipeline.add(CLIUIMiddleware(ui))

    if not isinstance(getattr(engine, "_hooks", None), CLIUIHooks):
        engine._hooks = CLIUIHooks(ui)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_repl(
    engine: AgentEngine,
    *,
    provider_name: str,
    system_prompt: str | None = None,
    model_config: ModelCallConfig | None = None,
    history_file: "Path | None" = None,
    session_id: str | None = None,
    verbose: bool = False,
    base_url: str | None = None,
    version: str = "0.1.0",
    show_banner: bool = True,
    resume_record: SessionRecord | None = None,
) -> None:
    """Run the interactive REPL against *engine*.

    Key bindings are documented inside :class:`aether.cli.app.AetherApp`.

    Pass ``resume_record`` to seed the loop with a previously persisted
    conversation; ``session_id`` is then derived from the record.
    """
    console = Console(theme=THEME, force_terminal=True if color_enabled() else None)
    ui = CLIUI(console, verbose=verbose)

    if resume_record is not None:
        effective_session_id = resume_record.session_id
        seed_messages = list(resume_record.messages)
        if resume_record.system_prompt and not system_prompt:
            system_prompt = resume_record.system_prompt
    else:
        effective_session_id = session_id or str(uuid.uuid4())
        seed_messages = []

    state = ReplState(
        engine=engine,
        ui=ui,
        provider_name=provider_name,
        session_id=effective_session_id,
        system_prompt=system_prompt,
        model_config=model_config or ModelCallConfig(),
        messages=seed_messages,
        verbose=verbose,
        session_record=resume_record,
    )

    _attach_ui_middleware(engine, ui)

    pt_history = FileHistory(str(history_file)) if history_file else None
    completer = SlashCompleter()

    # ------------------------------ banner --------------------------------

    if show_banner:
        banner_info = BannerInfo(
            version=version,
            provider=provider_name,
            model=getattr(engine.services.provider, "model", "?"),
            base_url=base_url or getattr(engine.services.provider, "base_url", None),
            session_id=state.session_id,
            cwd=_short_cwd(),
            tool_count=len(engine.services.tool_registry.list_descriptors()),
            system_prompt_set=bool(system_prompt),
        )
        render_banner(console, banner_info)
    else:
        ui.boot_line(
            f"Aether {version}  ·  {provider_name} / "
            f"{getattr(engine.services.provider, 'model', '?')}  ·  "
            f"session {state.session_id[:8]}"
        )
        ui.blank()

    if resume_record is not None:
        _announce_resume(state, resume_record)

    # ------------------------------ async loop ----------------------------

    async def _on_submit(line: str) -> None:
        line = line.strip()
        if not line:
            return
        ui.render_input_echo(line)
        if slash.is_slash(line):
            outcome = slash.dispatch(state, line)
            if outcome.exit:
                # The ``finally`` clause below is the single source of
                # the farewell ``Bye.`` line — printing one here as
                # well would land twice in scrollback once the app
                # tears down.
                app.request_exit()
            return
        # Engine is synchronous — run it in a worker thread so the
        # asyncio event loop (and therefore the activity bar / queued
        # badge / Ctrl-C interrupt) stays responsive.
        await asyncio.to_thread(_run_turn_blocking, state, line)

    def _on_interrupt() -> None:
        try:
            state.engine.interrupt(
                session_id=state.session_id,
                reason="user-interrupt",
            )
        except Exception:  # noqa: BLE001 — interrupt is best-effort
            pass
        ui.warn("interrupted")

    app = AetherApp(
        ui,
        on_submit=_on_submit,
        history=pt_history,
        completer=completer,
        on_interrupt=_on_interrupt,
    )

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        # Should not normally land here — Ctrl-C inside the running app
        # is captured by the AetherApp's keybinding.  Be defensive in
        # case the user signals before/after the event loop is alive.
        pass
    finally:
        ui.info("Bye.")


# ---------------------------------------------------------------------------
# Single agent turn
# ---------------------------------------------------------------------------

def _run_turn_blocking(state: ReplState, user_input: str) -> None:
    """Synchronous body of one user turn — invoked via ``asyncio.to_thread``.

    Stays sync because :class:`AgentEngine` is sync; the asyncio event
    loop continues to spin in the main thread so the activity bar, the
    queued-message badge, and Ctrl-C interrupts all remain responsive
    while we sit inside ``engine.run_loop``.
    """
    ui = state.ui
    # Sprint 3.5 / PR-2 — interactive prompter forwarded to
    # ``ExitPlanModeTool`` and ``AskUserQuestionTool``.  Built lazily
    # per turn so each turn sees the live stdin/stdout (REPL stdio
    # never changes mid-process today, but constructing on demand keeps
    # the engine free to swap streams in tests).
    from aether.cli.approval_prompter import ApprovalPrompter

    request = EngineRequest(
        session_id=state.session_id,
        user_message=user_input,
        messages=list(state.messages),
        system_message=state.system_prompt,
        model_config=state.model_config,
        stream_callback=ui.make_stream_callback(),
        # Sprint 3 / PR 3.1: separate count-only channel so providers
        # can tick the live token estimator during long tool-call
        # generation phases without polluting the visible body.
        stream_silent_callback=ui.make_stream_silent_callback(),
        approval_prompter=ApprovalPrompter(),
    )

    ui.begin_turn()

    status_value = "FAILED"
    exit_reason_value = "engine_error"
    iterations = 0
    result_messages: list[dict] | None = None
    phantom_synth_count = 0
    phantom_synth_notes: list[str] = []

    try:
        result = state.engine.run_loop(request)
    except KeyboardInterrupt:
        state.engine.interrupt(session_id=state.session_id, reason="user-interrupt")
        status_value = "INTERRUPTED"
        exit_reason_value = "user_interrupt"
        ui.warn("interrupted")
    except Exception as exc:  # noqa: BLE001 — surfaced to user
        ui.error(f"engine error: {exc}")
    else:
        status_value = result.status.value
        exit_reason_value = result.exit_reason.value
        iterations = result.iterations
        result_messages = list(result.messages)
        # Sprint 1.5 / P0-9: surface engine-side synthesis to the UI.
        # ``runtime.phantom_tool_synthesized`` counts how many prose
        # intents were rescued into structured ``tool_calls``;
        # ``turn.phantom_synth_notes`` carries the per-call
        # human-readable hints we want to render in the soft
        # acknowledgement line.
        try:
            runtime_meta = result.metadata.get("runtime", {}) or {}
            turn_meta = result.metadata.get("turn", {}) or {}
            phantom_synth_count = int(runtime_meta.get("phantom_tool_synthesized", 0) or 0)
            raw_notes = turn_meta.get("phantom_synth_notes") or []
            if isinstance(raw_notes, list):
                phantom_synth_notes = [str(n) for n in raw_notes]
        except Exception:  # noqa: BLE001 — UI hint must never fail the turn
            phantom_synth_count = 0
            phantom_synth_notes = []
        if (
            result.status != EngineStatus.FAILED
            and result.final_response
            and not result.streamed
        ):
            ui.render_assistant_block(result.final_response)
        if result.status == EngineStatus.FAILED:
            ui.error(result.error or "engine failed without an error message")

    ui.end_turn(
        status=status_value,
        exit_reason=exit_reason_value,
        iterations=iterations,
        error=None,
        phantom_synth_count=phantom_synth_count,
        phantom_synth_notes=phantom_synth_notes,
    )

    if result_messages is not None:
        state.messages = result_messages
        persist_session(state)


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

def persist_session(state: ReplState) -> None:
    """Atomically snapshot the current conversation to disk.

    No-op when there's nothing to remember (no messages exchanged yet).
    Called from the REPL after every successful turn and from
    ``/resume`` before swapping in a different conversation.
    """
    if not state.messages:
        return

    provider = state.engine.services.provider
    model = getattr(provider, "model", "") or ""
    base_url = getattr(provider, "base_url", None)

    if state.session_record is None:
        state.session_record = SessionRecord.new(
            session_id=state.session_id,
            provider=state.provider_name,
            model=model,
            base_url=base_url,
            system_prompt=state.system_prompt,
        )

    session_store.update_session_from_state(
        state.session_record,
        messages=state.messages,
        provider=state.provider_name,
        model=model,
        base_url=base_url,
        system_prompt=state.system_prompt,
    )

    try:
        session_store.save_session(state.session_record)
    except Exception as exc:  # noqa: BLE001 — persistence must never crash REPL
        state.ui.warn(f"could not save session: {exc}")


def reset_session_record(state: ReplState) -> None:
    """Forget the in-memory record; called from ``/new``."""
    state.session_record = None


def _announce_resume(state: ReplState, record: SessionRecord) -> None:
    """Print a one-line summary of the conversation we just restored."""
    summary = session_store.format_session_preview(record, max_chars=80)
    state.ui.success(
        f"Resumed session {record.session_id[:8]}  ·  {summary}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_cwd() -> str:
    import os
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        return "~" + cwd[len(home):]
    return cwd
