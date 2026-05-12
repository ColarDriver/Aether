"""CLI prompter for plan-mode approval and structured questions.

Sprint 3.5 / PR 3.5.7.  Backs ``ExitPlanModeTool`` and
``AskUserQuestionTool``.

Design constraints:

* Tools must remain testable without a TTY — they call
  :class:`ApprovalPrompter` through the abstract :class:`Prompter`
  protocol so unit tests can swap in :class:`StubPrompter`.
* The CLI builds the real :class:`ApprovalPrompter` and hands it to
  ``EngineRequest.approval_prompter``; the engine forwards it to
  ``TurnContext.metadata['_approval_prompter']`` for tools to pick up.
* No prompt-toolkit dependency in the **non-interactive** code path —
  ``is_interactive()`` checks ``isatty`` first, so headless runs (CI,
  scripts piping into ``aether``) do not import the heavy dialog code.

Question schema (kept aligned with ``AskUserQuestionTool`` JSON schema):

    {
      "id": "<unique>",
      "prompt": "<text>",
      "options": [{"id": "...", "label": "..."}],   # optional
      "allow_multiple": False,                       # optional
      "free_text": False                             # optional
    }
"""

from __future__ import annotations

import sys
from typing import Any, Mapping, Protocol, runtime_checkable

__all__ = ["Prompter", "ApprovalPrompter", "StubPrompter"]


@runtime_checkable
class Prompter(Protocol):
    """Minimal protocol the interaction tools depend on."""

    def is_interactive(self) -> bool: ...

    def confirm_plan(self, plan: str, *, context: Any | None = None) -> bool: ...

    def ask_questions(
        self,
        questions: list[Mapping[str, Any]],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------- stub


class StubPrompter:
    """Deterministic prompter for tests / non-interactive automation.

    All answers are passed in at construction time.  When a question
    is asked but no answer was prepared, ``"(no answer)"`` is returned —
    callers should treat empty/sentinel values as a valid response.
    """

    def __init__(
        self,
        *,
        approve_plan: bool = True,
        answers: Mapping[str, Any] | None = None,
        interactive: bool = True,
        raise_timeout: bool = False,
    ) -> None:
        self._approve = bool(approve_plan)
        self._answers: dict[str, Any] = dict(answers or {})
        self._interactive = bool(interactive)
        self._raise_timeout = bool(raise_timeout)
        self.confirm_calls: list[str] = []
        self.ask_calls: list[list[Mapping[str, Any]]] = []

    def is_interactive(self) -> bool:
        return self._interactive

    def confirm_plan(self, plan: str, *, context: Any | None = None) -> bool:
        self.confirm_calls.append(plan)
        return self._approve

    def ask_questions(
        self,
        questions: list[Mapping[str, Any]],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.ask_calls.append(list(questions))
        if self._raise_timeout:
            raise TimeoutError(f"stub prompter forced timeout (timeout={timeout})")
        out: dict[str, Any] = {}
        for q in questions:
            qid = q.get("id")
            if not qid:
                continue
            out[qid] = self._answers.get(qid, "(no answer)")
        return out


# --------------------------------------------------------------------- real


class ApprovalPrompter:
    """Real CLI implementation backed by ``prompt_toolkit``.

    All blocking calls happen on the calling thread — this is fine
    because the engine drives tools synchronously, and the REPL already
    runs each turn in ``asyncio.to_thread``.
    """

    def __init__(self, *, stdin: Any = None, stdout: Any = None) -> None:
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout

    def is_interactive(self) -> bool:
        try:
            return bool(self.stdin.isatty()) and bool(self.stdout.isatty())
        except (AttributeError, ValueError):
            return False

    def confirm_plan(self, plan: str, *, context: Any | None = None) -> bool:
        if not self.is_interactive():
            return False
        try:
            from prompt_toolkit import prompt as pt_prompt
        except ImportError:
            return self._fallback_confirm(plan)

        self._render_plan(plan)
        try:
            answer = pt_prompt("Approve? [y/N]: ", default="N")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in {"y", "yes"}

    def ask_questions(
        self,
        questions: list[Mapping[str, Any]],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not self.is_interactive():
            raise RuntimeError("ApprovalPrompter.ask_questions called in non-interactive mode")
        try:
            from prompt_toolkit import prompt as pt_prompt
            from prompt_toolkit.shortcuts import (
                checkboxlist_dialog,
                radiolist_dialog,
            )
        except ImportError:
            return self._fallback_ask(questions)

        answers: dict[str, Any] = {}
        for q in questions:
            qid = q.get("id")
            if not qid:
                continue
            prompt_text = str(q.get("prompt", ""))
            options = q.get("options") or []
            allow_multiple = bool(q.get("allow_multiple"))
            free_text = bool(q.get("free_text"))

            if options and not free_text:
                values = [(o["id"], str(o.get("label", o["id"]))) for o in options]
                if allow_multiple:
                    result = checkboxlist_dialog(
                        title=prompt_text or qid, values=values
                    ).run()
                else:
                    result = radiolist_dialog(
                        title=prompt_text or qid, values=values
                    ).run()
                answers[qid] = result if result is not None else ("(cancelled)" if not allow_multiple else [])
            else:
                try:
                    answers[qid] = pt_prompt(f"{prompt_text}\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    answers[qid] = "(cancelled)"
        return answers

    # ----------------------------------------------------------- fallbacks

    def _render_plan(self, plan: str) -> None:
        try:
            self.stdout.write("\n" + "─" * 60 + "\n")
            self.stdout.write("Proposed plan:\n\n")
            self.stdout.write(plan)
            if not plan.endswith("\n"):
                self.stdout.write("\n")
            self.stdout.write("─" * 60 + "\n")
            self.stdout.flush()
        except Exception:
            pass

    def _fallback_confirm(self, plan: str) -> bool:
        self._render_plan(plan)
        try:
            answer = input("Approve? [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in {"y", "yes"}

    def _fallback_ask(
        self, questions: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for q in questions:
            qid = q.get("id")
            if not qid:
                continue
            prompt_text = str(q.get("prompt", ""))
            options = q.get("options") or []
            if options:
                self.stdout.write(f"\n{prompt_text}\n")
                for i, o in enumerate(options, 1):
                    self.stdout.write(f"  {i}) {o.get('label', o['id'])}\n")
                self.stdout.flush()
                try:
                    raw = input("Choice [1]: ").strip() or "1"
                    idx = max(1, min(int(raw), len(options))) - 1
                    answers[qid] = options[idx]["id"]
                except (EOFError, KeyboardInterrupt, ValueError):
                    answers[qid] = "(cancelled)"
            else:
                try:
                    answers[qid] = input(f"{prompt_text}\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    answers[qid] = "(cancelled)"
        return answers
