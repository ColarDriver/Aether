"""Built-in ``ask_user_question`` tool.

Lets the model interrupt its turn to ask the user a structured
question via the configured :class:`Prompter`.  Each item in the
``questions`` array follows the same JSON shape as the parameters
schema below; the prompter is responsible for rendering radio /
checkbox / free-text inputs as appropriate.

This tool is intentionally **not** in ``cheap_tool_names``: it does
real interactive work on the user's behalf and consumes a real
iteration slot so the IterationBudget bookkeeping remains honest.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor


_DEFAULT_TIMEOUT_SECONDS = 600


class AskUserQuestionTool(ToolExecutor):
    """Ask the user one or more structured questions and return their answers."""

    NAME = "ask_user_question"

    def __init__(
        self,
        prompter: Any | None = None,
        *,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._prompter = prompter
        self._timeout_seconds = int(timeout_seconds)
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Ask the user one or more structured questions. Use only "
                "when you genuinely need user input to proceed (e.g. an "
                "ambiguous requirement). 1 to 4 questions per call.\n"
                "Each question must include:\n"
                "  - `id`: stable identifier used to key the answer.\n"
                "  - `prompt`: the full question text shown to the user "
                "(ends with a question mark).\n"
                "  - `header`: a short 1-3 word chip label "
                "(e.g. \"Interface\", \"Tools\"). Max 12 characters.\n"
                "Each question may be free-text (omit `options`) or "
                "multiple-choice. When you provide `options`, supply "
                "2-4 mutually exclusive choices, each shaped as "
                "`{id, label, description}`:\n"
                "  - `label`: concise 1-5 word choice text.\n"
                "  - `description`: one short sentence explaining what "
                "the option means or its trade-off.\n"
                "If you recommend a specific option, place it first and "
                "append `(Recommended)` to its label. Do NOT add an "
                "`Other` option — the UI offers free-text input "
                "automatically. A single-option question is a "
                "confirmation and will be rejected.\n"
                "In plan mode, use this only to clarify requirements or "
                "choose between approaches; never ask whether the plan "
                "is approved or whether to proceed — use exit_plan_mode "
                "for plan approval. Not allowed inside subagent contexts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Stable identifier used to key the answer.",
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "Full question text shown to the user.",
                                },
                                "header": {
                                    "type": "string",
                                    "maxLength": 12,
                                    "description": "1-3 word chip label for the question navigation bar.",
                                },
                                "options": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 4,
                                    "description": (
                                        "2-4 mutually exclusive choices. "
                                        "Omit entirely for a free-text question."
                                    ),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "label": {"type": "string"},
                                            "description": {
                                                "type": "string",
                                                "description": "One short sentence explaining the choice.",
                                            },
                                        },
                                        "required": ["id", "label", "description"],
                                    },
                                },
                                "multi_select": {"type": "boolean", "default": False},
                                "allow_multiple": {"type": "boolean", "default": False},
                                "free_text": {"type": "boolean", "default": False},
                            },
                            "required": ["id", "prompt"],
                        },
                    },
                },
                "required": ["questions"],
            },
            required=["questions"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        questions = args.get("questions")
        if not isinstance(questions, list) or not questions:
            return _error(call, "'questions' must be a non-empty list")
        validated: List[Dict[str, Any]] = []
        for i, q in enumerate(questions):
            err = _validate_question(q, i)
            if err is not None:
                return _error(call, err)
            validated.append(dict(q))

        config = context.metadata.get("_engine_config") if context.metadata else None
        if not bool(getattr(config, "ask_user_question_enabled", True)):
            return _error(call, "ask_user_question is disabled by configuration")

        parent_agent = context.metadata.get("_parent_agent") if context.metadata else None
        depth = int(getattr(parent_agent, "delegate_depth", 0) or 0)
        if depth > 0:
            return _error(
                call,
                "ask_user_question is not allowed inside subagent contexts. "
                "Either return your uncertainty in the subagent summary, "
                "or have the parent agent ask the user.",
            )

        prompter = self._prompter or (
            context.metadata.get("_approval_prompter") if context.metadata else None
        )
        if prompter is None or not getattr(prompter, "ask_questions", None):
            return _error(
                call,
                "ask_user_question unavailable: no approval prompter is "
                "configured. Make a best-effort decision and continue.",
            )
        if hasattr(prompter, "is_interactive") and not prompter.is_interactive():
            return _error(
                call,
                "ask_user_question unavailable: running in non-interactive "
                "mode. Make a best-effort decision and continue.",
            )

        timeout_seconds = int(
            getattr(config, "ask_user_question_timeout_seconds", self._timeout_seconds)
        )

        plan_path = _resolve_plan_path(context)

        try:
            kwargs: Dict[str, Any] = {"timeout": timeout_seconds}
            if plan_path is not None:
                kwargs["plan_path"] = plan_path
            try:
                answers = prompter.ask_questions(validated, **kwargs)
            except TypeError:
                # Older prompter implementations may not accept plan_path —
                # retry without it rather than failing the tool call.
                kwargs.pop("plan_path", None)
                answers = prompter.ask_questions(validated, **kwargs)
        except TimeoutError:
            return _error(
                call,
                f"User did not respond within {timeout_seconds}s",
                metadata={"timeout_seconds": timeout_seconds},
            )
        except Exception as exc:
            return _error(call, f"prompter failed: {exc}")

        if not isinstance(answers, Mapping):
            return _error(call, "prompter returned a non-mapping answer payload")

        user_action = answers.get("__user_action") if isinstance(answers, Mapping) else None
        if isinstance(user_action, str):
            action = user_action.strip().lower()
            if action == "chat":
                return _error(
                    call,
                    "User chose to chat about this instead of answering. "
                    "Pause and let them lead the conversation before "
                    "attempting another question.",
                    metadata={"user_action": "chat"},
                )
            if action == "skip":
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=(
                        "User chose to skip remaining questions and "
                        "proceed to planning. Finalize the plan now "
                        "without asking further clarifying questions."
                    ),
                    is_error=False,
                    metadata={
                        "question_count": len(validated),
                        "answer_count": 0,
                        "user_action": "skip",
                    },
                )

        pairs = self._answer_pairs(validated, answers)
        formatted = self._format_answers(pairs)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=formatted,
            is_error=False,
            metadata={
                "question_count": len(validated),
                "answer_count": sum(1 for q in validated if q["id"] in answers),
                # Structured pairs so the TUI can render its own
                # Claude-style "User answered the questions:" block
                # without parsing the prose content.
                "answer_pairs": [
                    {"label": label, "value": value} for label, value in pairs
                ],
            },
        )

    @staticmethod
    def _answer_pairs(
        questions: List[Dict[str, Any]],
        answers: Mapping[str, Any],
    ) -> List[tuple[str, str]]:
        clean = {k: v for k, v in answers.items() if not k.startswith("__")}
        pairs: List[tuple[str, str]] = []
        for q in questions:
            qid = q["id"]
            if qid not in clean:
                continue
            raw = clean[qid]
            if isinstance(raw, list):
                value = ", ".join(str(item) for item in raw) if raw else "(none selected)"
            else:
                value = str(raw)
            # Prefer the short chip header; fall back to a truncated prompt
            # so the result line stays compact.
            header = (q.get("header") or "").strip()
            label = header or _shorten(str(q.get("prompt") or qid), 60)
            pairs.append((label, value))
        return pairs

    @staticmethod
    def _format_answers(pairs: List[tuple[str, str]]) -> str:
        if not pairs:
            return "User has not provided any answers."
        joined = ", ".join(f'"{label}"="{value}"' for label, value in pairs)
        return (
            "User has answered your questions: "
            + joined
            + ". You can now continue with the user's answers in mind."
        )


def _shorten(text: str, limit: int) -> str:
    cleaned = text.strip().replace("\n", " ")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(1, limit - 1)] + "…"


def _resolve_plan_path(context: TurnContext) -> Optional[str]:
    """Return the absolute plan file path when plan mode is active."""
    metadata = context.metadata or {}
    if not metadata.get("plan_mode_active"):
        return None
    session_id = getattr(context, "session_id", None)
    if not isinstance(session_id, str) or not session_id:
        return None
    try:
        from aether.runtime.session.plan_artifact import get_plan_path
    except Exception:  # pragma: no cover - defensive
        return None
    try:
        return str(get_plan_path(session_id))
    except Exception:  # pragma: no cover - defensive
        return None


def _validate_question(q: Any, index: int) -> Optional[str]:
    if not isinstance(q, dict):
        return f"questions[{index}] must be an object"
    qid = q.get("id")
    if not isinstance(qid, str) or not qid.strip():
        return f"questions[{index}].id must be a non-empty string"
    prompt = q.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return f"questions[{index}].prompt must be a non-empty string"
    header = q.get("header")
    if header is not None and not isinstance(header, str):
        return f"questions[{index}].header must be a string if present"
    if isinstance(header, str) and len(header) > 12:
        return (
            f"questions[{index}].header must be at most 12 characters "
            "(use a 1-3 word chip label)"
        )
    options = q.get("options", [])
    if options is not None and not isinstance(options, list):
        return f"questions[{index}].options must be an array if present"
    option_list = options or []
    if option_list:
        if len(option_list) < 2:
            return (
                f"questions[{index}].options must contain 2-4 mutually "
                "exclusive choices (omit the field entirely for a "
                "free-text question; a single-option question is a "
                "confirmation and is not supported)"
            )
        if len(option_list) > 4:
            return (
                f"questions[{index}].options must contain at most 4 "
                "choices"
            )
    for j, opt in enumerate(option_list):
        if not isinstance(opt, dict):
            return f"questions[{index}].options[{j}] must be an object"
        if not isinstance(opt.get("id"), str) or not opt["id"].strip():
            return f"questions[{index}].options[{j}].id must be a non-empty string"
        if not isinstance(opt.get("label"), str):
            return f"questions[{index}].options[{j}].label must be a string"
        description = opt.get("description")
        if description is None or not isinstance(description, str) or not description.strip():
            return (
                f"questions[{index}].options[{j}].description must be a "
                "non-empty string (one short sentence explaining the choice)"
            )
    return None


def _error(
    call: ToolCall,
    message: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=message,
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["AskUserQuestionTool"]
