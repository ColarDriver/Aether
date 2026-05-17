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
                "ambiguous requirement). Each question may be free-text "
                "or multiple-choice. In plan mode, use this only to "
                "clarify requirements or choose between approaches; do "
                "not ask whether the plan is approved or whether to "
                "proceed. Use exit_plan_mode for plan approval. Not "
                "allowed inside subagent contexts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Stable identifier used to key the answer.",
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "Question text shown to the user.",
                                },
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "label": {"type": "string"},
                                        },
                                        "required": ["id", "label"],
                                    },
                                },
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

        try:
            answers = prompter.ask_questions(validated, timeout=timeout_seconds)
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

        formatted = self._format_answers(validated, answers)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=formatted,
            is_error=False,
            metadata={
                "question_count": len(validated),
                "answer_count": sum(1 for q in validated if q["id"] in answers),
            },
        )

    @staticmethod
    def _format_answers(
        questions: List[Dict[str, Any]], answers: Mapping[str, Any]
    ) -> str:
        lines = ["# User responses", ""]
        for q in questions:
            qid = q["id"]
            ans = answers.get(qid, "(no response)")
            lines.append(f"## {q['prompt']}")
            if isinstance(ans, list):
                if not ans:
                    lines.append("- (none selected)")
                else:
                    for item in ans:
                        lines.append(f"- {item}")
            else:
                lines.append(f"{ans}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def _validate_question(q: Any, index: int) -> Optional[str]:
    if not isinstance(q, dict):
        return f"questions[{index}] must be an object"
    qid = q.get("id")
    if not isinstance(qid, str) or not qid.strip():
        return f"questions[{index}].id must be a non-empty string"
    prompt = q.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return f"questions[{index}].prompt must be a non-empty string"
    options = q.get("options", [])
    if options is not None and not isinstance(options, list):
        return f"questions[{index}].options must be an array if present"
    for j, opt in enumerate(options or []):
        if not isinstance(opt, dict):
            return f"questions[{index}].options[{j}] must be an object"
        if not isinstance(opt.get("id"), str) or not opt["id"].strip():
            return f"questions[{index}].options[{j}].id must be a non-empty string"
        if not isinstance(opt.get("label"), str):
            return f"questions[{index}].options[{j}].label must be a string"
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
