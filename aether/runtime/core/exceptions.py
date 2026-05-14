"""Runtime control-flow exceptions.

Holds exception types that act as **control-flow signals** rather than
ordinary errors.  The defining trait of every type in this module is
that it subclasses :class:`BaseException` (not :class:`Exception`) so
broad ``except Exception:`` clauses scattered throughout providers and
middleware do not accidentally swallow the signal — the same idiom
Python uses for :class:`KeyboardInterrupt` and :class:`SystemExit`.

:class:`EngineInterrupted` crosses the stream-callback / provider /
run-loop boundary the instant a user requests cancellation.
"""

from __future__ import annotations


class EngineInterrupted(BaseException):
    """Raised when a user-interrupt is observed mid-flight.

    Subclasses :class:`BaseException` deliberately (mirrors
    :class:`KeyboardInterrupt`): the engine has dozens of
    ``except Exception:`` clauses in provider / middleware code paths
    that exist to swallow transient API / parse errors — they MUST NOT
    catch an interrupt signal.  Code that wants to act on this
    exception writes an explicit ``except EngineInterrupted`` clause.

    Carries enough context for the run loop to package the cancellation
    into ``EngineResult.metadata["interrupt"]`` without the upstream
    caller having to reach back into the stream buffer:

    * ``reason`` — human-readable label (default ``"user-interrupt"``).
      Forwarded to ``EngineResult.metadata["interrupt"]["reason"]``.
    * ``partial_text`` — whatever assistant text had been streamed up
      to the moment the flag was observed.  Empty string is fine when
      the flag flipped before the first delta.
    * ``was_in_tool_call`` — ``True`` when the interrupt arrived while
      a tool was executing rather than during the LLM stream. The run
      loop uses this to pick between the ``[Request interrupted by
      user]`` and ``[Request interrupted by user for tool use]``
      markers.
    """

    __slots__ = ("reason", "partial_text", "was_in_tool_call")

    def __init__(
        self,
        reason: str = "user-interrupt",
        *,
        partial_text: str = "",
        was_in_tool_call: bool = False,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.partial_text = partial_text
        self.was_in_tool_call = was_in_tool_call


__all__ = ["EngineInterrupted"]
