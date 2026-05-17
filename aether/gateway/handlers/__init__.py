"""RPC method handlers registered with the gateway dispatcher.

The submodules in this package each expose a ``register()`` function
that binds their handlers to the dispatcher's ``@method`` registry.
:func:`register_handler_methods` is the single entry point: import
the package and call it once at gateway boot, or after a test has
wiped the registry via
:func:`aether.gateway.dispatcher.reset_dispatcher_for_tests`.

Importing the package itself does NOT register anything — that's a
deliberate design choice to keep import side effects out of the
handler modules and make the registration order explicit.
"""

from __future__ import annotations

from aether.gateway.handlers import (
    agent_methods,
    commands_methods,
    plan_methods,
    prefs_methods,
    providers_methods,
    response_methods,
    session_methods,
    tools_methods,
)


def register_handler_methods() -> None:
    """Register every ``@method`` handler in this package on the dispatcher.

    Idempotent.  Safe to call after
    :func:`aether.gateway.dispatcher.reset_dispatcher_for_tests` to
    restore the production method catalog.
    """
    session_methods.register()
    prefs_methods.register()
    providers_methods.register()
    commands_methods.register()
    plan_methods.register()
    agent_methods.register()
    response_methods.register()
    tools_methods.register()


__all__ = [
    "agent_methods",
    "commands_methods",
    "plan_methods",
    "prefs_methods",
    "providers_methods",
    "register_handler_methods",
    "response_methods",
    "session_methods",
    "tools_methods",
]
