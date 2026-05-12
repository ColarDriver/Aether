"""Sandbox abstractions used by aether tools.

Marker file that makes ``aether.sandbox`` a proper Python package.

Sub-modules (``sandbox``, ``search``, ``exceptions``, …) are intentionally
*not* eagerly imported here: some of them still depend on optional or
in-progress modules (``aether.config.get_app_config``,
``aether.reflection.resolve_class``, ``deerflow.sandbox``), so importing
the package itself must not require those.  Import the concrete symbols
you need directly, e.g. ``from aether.sandbox.sandbox import Sandbox``.
"""
