"""Pin the PR 3 acceptance check: gateway code does not import UI deps.

The gateway is meant to be the Python backend, not the TUI.  Direct
imports of ``prompt_toolkit`` or ``rich`` in ``aether/gateway/`` would
betray a coupling to the UI that PR 9 plans to remove.  A lazy
``aether.cli.*`` import inside a handler body is allowed (PR 3 wraps
those modules) but a top-level ``from rich.table import Table`` etc.
is not.

This test is a simple grep over the gateway tree.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


_GATEWAY_DIR = Path(__file__).resolve().parents[2] / "gateway"

_FORBIDDEN_PATTERNS = [
    # Top-level direct imports.  Use ^…^ anchors so substring matches
    # inside docstrings or comment text don't trip the test.
    re.compile(r"^\s*import\s+prompt_toolkit", re.MULTILINE),
    re.compile(r"^\s*from\s+prompt_toolkit", re.MULTILINE),
    re.compile(r"^\s*import\s+rich(\s|$|\.)", re.MULTILINE),
    re.compile(r"^\s*from\s+rich\b", re.MULTILINE),
]


class GatewayHasNoUIImports(unittest.TestCase):
    def test_no_direct_imports_in_any_gateway_python_file(self) -> None:
        offenders: list[str] = []
        for path in _GATEWAY_DIR.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in _FORBIDDEN_PATTERNS:
                for match in pattern.finditer(text):
                    # ignore lazy imports inside def bodies — match.group()
                    # already filters by leading whitespace = 0 (top level).
                    # The MULTILINE anchor allows leading whitespace in
                    # regex \s*, so we check the line manually:
                    line = text[: match.start()].count("\n") + 1
                    snippet = text.splitlines()[line - 1].rstrip()
                    if snippet.lstrip() != snippet:
                        # Indented import — used inside a function body
                        # (lazy import).  That's allowed per the doc.
                        continue
                    offenders.append(
                        f"{path.relative_to(_GATEWAY_DIR.parent)}:{line}: {snippet}"
                    )

        if offenders:
            self.fail(
                "gateway code must not directly import prompt_toolkit or rich:\n  "
                + "\n  ".join(offenders)
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
