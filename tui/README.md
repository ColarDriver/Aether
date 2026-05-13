# TUI

TypeScript/Ink client scaffold for the Aether gateway.

This package is intentionally separate from the Python package so Node
dependencies, build output, and future terminal UI code do not become Python
package data.

## Development

```bash
npm install
npm run type-check
npm test
```

Run the smoke entrypoint from this directory:

```bash
AETHER_PYTHON_SRC_ROOT="$PWD/.." npm start
```

The entrypoint starts the Python gateway over stdio, prints `gateway.ready`,
and keeps the process alive until `Ctrl+C` so shutdown behavior can be checked.
