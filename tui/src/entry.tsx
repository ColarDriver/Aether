import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { render } from 'ink'

import { applyCliArgs } from './lib/cliArgs.js'

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repoRoot = resolve(packageRoot, '..')
const workspaceCwd = resolve(
  process.env.AETHER_WORKSPACE_CWD || process.env.INIT_CWD || repoRoot
)

// Translate CLI flags → env vars before mounting <App />. The TS entry is
// invoked by the Python launcher with the same flag surface as the legacy
// `aether` CLI (see `aether/cli/main.py`); we accept them directly here so a
// developer running `npm start -- --temperature 0.2` reaches feature parity
// with `aether --temperature 0.2`. Unknown flags pass through silently —
// downstream tooling may rely on them.
applyCliArgs(process.argv.slice(2))

const [{ App }, { GatewayClient }] = await Promise.all([
  import('./app.js'),
  import('./gatewayClient.js')
])
const client = new GatewayClient()

try {
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    process.stderr.write('aether requires an interactive terminal\n')
    process.exit(2)
  }

  const instance = render(
    <App client={client} repoRoot={repoRoot} workspaceCwd={workspaceCwd} />,
    {
      exitOnCtrlC: false
    }
  )
  await instance.waitUntilExit()
} catch (error) {
  const message = error instanceof Error ? error.message : String(error)
  process.stderr.write(`${message}\n`)
  process.exitCode = 1
} finally {
  await client.stop().catch(() => undefined)
}
