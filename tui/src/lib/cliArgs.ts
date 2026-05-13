export type ParsedFlags = Map<string, string | boolean>

const VALUE_FLAGS = new Set([
  'provider',
  'model',
  'api-key',
  'base-url',
  'system',
  'system-file',
  'session',
  'max-iterations',
  'temperature',
  'max-tokens',
  'log-level'
])

const OPTIONAL_VALUE_FLAGS = new Set(['resume'])

const SHORT_ALIASES = new Map([
  ['p', 'provider'],
  ['m', 'model'],
  ['v', 'verbose']
])

/**
 * Translate CLI flags to env vars before mounting the Ink app. This mirrors
 * `aether/cli/main.py` for direct TS development launches like
 * `npm start -- -p openai -m gpt-5.4`.
 */
export function applyCliArgs(
  argv: string[],
  env: NodeJS.ProcessEnv = process.env
): void {
  const flags = parseArgs(argv)
  setEnvFromFlag(flags, 'provider', 'AETHER_PROVIDER', env)
  setEnvFromFlag(flags, 'model', 'AETHER_MODEL', env)
  setEnvFromFlag(flags, 'api-key', 'AETHER_API_KEY', env)
  setEnvFromFlag(flags, 'base-url', 'AETHER_BASE_URL', env)
  setEnvFromFlag(flags, 'system', 'AETHER_SYSTEM', env)
  setEnvFromFlag(flags, 'system-file', 'AETHER_SYSTEM_FILE', env)
  setEnvFromFlag(flags, 'resume', 'AETHER_RESUME', env)
  setEnvFromFlag(flags, 'session', 'AETHER_SESSION_ID', env)
  setEnvFromFlag(flags, 'max-iterations', 'AETHER_MAX_ITERATIONS', env)
  setEnvFromFlag(flags, 'temperature', 'AETHER_TEMPERATURE', env)
  setEnvFromFlag(flags, 'max-tokens', 'AETHER_MAX_TOKENS', env)
  setEnvFromFlag(flags, 'log-level', 'AETHER_LOG_LEVEL', env)
  setEnvBoolFromFlag(flags, 'verbose', 'AETHER_VERBOSE', env)
  setEnvBoolFromFlag(flags, 'no-banner', 'AETHER_NO_BANNER', env)
  setEnvBoolFromFlag(flags, 'no-builtin-tools', 'AETHER_NO_BUILTIN_TOOLS', env)
}

export function parseArgs(argv: string[]): ParsedFlags {
  const flags: ParsedFlags = new Map()
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i]
    if (!token || token === '--') {
      break
    }
    if (token.startsWith('--')) {
      const body = token.slice(2)
      if (body.includes('=')) {
        const eq = body.indexOf('=')
        flags.set(body.slice(0, eq), body.slice(eq + 1))
        continue
      }
      if (VALUE_FLAGS.has(body)) {
        const next = argv[i + 1]
        if (next !== undefined) {
          flags.set(body, next)
          i++
        }
        continue
      }
      if (OPTIONAL_VALUE_FLAGS.has(body)) {
        const next = argv[i + 1]
        if (next !== undefined && !next.startsWith('-')) {
          flags.set(body, next)
          i++
        } else {
          flags.set(body, true)
        }
        continue
      }
      flags.set(body, true)
      continue
    }
    if (token.startsWith('-') && token.length > 1) {
      const alias = SHORT_ALIASES.get(token.slice(1))
      if (!alias) {
        continue
      }
      if (VALUE_FLAGS.has(alias)) {
        const next = argv[i + 1]
        if (next !== undefined) {
          flags.set(alias, next)
          i++
        }
      } else {
        flags.set(alias, true)
      }
    }
  }
  return flags
}

function setEnvFromFlag(
  flags: ParsedFlags,
  name: string,
  envKey: string,
  env: NodeJS.ProcessEnv
): void {
  if (env[envKey]) {
    return
  }
  const value = flags.get(name)
  if (typeof value === 'string') {
    env[envKey] = value
  } else if (value === true) {
    env[envKey] = '1'
  }
}

function setEnvBoolFromFlag(
  flags: ParsedFlags,
  name: string,
  envKey: string,
  env: NodeJS.ProcessEnv
): void {
  if (env[envKey]) {
    return
  }
  if (flags.has(name)) {
    env[envKey] = '1'
  }
}
