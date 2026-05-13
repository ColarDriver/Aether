import { describe, expect, it } from 'vitest'

import { applyCliArgs, parseArgs } from '../lib/cliArgs.js'

describe('cli args', () => {
  it('parses long value and boolean flags', () => {
    const flags = parseArgs([
      '--provider',
      'claude',
      '--model=gpt-5.4',
      '--verbose',
      '--resume'
    ])

    expect(flags.get('provider')).toBe('claude')
    expect(flags.get('model')).toBe('gpt-5.4')
    expect(flags.get('verbose')).toBe(true)
    expect(flags.get('resume')).toBe(true)
  })

  it('parses Python CLI short aliases', () => {
    const env: NodeJS.ProcessEnv = {}

    applyCliArgs(['-p', 'openai', '-m', 'gpt-4o', '-v'], env)

    expect(env.AETHER_PROVIDER).toBe('openai')
    expect(env.AETHER_MODEL).toBe('gpt-4o')
    expect(env.AETHER_VERBOSE).toBe('1')
  })

  it('does not overwrite existing env values', () => {
    const env: NodeJS.ProcessEnv = { AETHER_PROVIDER: 'codex' }

    applyCliArgs(['--provider', 'claude'], env)

    expect(env.AETHER_PROVIDER).toBe('codex')
  })

  it('maps resume without an id to the picker sentinel', () => {
    const env: NodeJS.ProcessEnv = {}

    applyCliArgs(['--resume'], env)

    expect(env.AETHER_RESUME).toBe('1')
  })
})
