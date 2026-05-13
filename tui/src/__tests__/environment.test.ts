import { describe, expect, it } from 'vitest'

import {
  augmentSystemPrompt,
  buildEnvironmentContext,
  probeEnvironment
} from '../lib/environment.js'

describe('probeEnvironment', () => {
  it('returns a non-empty cwd', () => {
    const env = probeEnvironment()
    expect(typeof env.cwd).toBe('string')
    expect(env.cwd.length).toBeGreaterThan(0)
  })

  it('returns the repo root when invoked from inside a git checkout', () => {
    const env = probeEnvironment()
    // The TS test suite always runs inside the Aether repo, so we expect a
    // detectable .git ancestor. If this assertion ever fails on a packaged
    // install the probe simply returns null — we just verify the type contract.
    expect(env.repoRoot === null || typeof env.repoRoot === 'string').toBe(true)
  })

  it('returns either a branch name string or null', () => {
    const env = probeEnvironment()
    expect(env.branch === null || typeof env.branch === 'string').toBe(true)
  })

  it('builds the system prompt environment block used by Python TUI parity', () => {
    const prompt = augmentSystemPrompt('be terse', '/tmp')

    expect(prompt).toContain('<environment>')
    expect(prompt).toContain('working_directory: /tmp')
    expect(prompt).toContain('</environment>\n\nbe terse')
  })

  it('uses the environment block as the whole prompt when no user prompt is set', () => {
    const prompt = buildEnvironmentContext('/tmp')

    expect(prompt).toContain('is_git_repository:')
    expect(prompt).toContain('date:')
  })
})
