import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { envBaseUrl, resolveDiscoveredBaseUrl } from '../lib/sessionBaseUrl.js'

const ENV_KEYS = ['AETHER_BASE_URL', 'OPENAI_BASE_URL', 'ANTHROPIC_BASE_URL']
let snapshot: Record<string, string | undefined>

beforeEach(() => {
  snapshot = {}
  for (const key of ENV_KEYS) {
    snapshot[key] = process.env[key]
    delete process.env[key]
  }
})

afterEach(() => {
  for (const key of ENV_KEYS) {
    if (snapshot[key] === undefined) {
      delete process.env[key]
    } else {
      process.env[key] = snapshot[key]
    }
  }
})

describe('sessionBaseUrl', () => {
  it('prefers the explicit AETHER_BASE_URL override', () => {
    process.env.AETHER_BASE_URL = 'http://override.test/v1'
    process.env.OPENAI_BASE_URL = 'http://openai.test/v1'
    expect(envBaseUrl('openai')).toBe('http://override.test/v1')
  })

  it('uses provider-specific env vars when no explicit override exists', () => {
    process.env.OPENAI_BASE_URL = 'http://openai.test/v1'
    process.env.ANTHROPIC_BASE_URL = 'http://anthropic.test'
    expect(envBaseUrl('openai')).toBe('http://openai.test/v1')
    expect(envBaseUrl('claude')).toBe('http://anthropic.test')
  })

  it('prefers suggested_base_url over raw discovery/base input', () => {
    expect(
      resolveDiscoveredBaseUrl('http://example.test', {
        base_url: 'http://example.test',
        suggested_base_url: 'http://example.test/v1'
      })
    ).toBe('http://example.test/v1')
  })
})
