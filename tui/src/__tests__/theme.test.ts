import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { theme } from '../lib/theme.js'

const ENV_KEYS = ['AETHER_ASCII', 'NO_COLOR', 'TERM', 'LC_ALL', 'LC_CTYPE', 'LANG']
let snapshot: Record<string, string | undefined>

beforeEach(() => {
  snapshot = {}
  for (const key of ENV_KEYS) {
    snapshot[key] = process.env[key]
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

describe('theme.icon', () => {
  it('returns Unicode glyphs by default', () => {
    delete process.env.AETHER_ASCII
    process.env.LC_ALL = 'C.UTF-8'
    expect(theme.icon('assistant')).toBe('●')
  })

  it('falls back to ASCII when AETHER_ASCII=1', () => {
    process.env.AETHER_ASCII = '1'
    expect(theme.icon('assistant')).toBe('*')
    expect(theme.icon('user')).toBe('>')
    expect(theme.icon('warn')).toBe('!')
  })

  it('returns empty string for unknown icons', () => {
    expect(theme.icon('definitely-not-an-icon')).toBe('')
  })
})

describe('theme.color / colorProps', () => {
  it('returns hex strings when colour is enabled', () => {
    delete process.env.NO_COLOR
    delete process.env.TERM
    expect(theme.color('brand')).toBe('#7C5CFF')
    expect(theme.colorProps('error')).toEqual({ color: '#EF4444' })
  })

  it('returns undefined / {} when NO_COLOR is set', () => {
    process.env.NO_COLOR = '1'
    expect(theme.color('brand')).toBeUndefined()
    expect(theme.colorProps('brand')).toEqual({})
  })

  it('returns undefined when TERM=dumb', () => {
    process.env.TERM = 'dumb'
    expect(theme.color('brand')).toBeUndefined()
  })
})

describe('theme.isUnicodeAllowed', () => {
  it('respects AETHER_ASCII=1', () => {
    process.env.AETHER_ASCII = '1'
    expect(theme.isUnicodeAllowed()).toBe(false)
  })

  it('returns true for utf-8 locales', () => {
    delete process.env.AETHER_ASCII
    process.env.LANG = 'en_US.UTF-8'
    expect(theme.isUnicodeAllowed()).toBe(true)
  })

  it('returns false for explicit non-utf locales', () => {
    delete process.env.AETHER_ASCII
    process.env.LC_ALL = 'C'
    expect(theme.isUnicodeAllowed()).toBe(false)
  })
})
