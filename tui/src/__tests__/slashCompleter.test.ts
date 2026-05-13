import { describe, expect, it } from 'vitest'

import {
  applyCompletion,
  buildCompleterState,
  leadingSlashToken,
  rankSlashMatches
} from '../lib/slashCompleter.js'
import type { SlashCommandInfo } from '../gatewayTypes.js'

const CATALOG: SlashCommandInfo[] = [
  { name: '/help', description: 'show help', category: 'local' },
  { name: '/exit', description: 'quit', category: 'local' },
  { name: '/resume', description: 'resume a session', category: 'remote' },
  { name: '/refresh', description: 'redraw', category: 'local' },
  { name: '/model', description: 'switch model', category: 'session' }
]

describe('leadingSlashToken', () => {
  it('returns null for non-slash drafts and multi-line drafts', () => {
    expect(leadingSlashToken('hello')).toBeNull()
    expect(leadingSlashToken('/help\nmore')).toBeNull()
  })

  it('returns the leading word for a single-line slash draft', () => {
    expect(leadingSlashToken('/help')).toBe('/help')
    expect(leadingSlashToken('/resume abc')).toBe('/resume')
    expect(leadingSlashToken('/model gpt-5.4')).toBe('/model')
  })
})

describe('rankSlashMatches', () => {
  it('prefix match comes before substring match', () => {
    const matches = rankSlashMatches('/re', CATALOG)
    expect(matches.map((m) => m.name)).toEqual(['/refresh', '/resume'])
  })

  it('returns the entire catalog for a bare /', () => {
    expect(rankSlashMatches('/', CATALOG).length).toBe(CATALOG.length)
  })

  it('falls back to substring match when no prefix matches', () => {
    const matches = rankSlashMatches('/odel', CATALOG)
    expect(matches.map((m) => m.name)).toContain('/model')
  })

  it('returns an empty list when token does not start with /', () => {
    expect(rankSlashMatches('he', CATALOG)).toEqual([])
  })
})

describe('buildCompleterState', () => {
  it('is inactive when the draft has no leading slash', () => {
    const state = buildCompleterState('hello', CATALOG)
    expect(state.active).toBe(false)
    expect(state.matches).toEqual([])
  })

  it('is active when matches are available', () => {
    const state = buildCompleterState('/h', CATALOG)
    expect(state.active).toBe(true)
    expect(state.matches[0]?.name).toBe('/help')
    expect(state.index).toBe(0)
  })
})

describe('applyCompletion', () => {
  it('replaces the leading token only, preserving args', () => {
    const command = CATALOG[2]!
    expect(applyCompletion('/re abc', command)).toBe('/resume abc')
  })

  it('returns the bare command name when the draft is just the prefix', () => {
    const command = CATALOG[0]!
    expect(applyCompletion('/he', command)).toBe('/help')
  })
})
