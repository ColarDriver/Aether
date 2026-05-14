import { describe, expect, it } from 'vitest'

import {
  CATEGORY_ORDER,
  EXPLORE_CATEGORIES,
  categoryFor,
  hintForCall,
  nounForCategory,
  verbForCategory,
  verbForTool
} from '../lib/toolCategory.js'

describe('categoryFor', () => {
  it('hits the explicit name table for canonical aliases', () => {
    expect(categoryFor('read_file')).toBe('read')
    expect(categoryFor('Grep')).toBe('search')
    expect(categoryFor('shell')).toBe('bash')
    expect(categoryFor('Edit')).toBe('edit')
  })

  it('strips MCP / namespace prefixes before bucketing', () => {
    expect(categoryFor('mcp__filesystem__read_file')).toBe('read')
    expect(categoryFor('aether.tools.list_directory')).toBe('list')
    expect(categoryFor('ns:Bash')).toBe('bash')
  })

  it('falls back to MCP only when canonical name does not match', () => {
    expect(categoryFor('mcp__server__do_some_thing')).toBe('mcp')
  })

  it('uses prefix heuristics for unrecognised names', () => {
    expect(categoryFor('read_my_thing')).toBe('read')
    expect(categoryFor('write_thing')).toBe('write')
    expect(categoryFor('delete_one')).toBe('edit')
    expect(categoryFor('totally_unknown')).toBe('other')
  })
})

describe('EXPLORE_CATEGORIES', () => {
  it('contains exactly read/list/search/write/edit', () => {
    expect(Array.from(EXPLORE_CATEGORIES).sort()).toEqual(
      ['edit', 'list', 'read', 'search', 'write'].sort()
    )
  })
})

describe('hintForCall', () => {
  it('formats bash commands with a $ prefix', () => {
    expect(hintForCall('shell', { command: 'ls -la /tmp' })).toBe('$ ls -la /tmp')
  })

  it('formats search hints with pattern + path', () => {
    expect(hintForCall('Grep', { pattern: 'foo', path: 'src/' })).toBe('"foo" in src/')
  })

  it('falls back to path / file keys for read/write/edit', () => {
    expect(hintForCall('read_file', { path: 'foo.txt' })).toBe('foo.txt')
    expect(hintForCall('write_file', { filename: 'bar.json' })).toBe('bar.json')
  })

  it('shows read_file line ranges when offset or limit is present', () => {
    expect(hintForCall('read_file', { path: 'agent.py', offset: 1398, limit: 30 })).toBe(
      'agent.py · lines 1398-1427'
    )
    expect(hintForCall('read_file', { path: 'agent.py', offset: 1855 })).toBe(
      'agent.py · from line 1855'
    )
    expect(hintForCall('read_file', { path: 'agent.py', limit: 20 })).toBe(
      'agent.py · lines 1-20'
    )
  })

  it('returns the URL for web fetches', () => {
    expect(hintForCall('WebFetch', { url: 'https://example.com' })).toBe('https://example.com')
  })

  it('truncates long values', () => {
    const long = 'x'.repeat(200)
    expect(hintForCall('read_file', { path: long }).length).toBeLessThanOrEqual(81)
  })

  it('returns empty string when args is empty / missing', () => {
    expect(hintForCall('shell', undefined)).toBe('')
    expect(hintForCall('shell', {})).toBe('')
  })
})

describe('verb / noun tables', () => {
  it('returns past + present forms for every category', () => {
    for (const category of CATEGORY_ORDER) {
      expect(verbForCategory(category, false)).toHaveLength(2)
      expect(verbForCategory(category, true)).toHaveLength(2)
      expect(nounForCategory(category)).toHaveLength(2)
    }
  })

  it('verbForTool maps via category table', () => {
    expect(verbForTool('shell')).toBe('Ran')
    expect(verbForTool('read_file')).toBe('Read')
  })
})
