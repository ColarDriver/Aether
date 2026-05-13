import { appendFileSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createHistoryFile } from '../lib/fileHistory.js'

let tmpDir: string

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'aether-tui-history-'))
})

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true })
})

describe('fileHistory', () => {
  it('returns empty array when the file does not exist yet', () => {
    const file = createHistoryFile(join(tmpDir, 'history'))
    expect(file.load()).toEqual([])
  })

  it('appends entries as JSON-encoded NDJSON rows', () => {
    const path = join(tmpDir, 'nested', 'history')
    const file = createHistoryFile(path)
    file.append('hello world')
    file.append('multi\nline entry')

    const raw = readFileSync(path, 'utf8')
    const lines = raw.split('\n').filter(Boolean)
    expect(lines).toEqual([
      JSON.stringify('hello world'),
      JSON.stringify('multi\nline entry')
    ])
  })

  it('round-trips entries through load()', () => {
    const file = createHistoryFile(join(tmpDir, 'history'))
    file.append('first')
    file.append('second')
    file.append('third')

    const loaded = file.load()
    expect(loaded).toEqual(['first', 'second', 'third'])
  })

  it('skips malformed lines instead of crashing', () => {
    const path = join(tmpDir, 'history')
    const file = createHistoryFile(path)
    file.append('valid')
    // Manually inject a bad line.
    appendFileSync(path, 'not-json\n', 'utf8')
    file.append('valid-again')

    expect(file.load()).toEqual(['valid', 'valid-again'])
  })

  it('skips empty / whitespace-only entries on append', () => {
    const file = createHistoryFile(join(tmpDir, 'history'))
    file.append('')
    file.append('   ')
    file.append('keep')
    expect(file.load()).toEqual(['keep'])
  })
})
