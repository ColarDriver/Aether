import { Writable } from 'node:stream'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const BSU = '\x1b[?2026h'
const ESU = '\x1b[?2026l'

interface FakeStdout extends NodeJS.WriteStream {
  writes: string[]
}

function fakeStdout(): FakeStdout {
  const writes: string[] = []
  const stream = new Writable({
    write(chunk, _encoding, callback) {
      writes.push(chunk.toString())
      callback()
    }
  }) as FakeStdout
  stream.writes = writes
  Object.assign(stream, {
    isTTY: true,
    columns: 80,
    rows: 24
  })
  return stream
}

async function importModule() {
  vi.resetModules()
  return import('../lib/atomicSyncStdout.js')
}

describe('atomicSyncStdout', () => {
  const originalEnv = { ...process.env }

  beforeEach(() => {
    process.env = {}
  })

  afterEach(() => {
    process.env = { ...originalEnv }
    vi.restoreAllMocks()
  })

  it('coalesces BSU, content, and ESU into one write on supported terminals', async () => {
    process.env.TERM_PROGRAM = 'WezTerm'
    const { wrapStdoutWithAtomicSync } = await importModule()
    const inner = fakeStdout()
    const stdout = wrapStdoutWithAtomicSync(inner)

    stdout.write(BSU)
    stdout.write('hello')
    stdout.write('world')
    stdout.write(ESU)

    expect(inner.writes).toEqual([`${BSU}helloworld${ESU}`])
  })

  it('drops BSU and ESU on unsupported terminals while passing content through', async () => {
    process.env.TMUX = '/tmp/tmux-1000/default,123,0'
    const { wrapStdoutWithAtomicSync } = await importModule()
    const inner = fakeStdout()
    const stdout = wrapStdoutWithAtomicSync(inner)

    stdout.write(BSU)
    stdout.write('payload')
    stdout.write(ESU)

    expect(inner.writes).toEqual(['payload'])
  })

  it('passes ordinary writes through outside synchronized windows', async () => {
    process.env.TERM = 'xterm-kitty'
    const { wrapStdoutWithAtomicSync } = await importModule()
    const inner = fakeStdout()
    const stdout = wrapStdoutWithAtomicSync(inner)

    stdout.write('plain')

    expect(inner.writes).toEqual(['plain'])
  })

  it('preserves stdout properties and methods through the proxy', async () => {
    process.env.TERM_PROGRAM = 'iTerm.app'
    const { wrapStdoutWithAtomicSync } = await importModule()
    const inner = fakeStdout()
    const stdout = wrapStdoutWithAtomicSync(inner)

    expect(stdout.isTTY).toBe(true)
    expect(stdout.columns).toBe(80)
    expect(stdout.rows).toBe(24)

    const listener = vi.fn()
    stdout.on('resize', listener)
    inner.emit('resize')

    expect(listener).toHaveBeenCalledOnce()
  })

  it('detects known synchronized-output terminal environments', async () => {
    process.env.VTE_VERSION = '6800'
    const { isSynchronizedOutputSupported } = await importModule()

    expect(isSynchronizedOutputSupported()).toBe(true)
  })
})
