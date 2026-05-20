import { Writable } from 'node:stream'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  buildShimmerFrame,
  isDirectWriteShimmerEnabled,
  startShimmerWriter
} from '../lib/shimmerWriter.js'
import { _resetClockForTests } from '../lib/animationClock.js'

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

describe('shimmerWriter', () => {
  const originalEnv = { ...process.env }

  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    process.env = { ...originalEnv }
    delete process.env.AETHER_SHIMMER_DIRECT_WRITE
    _resetClockForTests()
  })

  afterEach(() => {
    _resetClockForTests()
    process.env = { ...originalEnv }
    vi.useRealTimers()
  })

  it('is opt-in via AETHER_SHIMMER_DIRECT_WRITE', () => {
    expect(isDirectWriteShimmerEnabled()).toBe(false)
    process.env.AETHER_SHIMMER_DIRECT_WRITE = '1'
    expect(isDirectWriteShimmerEnabled()).toBe(true)
  })

  it('builds a cursor-preserving shimmer frame', () => {
    const frame = buildShimmerFrame(
      {
        row: 3,
        col: 4,
        label: 'Thinking',
        baseColor: '#64748B',
        highlightColor: '#E5E7EB'
      },
      0,
      true
    )

    expect(frame).toContain('\x1b[?2026h')
    expect(frame).toContain('\x1b7')
    expect(frame).toContain('\x1b[3;4H')
    expect(frame).toContain('\x1b[38;2;100;116;139m')
    expect(frame).toContain('\x1b[38;2;229;231;235m')
    expect(frame).toContain('\x1b8')
    expect(frame).toContain('\x1b[?2026l')
  })

  it('does not start when disabled', () => {
    const stdout = fakeStdout()

    const writer = startShimmerWriter({
      stdout,
      row: 1,
      col: 1,
      label: 'Thinking',
      baseColor: '#64748B',
      highlightColor: '#E5E7EB',
      intervalMs: 150
    })

    expect(writer).toBeNull()
    expect(stdout.writes).toEqual([])
  })

  it('writes frames through the provided stdout when enabled', () => {
    process.env.AETHER_SHIMMER_DIRECT_WRITE = '1'
    const stdout = fakeStdout()

    const writer = startShimmerWriter({
      stdout,
      row: 1,
      col: 2,
      label: 'Thinking',
      baseColor: '#64748B',
      highlightColor: '#E5E7EB',
      intervalMs: 150
    })

    expect(writer).not.toBeNull()
    expect(stdout.writes.length).toBe(1)
    expect(stdout.writes[0]).toContain('\x1b[1;2H')

    vi.advanceTimersByTime(150)
    expect(stdout.writes.length).toBeGreaterThan(1)

    writer?.stop()
    const count = stdout.writes.length
    vi.advanceTimersByTime(150)
    expect(stdout.writes.length).toBe(count)
  })
})
