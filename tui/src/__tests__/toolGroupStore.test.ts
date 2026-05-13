import { beforeEach, describe, expect, it } from 'vitest'

import { buildHeadline, toolGroupActions, toolGroupState } from '../store/toolGroupStore.js'

describe('toolGroupStore', () => {
  beforeEach(() => {
    toolGroupActions.resetForTests()
  })

  it('explore tools accumulate into the active group', () => {
    const a = toolGroupActions.startCall({
      toolCallId: 'c1',
      toolName: 'read_file',
      args: { path: 'foo.txt' },
      iteration: 1
    })
    const b = toolGroupActions.startCall({
      toolCallId: 'c2',
      toolName: 'Grep',
      args: { pattern: 'x', path: 'src/' },
      iteration: 1
    })
    expect(a.isExplore).toBe(true)
    expect(b.isExplore).toBe(true)

    const active = toolGroupState.get().active
    expect(active).not.toBeNull()
    expect(active?.totalCalls).toBe(2)
    expect(active?.entries.map((entry) => entry.toolName)).toEqual(['read_file', 'Grep'])
    expect(active?.counts.read).toBe(1)
    expect(active?.counts.search).toBe(1)
  })

  it('non-explore tools flush the active group and never enter it', () => {
    toolGroupActions.startCall({
      toolCallId: 'c1',
      toolName: 'read_file',
      args: { path: 'foo.txt' },
      iteration: 1
    })
    expect(toolGroupState.get().active).not.toBeNull()

    const dispatch = toolGroupActions.startCall({
      toolCallId: 'c2',
      toolName: 'shell',
      args: { command: 'ls' },
      iteration: 1
    })
    expect(dispatch.isExplore).toBe(false)
    expect(toolGroupState.get().active).toBeNull()
    expect(toolGroupState.get().lastFlushed?.totalCalls).toBe(1)
    expect(toolGroupState.get().lastFlushed?.entries[0]?.toolName).toBe('read_file')
  })

  it('finishCall marks the matching entry as errored', () => {
    toolGroupActions.startCall({
      toolCallId: 'c1',
      toolName: 'read_file',
      args: { path: 'foo.txt' },
      iteration: 1
    })
    toolGroupActions.startCall({
      toolCallId: 'c2',
      toolName: 'Grep',
      args: { pattern: 'x', path: 'src/' },
      iteration: 1
    })
    toolGroupActions.finishCall({ toolCallId: 'c2', isError: true })

    const active = toolGroupState.get().active
    expect(active?.entries.find((e) => e.toolCallId === 'c2')?.isError).toBe(true)
    expect(active?.hasError).toBe(true)
    expect(active?.inFlight).toBe(1)
  })

  it('flushActive returns null when there is nothing to flush and clears the slot', () => {
    expect(toolGroupActions.flushActive()).toBeNull()

    toolGroupActions.startCall({
      toolCallId: 'c1',
      toolName: 'read_file',
      args: { path: 'foo.txt' },
      iteration: 1
    })
    const flushed = toolGroupActions.flushActive()
    expect(flushed).not.toBeNull()
    expect(flushed?.entries[0]?.toolName).toBe('read_file')
    expect(toolGroupState.get().active).toBeNull()
    expect(toolGroupState.get().lastFlushed?.id).toBe(flushed?.id)
  })

  it('discardActive drops the group without producing a flush', () => {
    toolGroupActions.startCall({
      toolCallId: 'c1',
      toolName: 'read_file',
      args: { path: 'foo.txt' },
      iteration: 1
    })
    toolGroupActions.discardActive()
    expect(toolGroupState.get().active).toBeNull()
    expect(toolGroupState.get().lastFlushed).toBeNull()
  })

  it('beginIteration is a flush alias', () => {
    toolGroupActions.startCall({
      toolCallId: 'c1',
      toolName: 'read_file',
      args: { path: 'foo.txt' },
      iteration: 1
    })
    const flushed = toolGroupActions.beginIteration()
    expect(flushed?.entries).toHaveLength(1)
    expect(toolGroupState.get().active).toBeNull()
  })
})

describe('buildHeadline', () => {
  it('emits one segment per category with first-position capitalised verb', () => {
    toolGroupActions.resetForTests()
    toolGroupActions.startCall({
      toolCallId: 'c1',
      toolName: 'read_file',
      args: { path: 'foo' },
      iteration: 1
    })
    toolGroupActions.startCall({
      toolCallId: 'c2',
      toolName: 'read_file',
      args: { path: 'bar' },
      iteration: 1
    })
    toolGroupActions.startCall({
      toolCallId: 'c3',
      toolName: 'Grep',
      args: { pattern: 'x' },
      iteration: 1
    })
    const flushed = toolGroupActions.flushActive()
    expect(flushed).not.toBeNull()
    const segments = buildHeadline(flushed!, false)
    expect(segments).toEqual([
      { category: 'search', verb: 'Searched for', count: 1, noun: 'pattern' },
      { category: 'read', verb: 'read', count: 2, noun: 'files' }
    ])
  })
})
