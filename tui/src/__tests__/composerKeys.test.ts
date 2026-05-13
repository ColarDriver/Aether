import { beforeEach, describe, expect, it } from 'vitest'

import { composerActions, composerState } from '../store/composerStore.js'

describe('composer queued messages', () => {
  beforeEach(() => {
    composerActions.resetForTests()
  })

  it('enqueue clears the draft and stores the entry on the queue', () => {
    composerActions.setDraft('hello')
    composerActions.enqueue('hello')

    const state = composerState.get()
    expect(state.draft).toBe('')
    expect(state.queued).toEqual(['hello'])
  })

  it('popQueued returns the most recently queued entry into the draft', () => {
    composerActions.enqueue('first')
    composerActions.enqueue('second')

    const popped = composerActions.popQueued()
    expect(popped).toBe('second')

    const state = composerState.get()
    expect(state.draft).toBe('second')
    expect(state.queued).toEqual(['first'])
  })

  it('shiftQueued returns the FRONT of the queue without touching the draft', () => {
    composerActions.enqueue('first')
    composerActions.enqueue('second')

    const head = composerActions.shiftQueued()
    expect(head).toBe('first')

    const state = composerState.get()
    expect(state.draft).toBe('')
    expect(state.queued).toEqual(['second'])
  })

  it('clearQueued resets the queue', () => {
    composerActions.enqueue('a')
    composerActions.enqueue('b')
    composerActions.clearQueued()
    expect(composerState.get().queued).toEqual([])
  })

  it('attachHistoryFile merges existing entries with in-memory history', () => {
    const log: string[] = []
    composerActions.commit('alpha')
    composerActions.attachHistoryFile({
      path: ':memory:',
      load: () => ['gamma', 'beta'],
      append: (entry: string) => {
        log.push(entry)
      }
    })

    const history = composerState.get().history
    expect(history).toContain('alpha')
    expect(history).toContain('beta')
    expect(history).toContain('gamma')

    composerActions.commit('delta')
    expect(log).toEqual(['delta'])
  })
})
