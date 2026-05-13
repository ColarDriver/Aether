import { beforeEach, describe, expect, it } from 'vitest'

import { reasoningActions, reasoningState } from '../store/reasoningStore.js'

describe('reasoningStore', () => {
  beforeEach(() => {
    reasoningActions.resetForTests()
  })

  it('starts empty', () => {
    expect(reasoningState.get()).toEqual({ text: '', updatedAt: null })
  })

  it('appendDelta accumulates text and updates updatedAt', () => {
    reasoningActions.appendDelta('hello ')
    reasoningActions.appendDelta('world')
    const state = reasoningState.get()
    expect(state.text).toBe('hello world')
    expect(state.updatedAt).not.toBeNull()
  })

  it('appendDelta truncates from the head once the buffer exceeds the cap', () => {
    const long = 'a'.repeat(300)
    reasoningActions.appendDelta(long)
    expect(reasoningState.get().text.length).toBeLessThanOrEqual(240)
  })

  it('clear resets to initial state', () => {
    reasoningActions.appendDelta('something')
    reasoningActions.clear()
    expect(reasoningState.get()).toEqual({ text: '', updatedAt: null })
  })

  it('isStale returns true when nothing has been appended', () => {
    expect(reasoningActions.isStale()).toBe(true)
  })

  it('isStale returns false right after a delta and true after the fade window', () => {
    reasoningActions.appendDelta('thinking…')
    expect(reasoningActions.isStale()).toBe(false)
    expect(reasoningActions.isStale(Date.now() + 9000)).toBe(true)
  })
})
