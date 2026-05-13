import { beforeEach, describe, expect, it } from 'vitest'

import { composerActions, composerState } from '../store/composerStore.js'

describe('composer store', () => {
  beforeEach(() => {
    composerActions.resetForTests()
  })

  it('enters multiline mode from trailing backslash', () => {
    composerActions.setDraft('hello\\')
    composerActions.newline()

    expect(composerState.get().draft).toBe('hello\n')
    expect(composerState.get().multiline).toBe(true)
  })

  it('commits history and recalls it', () => {
    composerActions.commit('first')
    composerActions.commit('second')
    composerActions.previousHistory()

    expect(composerState.get().draft).toBe('second')

    composerActions.clear()
    composerActions.previousHistory()
    composerActions.previousHistory()

    expect(composerState.get().draft).toBe('first')
  })
})
