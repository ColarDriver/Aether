import { beforeEach, describe, expect, it } from 'vitest'

import { composerActions, composerState } from '../store/composerStore.js'

beforeEach(() => {
  composerActions.resetForTests()
})

describe('composer cursor — basic editing', () => {
  it('insert advances the cursor by the inserted length', () => {
    composerActions.insert('hello')
    expect(composerState.get()).toMatchObject({ draft: 'hello', cursor: 5 })
  })

  it('moveLeft / moveRight navigate one character at a time', () => {
    composerActions.insert('ab')
    composerActions.moveLeft()
    expect(composerState.get().cursor).toBe(1)
    composerActions.moveLeft()
    expect(composerState.get().cursor).toBe(0)
    composerActions.moveLeft()
    expect(composerState.get().cursor).toBe(0) // clamps at start
    composerActions.moveRight()
    expect(composerState.get().cursor).toBe(1)
  })

  it('insert at non-end position splices into the draft', () => {
    composerActions.insert('hello')
    composerActions.moveLeft()
    composerActions.moveLeft() // cursor between 'l' and 'o' — index 3
    composerActions.insert('XX')
    expect(composerState.get()).toMatchObject({ draft: 'helXXlo', cursor: 5 })
  })

  it('backspace removes char to the left of cursor', () => {
    composerActions.insert('hello')
    composerActions.moveLeft() // cursor at 4
    composerActions.backspace()
    expect(composerState.get()).toMatchObject({ draft: 'hello'.slice(0, 3) + 'o', cursor: 3 })
  })

  it('deleteForward removes char to the right of cursor', () => {
    composerActions.insert('hello')
    composerActions.moveLeft()
    composerActions.deleteForward()
    expect(composerState.get()).toMatchObject({ draft: 'hell', cursor: 4 })
  })
})

describe('composer cursor — word ops', () => {
  it('moveWordLeft jumps over the previous word', () => {
    composerActions.insert('hello world!')
    composerActions.moveWordLeft()
    expect(composerState.get().cursor).toBe(6) // start of 'world'
    composerActions.moveWordLeft()
    expect(composerState.get().cursor).toBe(0)
  })

  it('moveWordRight jumps to the end of the next word', () => {
    composerActions.insert('hello world!')
    composerActions.moveLineStart()
    composerActions.moveWordRight()
    expect(composerState.get().cursor).toBe(5) // end of 'hello'
    composerActions.moveWordRight()
    expect(composerState.get().cursor).toBe(11) // end of 'world'
  })

  it('deleteWordBackward removes the preceding word', () => {
    composerActions.insert('hello world')
    composerActions.deleteWordBackward()
    expect(composerState.get()).toMatchObject({ draft: 'hello ', cursor: 6 })
  })
})

describe('composer cursor — line ops', () => {
  it('moveLineStart / moveLineEnd jump within the current line', () => {
    composerActions.insert('alpha\nbeta gamma')
    composerActions.moveLineStart()
    expect(composerState.get().cursor).toBe(6) // start of 'beta gamma'
    composerActions.moveLineEnd()
    expect(composerState.get().cursor).toBe(16)
  })

  it('moveLineUp / moveLineDown preserve column when possible', () => {
    composerActions.insert('alpha\nbeta\ngamma')
    composerActions.moveLineStart()
    composerActions.moveLineUp() // → row 1, col 0
    expect(composerState.get().cursor).toBe(6)
    composerActions.moveLineUp() // → row 0, col 0
    expect(composerState.get().cursor).toBe(0)
    expect(composerActions.moveLineUp()).toBe(false) // already at top
  })

  it('killToLineStart removes from cursor back to the line start', () => {
    composerActions.insert('alpha\nbeta gamma')
    composerActions.moveLineEnd()
    composerActions.killToLineStart()
    expect(composerState.get()).toMatchObject({ draft: 'alpha\n', cursor: 6 })
  })

  it('killToLineEnd removes from cursor to the line end', () => {
    composerActions.insert('alpha\nbeta gamma')
    composerActions.moveLineStart() // start of "beta gamma"
    composerActions.moveRight()
    composerActions.killToLineEnd()
    expect(composerState.get()).toMatchObject({ draft: 'alpha\nb', cursor: 7 })
  })
})
