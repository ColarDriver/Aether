import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SPINNER_VERBS } from '../lib/shimmer.js'
import {
  activityActions,
  activityInterruptPending,
  activityState,
  activityStatus,
  activityTodoCount
} from '../store/activityStore.js'

function expectValidVerbIndex(index: number): void {
  expect(Number.isInteger(index)).toBe(true)
  expect(index).toBeGreaterThanOrEqual(0)
  expect(index).toBeLessThan(SPINNER_VERBS.length)
}

describe('activityStore', () => {
  beforeEach(() => {
    activityActions.resetForTests()
    vi.useFakeTimers()
  })
  afterEach(() => {
    activityActions.resetForTests()
    vi.useRealTimers()
  })

  it('beginTurn flips status to requesting and stamps thinkingStartedAt', () => {
    activityActions.beginTurn()
    const state = activityState.get()
    expect(state.status).toBe('requesting')
    expect(state.thinkingStartedAt).not.toBeNull()
    expect(state.iteration).toBe(0)
    expectValidVerbIndex(state.turnVerbIndex)
  })

  it('beginTurn samples a fresh in-range verb seed each turn', () => {
    activityActions.beginTurn()
    expectValidVerbIndex(activityState.get().turnVerbIndex)
    activityActions.endTurn('done')
    activityActions.beginTurn()
    expectValidVerbIndex(activityState.get().turnVerbIndex)
  })

  it('beginTurn clears previous-turn usage and pending usage flushes', () => {
    activityActions.addUsage({ input: 100, output: 25, cacheRead: 5, cacheWrite: 3 })
    activityActions.flushUsage()
    expect(activityState.get().tokensOut).toBe(25)

    activityActions.addUsage({ input: 10, output: 4 })
    activityActions.beginTurn()
    vi.advanceTimersByTime(150)

    const state = activityState.get()
    expect(state.tokensIn).toBe(0)
    expect(state.tokensOut).toBe(0)
    expect(state.cacheRead).toBe(0)
    expect(state.cacheWrite).toBe(0)
  })

  it('setStatus transitions update timestamps appropriately', () => {
    activityActions.setStatus('thinking')
    expect(activityState.get().thinkingStartedAt).not.toBeNull()
    activityActions.setStatus('responding')
    expect(activityState.get().responseStartedAt).not.toBeNull()
    activityActions.setStatus('idle')
    expect(activityState.get().thinkingStartedAt).toBeNull()
    expect(activityState.get().responseStartedAt).toBeNull()
  })

  it('addUsage throttles into a single store update per 100 ms window', () => {
    activityActions.addUsage({ input: 5, output: 1 })
    activityActions.addUsage({ input: 3, output: 2 })
    activityActions.addUsage({ input: 1, output: 1 })

    expect(activityState.get().tokensIn).toBe(0)
    expect(activityState.get().tokensOut).toBe(0)

    vi.advanceTimersByTime(150)
    expect(activityState.get().tokensIn).toBe(9)
    expect(activityState.get().tokensOut).toBe(4)
  })

  it('flushUsage forces immediate write of pending tokens', () => {
    activityActions.addUsage({ input: 5, output: 2 })
    activityActions.flushUsage()
    expect(activityState.get().tokensIn).toBe(5)
    expect(activityState.get().tokensOut).toBe(2)
  })

  it('endTurn maps cancelled / error / done to the correct terminal status', () => {
    activityActions.beginTurn()
    activityActions.endTurn('cancelled')
    expect(activityState.get().status).toBe('cancelled')

    activityActions.beginTurn()
    activityActions.endTurn('error')
    expect(activityState.get().status).toBe('error')

    activityActions.beginTurn()
    activityActions.endTurn('done')
    expect(activityState.get().status).toBe('idle')
  })

  it('setIteration tracks the current iteration', () => {
    activityActions.setIteration(3, 8)
    expect(activityState.get().iteration).toBe(3)
    expect(activityState.get().maxIterations).toBe(8)
  })

  it('stores the current todo list for activity rendering', () => {
    activityActions.setTodos([
      { id: '1', content: 'Implement registry', status: 'in_progress' }
    ])
    expect(activityState.get().todos).toEqual([
      { id: '1', content: 'Implement registry', status: 'in_progress' }
    ])
  })

  it('bumpAnimation increments the animation tick monotonically', () => {
    const start = activityState.get().animationTick
    activityActions.bumpAnimation()
    activityActions.bumpAnimation()
    expect(activityState.get().animationTick).toBe(start + 2)
  })

  it('does not notify narrow UI stores for animation-only ticks', () => {
    const statusListener = vi.fn()
    const interruptListener = vi.fn()
    const todoCountListener = vi.fn()
    const unlistenStatus = activityStatus.listen(statusListener)
    const unlistenInterrupt = activityInterruptPending.listen(interruptListener)
    const unlistenTodoCount = activityTodoCount.listen(todoCountListener)

    statusListener.mockClear()
    interruptListener.mockClear()
    todoCountListener.mockClear()

    activityActions.bumpAnimation()
    activityActions.bumpAnimation()

    expect(statusListener).not.toHaveBeenCalled()
    expect(interruptListener).not.toHaveBeenCalled()
    expect(todoCountListener).not.toHaveBeenCalled()

    unlistenStatus()
    unlistenInterrupt()
    unlistenTodoCount()
  })
})
