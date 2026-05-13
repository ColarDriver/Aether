import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { activityActions, activityState } from '../store/activityStore.js'

describe('activityStore', () => {
  beforeEach(() => {
    activityActions.resetForTests()
    vi.useFakeTimers()
  })
  afterEach(() => {
    activityActions.resetForTests()
    vi.useRealTimers()
  })

  it('beginTurn flips status to thinking and stamps thinkingStartedAt', () => {
    activityActions.beginTurn()
    const state = activityState.get()
    expect(state.status).toBe('thinking')
    expect(state.thinkingStartedAt).not.toBeNull()
    expect(state.iteration).toBe(0)
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

  it('bumpAnimation increments the animation tick monotonically', () => {
    const start = activityState.get().animationTick
    activityActions.bumpAnimation()
    activityActions.bumpAnimation()
    expect(activityState.get().animationTick).toBe(start + 2)
  })
})
