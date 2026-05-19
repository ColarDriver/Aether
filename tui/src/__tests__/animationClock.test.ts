import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  _clockStateForTests,
  _resetClockForTests,
  clockStartedAt,
  subscribe
} from '../lib/animationClock.js'

describe('animationClock', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    _resetClockForTests()
  })

  afterEach(() => {
    _resetClockForTests()
    vi.useRealTimers()
  })

  it('starts when the first listener subscribes and stops after the last unsubscribe', () => {
    const listener = vi.fn()
    const unsubscribe = subscribe(listener)

    expect(_clockStateForTests()).toEqual({ listeners: 1, running: true })
    expect(clockStartedAt()).toBe(0)

    vi.advanceTimersByTime(50)
    expect(listener).toHaveBeenCalledTimes(1)

    unsubscribe()
    expect(_clockStateForTests()).toEqual({ listeners: 0, running: false })

    vi.advanceTimersByTime(50)
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('uses one shared heartbeat for multiple listeners', () => {
    const first = vi.fn()
    const second = vi.fn()
    const stopFirst = subscribe(first)
    const stopSecond = subscribe(second)

    expect(_clockStateForTests()).toEqual({ listeners: 2, running: true })

    vi.advanceTimersByTime(50)
    expect(first).toHaveBeenCalledTimes(1)
    expect(second).toHaveBeenCalledTimes(1)

    stopFirst()
    vi.advanceTimersByTime(50)
    expect(first).toHaveBeenCalledTimes(1)
    expect(second).toHaveBeenCalledTimes(2)

    stopSecond()
    expect(_clockStateForTests()).toEqual({ listeners: 0, running: false })
  })
})
