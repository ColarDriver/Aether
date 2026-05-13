import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render } from 'ink-testing-library'

import { ActivityBar } from '../components/ActivityBar.js'
import { activityActions, activityState } from '../store/activityStore.js'
import { sessionActions, sessionState } from '../store/sessionStore.js'

describe('ActivityBar', () => {
  beforeEach(() => {
    activityActions.resetForTests()
    sessionActions.resetForTests()
    vi.useFakeTimers()
  })
  afterEach(() => {
    activityActions.resetForTests()
    sessionActions.resetForTests()
    vi.useRealTimers()
  })

  it('renders the idle state without a spinner', () => {
    activityState.set({ ...activityState.get(), status: 'idle' })
    const { lastFrame, unmount } = render(<ActivityBar />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('idle')
    unmount()
  })

  it('shows iter counter, token segments, model and session id when active', () => {
    activityActions.beginTurn()
    activityActions.setIteration(2, 8)
    activityActions.addUsage({ input: 1200, output: 480 })
    activityActions.flushUsage()
    sessionState.set({
      ...sessionState.get(),
      sessionId: 'ses_abcdef',
      model: 'sonnet-4-6'
    })

    const { lastFrame, unmount } = render(<ActivityBar />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('thinking')
    expect(frame).toContain('iter 2/8')
    expect(frame).toContain('1.2k in')
    expect(frame).toContain('480 out')
    expect(frame).toContain('ses_abcd')
    expect(frame).toContain('sonnet-4-6')
    unmount()
  })

  it('renders the cancelled status when endTurn(cancelled) is called', () => {
    activityActions.beginTurn()
    activityActions.endTurn('cancelled')
    const { lastFrame, unmount } = render(<ActivityBar />)
    expect(lastFrame()).toContain('cancelled')
    unmount()
  })
})
