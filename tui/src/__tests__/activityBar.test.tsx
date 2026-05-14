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
    expect(frame).toContain('Idle')
    unmount()
  })

  it('shows token segments, model and session id when active', () => {
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
    expect(frame).toContain('Thinking')
    expect(frame).not.toContain('iter 2/8')
    // Mirrors Python `activity.py:239-241` — output-only token line with `↓`
    // flow-direction arrow. Input tokens are surfaced via /stats, not the
    // live bar.
    expect(frame).toContain('↓ 480 tokens')
    expect(frame).toContain('ses_abcd')
    expect(frame).toContain('sonnet-4-6')
    unmount()
  })

  it('uses a tool-specific verb instead of the raw responding/tool_use label', () => {
    activityActions.beginTurn()
    activityActions.setStatus('tool_use', 'read_file')
    const { lastFrame, unmount } = render(<ActivityBar />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Reading')
    expect(frame).not.toContain('responding')
    expect(frame).not.toContain('tool use')
    unmount()
  })

  it('renders the cancelled status when endTurn(cancelled) is called', () => {
    activityActions.beginTurn()
    activityActions.endTurn('cancelled')
    const { lastFrame, unmount } = render(<ActivityBar />)
    expect(lastFrame()).toContain('Cancelled')
    unmount()
  })
})
