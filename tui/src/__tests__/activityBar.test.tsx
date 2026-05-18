import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render } from 'ink-testing-library'

import { ActivityBar } from '../components/ActivityBar.js'
import { activityActions, activityState } from '../store/activityStore.js'
import { sessionActions, sessionState } from '../store/sessionStore.js'

function tokenCountFrom(frame: string): number | null {
  const match = frame.match(/↓ ([\d,.]+) tokens/)
  if (!match) {
    return null
  }
  const raw = match[1]
  if (!raw) {
    return null
  }
  return Number(raw.replace(/[,.]/g, ''))
}

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
    activityActions.addResponseChars(480 * 4)
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

  it('uses provider usage as a fallback and lets streamed progress overtake it', () => {
    activityActions.beginTurn()
    activityActions.addUsage({ input: 100, output: 321 })
    activityActions.flushUsage()

    const { lastFrame, rerender, unmount } = render(<ActivityBar />)
    expect(lastFrame() ?? '').toContain('↓ 321 tokens')

    activityActions.addResponseChars(321 * 4 + 120)
    rerender(<ActivityBar />)
    const firstFrame = lastFrame() ?? ''
    const firstCount = tokenCountFrom(firstFrame)
    expect(firstCount).not.toBeNull()
    expect(firstCount ?? 0).toBeGreaterThan(321)

    activityActions.bumpAnimation()
    rerender(<ActivityBar />)
    const secondCount = tokenCountFrom(lastFrame() ?? '')
    expect(secondCount).not.toBeNull()
    expect(secondCount ?? 0).toBeGreaterThan(firstCount ?? 0)
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

  it('renders todo_write items as an activity checklist', () => {
    activityActions.beginTurn()
    activityActions.setTodos([
      {
        id: '1',
        content: 'Fix unwanted blank lines in nested markdown lists',
        activeForm: 'Fixing nested markdown list spacing',
        status: 'in_progress'
      },
      {
        id: '2',
        content: 'Verify with build and tests',
        status: 'pending'
      }
    ])

    const { lastFrame, unmount } = render(<ActivityBar />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Fixing nested markdown list spacing')
    expect(frame).toContain('└ ■ Fix unwanted blank lines in nested markdown lists')
    expect(frame).toContain('□ Verify with build and tests')
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
