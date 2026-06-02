import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render } from 'ink-testing-library'

import { ActivityBar } from '../components/ActivityBar.js'
import { SPINNER_VERBS } from '../lib/shimmer.js'
import { activityActions, activityState } from '../store/activityStore.js'
import { sessionActions, sessionState } from '../store/sessionStore.js'

function tokenCountFrom(frame: string): number | null {
  const match = frame.match(/[↑↓] ([\d,.]+) tokens/)
  if (!match) {
    return null
  }
  const raw = match[1]
  if (!raw) {
    return null
  }
  return Number(raw.replace(/[,.]/g, ''))
}

/** The headline verb is now a randomly-sampled whimsical word, so assert it is
 * one of the pool rather than any specific value. */
function expectSpinnerVerb(frame: string): void {
  expect(SPINNER_VERBS.some((verb) => frame.includes(verb))).toBe(true)
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
    // Output tokens stream back during `responding`; `requesting` would show the
    // ↑ (model-wait) arrow instead. See the dedicated arrow tests below.
    activityActions.setStatus('responding')
    activityActions.setIteration(2, 8)
    activityActions.addResponseChars(480 * 4)
    sessionState.set({
      ...sessionState.get(),
      sessionId: 'ses_abcdef',
      model: 'sonnet-4-6'
    })

    const { lastFrame, unmount } = render(<ActivityBar />)
    const frame = lastFrame() ?? ''
    expectSpinnerVerb(frame)
    expect(frame).not.toContain('iter 2/8')
    // Mirrors Python `activity.py:239-241` — output-only token line with `↓`
    // flow-direction arrow. Input tokens are surfaced via /stats, not the
    // live bar.
    expect(frame).toContain('↓ 480 tokens')
    expect(frame).toContain('ses_abcd')
    expect(frame).toContain('sonnet-4-6')
    unmount()
  })

  it('flags the requesting (model-wait) phase with an up arrow', () => {
    activityActions.beginTurn()
    // beginTurn lands in `requesting`; output count is shown with the ↑ arrow
    // because bytes are still flowing out to the model.
    activityActions.addResponseChars(120 * 4)
    const { lastFrame, unmount } = render(<ActivityBar />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('↑ 120 tokens')
    expect(frame).not.toContain('↓ 120 tokens')
    unmount()
  })

  it('uses provider usage as a fallback and lets streamed progress overtake it', () => {
    activityActions.beginTurn()
    activityActions.setStatus('responding')
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

  it('keeps animation ticks local instead of mutating the activity store', () => {
    activityActions.beginTurn()
    const before = activityState.get().animationTick
    const { lastFrame, unmount } = render(<ActivityBar />)

    vi.advanceTimersByTime(500)

    expect(activityState.get().animationTick).toBe(before)
    expectSpinnerVerb(lastFrame() ?? '')
    unmount()
  })

  it('can disable animation rendering explicitly', () => {
    activityActions.beginTurn()
    const { lastFrame, unmount } = render(<ActivityBar animate={false} />)

    vi.advanceTimersByTime(500)

    expectSpinnerVerb(lastFrame() ?? '')
    expect(activityState.get().animationTick).toBe(0)
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
