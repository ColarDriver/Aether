import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { render } from 'ink-testing-library'

import { ActivityBar } from '../components/ActivityBar.js'
import { activityActions } from '../store/activityStore.js'
import { sessionActions } from '../store/sessionStore.js'

beforeEach(() => {
  activityActions.resetForTests()
  sessionActions.resetForTests()
})
afterEach(() => {
  activityActions.resetForTests()
  sessionActions.resetForTests()
})

describe('ActivityBar loop state translation', () => {
  it('hides routine engine sub-phases like LLM_CALL', () => {
    activityActions.beginTurn()
    activityActions.setLoopState('LLM_CALL')
    const { lastFrame, unmount } = render(<ActivityBar />)
    const frame = lastFrame() ?? ''
    expect(frame).not.toContain('LLM_CALL')
    expect(frame).not.toContain('loop:')
    unmount()
  })

  it('translates terminal states like FAILED to a friendly label', () => {
    activityActions.beginTurn()
    activityActions.setLoopState('FAILED')
    const { lastFrame, unmount } = render(<ActivityBar />)
    expect(lastFrame()).toContain('failed')
    unmount()
  })

  it('translates COMPACTION → "compacting"', () => {
    activityActions.beginTurn()
    activityActions.setLoopState('COMPACTION')
    const { lastFrame, unmount } = render(<ActivityBar />)
    expect(lastFrame()).toContain('compacting')
    unmount()
  })

  it('drops unknown ALL_CAPS_WITH_UNDERSCORES states as engine jargon', () => {
    activityActions.beginTurn()
    activityActions.setLoopState('SOMETHING_INTERNAL')
    const { lastFrame, unmount } = render(<ActivityBar />)
    expect(lastFrame()).not.toContain('SOMETHING_INTERNAL')
    unmount()
  })
})
