import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { render } from 'ink-testing-library'

import { Banner } from '../components/Banner.js'
import { sessionActions, sessionState } from '../store/sessionStore.js'

describe('Banner', () => {
  beforeEach(() => {
    sessionActions.resetForTests()
    delete process.env.AETHER_NO_BANNER
  })
  afterEach(() => {
    sessionActions.resetForTests()
    delete process.env.AETHER_NO_BANNER
  })

  it('renders title, tagline, provider, model, session, gateway rows', () => {
    sessionState.set({
      ...sessionState.get(),
      sessionId: 'ses_abcdef12',
      provider: 'claude',
      model: 'claude-sonnet-4-6'
    })
    const { lastFrame, unmount } = render(<Banner />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Aether')
    expect(frame).toContain('industrial agent harness')
    expect(frame).toContain('provider')
    expect(frame).toContain('claude')
    expect(frame).toContain('model')
    expect(frame).toContain('claude-sonnet-4-6')
    expect(frame).toContain('session')
    expect(frame).toContain('ses_abcd')
    expect(frame).toContain('gateway')
    unmount()
  })

  it('shows the "custom system prompt loaded" row when systemOverride is set', () => {
    sessionState.set({
      ...sessionState.get(),
      sessionId: 'ses_xyz',
      systemOverride: 'be terse'
    })
    const { lastFrame, unmount } = render(<Banner />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('system')
    expect(frame).toContain('custom system prompt loaded')
    unmount()
  })

  it('falls back to BootLine when AETHER_NO_BANNER=1', () => {
    process.env.AETHER_NO_BANNER = '1'
    sessionState.set({
      ...sessionState.get(),
      sessionId: 'ses_xyz',
      provider: 'openai',
      model: 'gpt-4o'
    })
    const { lastFrame, unmount } = render(<Banner />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('aether')
    expect(frame).toContain('openai/gpt-4o')
    expect(frame).not.toContain('industrial agent harness')
    unmount()
  })
})
