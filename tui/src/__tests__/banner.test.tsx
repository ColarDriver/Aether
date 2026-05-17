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

  it('renders grouped provider, session, tools, and skills rows', () => {
    sessionState.set({
      ...sessionState.get(),
      sessionId: 'ses_abcdef12',
      provider: 'claude',
      model: 'claude-sonnet-4-6',
      baseUrl: 'http://localhost:8000/v1',
      recentSessions: [
        {
          session_id: 'ses_other01',
          created_at: 0,
          updated_at: 0,
          provider: 'openai',
          model: 'gpt-4o',
          summary: 'first recent session'
        }
      ],
      bannerTools: ['read_file', 'list_directory', 'search_code'],
      bannerToolCount: 3,
      bannerSkills: ['dogfood'],
      bannerSkillCount: 1,
      gatewayReady: {
        type: 'gateway.ready',
        version: '0.5.0',
        capabilities: [],
        methods: ['a', 'b', 'c']
      }
    })
    const { lastFrame, unmount } = render(<Banner />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Aether')
    expect(frame).toContain('v1.0.0')
    expect(frame).not.toContain('industrial agent harness')
    expect(frame).toContain('provider')
    expect(frame).toContain('claude')
    expect(frame).toContain('model')
    expect(frame).toContain('claude-sonnet-4-6')
    expect(frame).not.toContain('http://localhost:8000/v1')
    expect(frame).toContain('session')
    expect(frame).toContain('ses_abcd')
    expect(frame).toContain('ses_othe')
    expect(frame).toContain('first recent session')
    expect(frame).toContain('tools')
    expect(frame).toContain('3')
    expect(frame).toContain('read_file')
    expect(frame).toContain('list_directory')
    expect(frame).toContain('search_code')
    expect(frame).toContain('skills')
    expect(frame).toContain('dogfood')
    expect(frame).toContain('cwd')
    expect(frame).not.toContain('0.5.0')
    expect(frame).not.toContain('Type your message')
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

  it('shows plan mode when the current session is planning', () => {
    sessionState.set({
      ...sessionState.get(),
      sessionId: 'ses_planmode',
      mode: 'plan'
    })
    const { lastFrame, unmount } = render(<Banner />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('mode')
    expect(frame).toContain('plan')
    unmount()
  })

  it('falls back to BootLine when AETHER_NO_BANNER=1', () => {
    process.env.AETHER_NO_BANNER = '1'
    sessionState.set({
      ...sessionState.get(),
      sessionId: 'ses_xyz',
      mode: 'plan',
      provider: 'openai',
      model: 'gpt-4o'
    })
    const { lastFrame, unmount } = render(<Banner />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('aether')
    expect(frame).toContain('openai/gpt-4o')
    expect(frame).toContain('plan')
    expect(frame).not.toContain('industrial agent harness')
    unmount()
  })
})
