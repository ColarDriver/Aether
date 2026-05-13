import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { activityActions, activityState } from '../store/activityStore.js'
import { sessionActions, sessionState } from '../store/sessionStore.js'
import { statsCommand } from '../slash/commands/stats.js'
import type { SlashCtx } from '../slash/dispatcher.js'

function ctx(): SlashCtx {
  return {
    client: { request: async () => ({}) as never },
    catalog: [],
    getSession: () => sessionState.get(),
    createSession: async () => ({
      session_id: 'ses_x',
      created_at: 0,
      updated_at: 0,
      provider: 'openai',
      model: 'gpt-4o',
      system_prompt: null
    }),
    setSystemOverride: () => undefined,
    toggleVerbose: () => false
  }
}

describe('/stats command', () => {
  beforeEach(() => {
    activityActions.resetForTests()
    sessionActions.resetForTests()
  })
  afterEach(() => {
    activityActions.resetForTests()
    sessionActions.resetForTests()
  })

  it('reports a friendly placeholder when no turn has completed', async () => {
    const result = await statsCommand.execute([], ctx())
    expect(result.kind).toBe('note')
    if (result.kind === 'note') {
      expect(result.text).toContain('no turn completed yet')
    }
  })

  it('renders last-turn metrics after endTurn populates them', async () => {
    activityActions.beginTurn()
    activityActions.setIteration(3, 8)
    activityActions.bumpToolCounter(false)
    activityActions.bumpToolCounter(false)
    activityActions.addResponseChars(120)
    // Force the turn timer back so durationMs > 0.
    activityState.set({
      ...activityState.get(),
      turnStartedAt: Date.now() - 1500
    })
    activityActions.endTurn('done')
    const result = await statsCommand.execute([], ctx())
    expect(result.kind).toBe('note')
    if (result.kind === 'note') {
      expect(result.text).toContain('last turn:')
      expect(result.text).toContain('iter=3')
      expect(result.text).toContain('tool_calls=2')
      expect(result.text).toContain('errors=0')
      expect(result.text).toContain('chars=120')
      expect(result.text).toMatch(/elapsed=1\.\d+s/)
    }
  })
})
