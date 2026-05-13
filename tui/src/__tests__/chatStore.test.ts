import { beforeEach, describe, expect, it } from 'vitest'

import { applyGatewayEvent } from '../hooks/useGatewayEvents.js'
import { chatActions, chatItems } from '../store/chatStore.js'
import { sessionActions, sessionState } from '../store/sessionStore.js'

describe('chat store event mapping', () => {
  beforeEach(() => {
    chatActions.resetForTests()
    sessionActions.resetForTests()
  })

  it('appends streaming text deltas and marks done', () => {
    applyGatewayEvent({
      type: 'text.delta',
      session_id: 's1',
      run_id: 'r1',
      text: 'hel',
      sequence: 0
    })
    applyGatewayEvent({
      type: 'text.delta',
      session_id: 's1',
      run_id: 'r1',
      text: 'lo',
      sequence: 1
    })
    applyGatewayEvent({
      type: 'done',
      session_id: 's1',
      run_id: 'r1',
      final_text: 'hello',
      exit_reason: 'done'
    })

    const items = chatItems.get()
    expect(items[0]).toMatchObject({
      kind: 'assistant',
      id: 'r1',
      text: 'hello',
      streaming: false
    })
    // The done event also appends the per-turn footer note (icon + verb).
    expect(
      items.some((item) => item.kind === 'note' && /^[✓⏹✗] done\b/.test(item.text))
    ).toBe(true)
    expect(sessionState.get().status).toBe('idle')
  })

  it('maps status and usage events into session state', () => {
    applyGatewayEvent({
      type: 'status',
      session_id: 's1',
      run_id: 'r1',
      kind: 'thinking',
      detail: null
    })
    applyGatewayEvent({
      type: 'usage',
      session_id: 's1',
      run_id: 'r1',
      input_tokens: 10,
      output_tokens: 3,
      cache_read_tokens: 2,
      cache_write_tokens: 1
    })

    expect(sessionState.get().status).toBe('thinking')
    expect(sessionState.get().usage).toEqual({
      input: 10,
      output: 3,
      cacheRead: 2,
      cacheWrite: 1
    })
  })
})
