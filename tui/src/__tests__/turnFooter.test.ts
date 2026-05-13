import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { applyGatewayEvent, resetGatewayEventDedupeForTests } from '../hooks/useGatewayEvents.js'
import { activityActions } from '../store/activityStore.js'
import { chatActions, chatItems } from '../store/chatStore.js'
import { reasoningActions } from '../store/reasoningStore.js'
import { sessionActions } from '../store/sessionStore.js'
import { toolGroupActions } from '../store/toolGroupStore.js'

beforeEach(() => {
  chatActions.resetForTests()
  sessionActions.resetForTests()
  activityActions.resetForTests()
  toolGroupActions.resetForTests()
  reasoningActions.resetForTests()
  resetGatewayEventDedupeForTests()
})

afterEach(() => {
  resetGatewayEventDedupeForTests()
})

const FOOTER_RE = /^[✓⏹✗] (done|cancelled|failed)\b/

function footers(): string[] {
  return chatItems.get().flatMap((item) => {
    if (item.kind !== 'note' || !FOOTER_RE.test(item.text)) {
      return []
    }
    return [item.text]
  })
}

describe('per-turn footer', () => {
  it('done event prints "● done" with iteration + tool counts', () => {
    activityActions.beginTurn()
    applyGatewayEvent({
      type: 'iteration.start',
      session_id: 's',
      run_id: 'r',
      iteration: 1
    })
    applyGatewayEvent({
      type: 'tool.call',
      session_id: 's',
      run_id: 'r',
      tool_call_id: 'tc1',
      tool_name: 'shell',
      arguments: { command: 'ls' },
      iteration: 1
    })
    applyGatewayEvent({
      type: 'tool.result',
      session_id: 's',
      run_id: 'r',
      tool_call_id: 'tc1',
      tool_name: 'shell',
      content: 'ok',
      is_error: false,
      iteration: 1
    })
    applyGatewayEvent({
      type: 'iteration.end',
      session_id: 's',
      run_id: 'r',
      iteration: 1
    })
    applyGatewayEvent({
      type: 'done',
      session_id: 's',
      run_id: 'r',
      final_text: 'all good',
      exit_reason: 'done'
    })
    const list = footers()
    expect(list).toHaveLength(1)
    expect(list[0] ?? '').toMatch(/done.* 1 iter.* 1 tool/)
  })

  it('cancelled event prints a "cancelled" footer with the interrupt glyph', () => {
    activityActions.beginTurn()
    applyGatewayEvent({
      type: 'cancelled',
      session_id: 's',
      run_id: 'r',
      reason: 'user'
    })
    const list = footers()
    expect(list).toHaveLength(1)
    expect(list[0] ?? '').toMatch(/^[⏹] cancelled/)
  })

  it('error event prints a "failed" footer with the error counter', () => {
    activityActions.beginTurn()
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'something failed'
    })
    const list = footers()
    expect(list).toHaveLength(1)
    expect(list[0] ?? '').toMatch(/^[✗] failed/)
    expect(list[0] ?? '').toContain('1 error')
  })
})
