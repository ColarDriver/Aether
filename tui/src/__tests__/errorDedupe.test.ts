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

function rawErrorNotes() {
  // Per-turn footers also use level: 'error' (e.g. `✗ failed · …`); filter
  // them out so the assertions stay focused on the actual error message rows.
  // The footer always begins with ✓ / ⏹ / ✗ followed by " done"/" cancelled"/
  // " failed", so a plain prefix check is enough.
  return chatItems
    .get()
    .filter(
      (item) =>
        item.kind === 'note' &&
        item.level === 'error' &&
        !/^[✓⏹✗] (done|cancelled|failed)\b/.test(item.text)
    )
}

describe('error event dedupe', () => {
  it('strips loop-state prefixes like "LLM_CALL:" so users see the cleaned message', () => {
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'LLM_CALL: provider HTTP 404: 404 page not found'
    })
    const errors = rawErrorNotes()
    expect(errors).toHaveLength(1)
    expect(errors[0]).toMatchObject({
      kind: 'note',
      level: 'error',
      text: 'provider HTTP 404: 404 page not found'
    })
  })

  it('suppresses a second error whose body matches the first within the dedupe window', () => {
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'LLM_CALL: provider HTTP 404: 404 page not found'
    })
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'provider HTTP 404: 404 page not found'
    })
    expect(rawErrorNotes()).toHaveLength(1)
  })

  it('emits a fresh error after a `done` event resets the dedupe window', () => {
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'temporary outage'
    })
    applyGatewayEvent({
      type: 'done',
      session_id: 's',
      run_id: 'r2',
      final_text: 'recovered',
      exit_reason: 'done'
    })
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r3',
      message: 'temporary outage'
    })
    expect(rawErrorNotes()).toHaveLength(2)
  })

  it('emits distinct errors when their messages differ', () => {
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'first failure'
    })
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'second, unrelated failure'
    })
    expect(rawErrorNotes()).toHaveLength(2)
  })

  it('also dedupes a `gateway.error` event that follows an agent error', () => {
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'LLM_CALL: provider HTTP 404'
    })
    applyGatewayEvent({
      type: 'gateway.error',
      message: 'provider HTTP 404'
    })
    expect(rawErrorNotes()).toHaveLength(1)
  })

  it('appends a hint note when the error matches a known misconfiguration pattern', () => {
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'provider HTTP 404: 404 page not found'
    })
    const hints = chatItems.get().flatMap((item) => {
      if (item.kind !== 'note' || item.level !== 'warn') {
        return []
      }
      return [item.text]
    })
    expect(hints.some((text) => text.startsWith('hint:'))).toBe(true)
  })

  it('appends a per-turn footer in the error path', () => {
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'something failed'
    })
    const footers = chatItems.get().flatMap((item) => {
      if (item.kind !== 'note') {
        return []
      }
      return [item.text]
    })
    expect(footers.some((text) => /^[✓⏹✗] failed/.test(text))).toBe(true)
  })

  it('only emits one turn footer when the gateway double-fires Error for one run', () => {
    // Regression for the duplicate `✗ failed · 2 errors` footer reported in
    // tmp/error1.png: the gateway emits one Error from `on_error` middleware
    // and a second from the top-level except in `agent_methods.agent_run`.
    // The TS handler must dedupe the finalisation work per run_id.
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'LLM_CALL: provider HTTP 404: 404 page not found'
    })
    applyGatewayEvent({
      type: 'error',
      session_id: 's',
      run_id: 'r',
      message: 'provider HTTP 404: 404 page not found'
    })
    const footers = chatItems
      .get()
      .filter((item) => item.kind === 'note' && /^[✓⏹✗] failed/.test(item.text))
    expect(footers).toHaveLength(1)
  })
})
