import { beforeEach, describe, expect, it } from 'vitest'

import { applyGatewayEvent, resetGatewayEventDedupeForTests } from '../hooks/useGatewayEvents.js'
import { activityActions, activityState } from '../store/activityStore.js'
import { chatActions, chatItems } from '../store/chatStore.js'
import { sessionActions, sessionState } from '../store/sessionStore.js'

describe('chat store event mapping', () => {
  beforeEach(() => {
    chatActions.resetForTests()
    activityActions.resetForTests()
    sessionActions.resetForTests()
    resetGatewayEventDedupeForTests()
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
      runId: 'r1',
      text: 'hello',
      streaming: false
    })
    // The done event also appends the per-turn footer note (icon + verb).
    expect(
      items.some((item) => item.kind === 'note' && /^[✓⏹✗] done\b/.test(item.text))
    ).toBe(true)
    expect(sessionState.get().status).toBe('idle')
  })

  it('keeps later assistant streaming below interleaved shell output', () => {
    applyGatewayEvent({
      type: 'text.delta',
      session_id: 's1',
      run_id: 'r1',
      text: 'before ',
      sequence: 0
    })
    applyGatewayEvent({
      type: 'tool.call',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'tc1',
      tool_name: 'shell',
      arguments: { command: 'echo hi' },
      iteration: 1
    })
    applyGatewayEvent({
      type: 'tool.result',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'tc1',
      tool_name: 'shell',
      content: 'hi',
      is_error: false,
      iteration: 1
    })
    applyGatewayEvent({
      type: 'text.delta',
      session_id: 's1',
      run_id: 'r1',
      text: 'after',
      sequence: 1
    })
    applyGatewayEvent({
      type: 'done',
      session_id: 's1',
      run_id: 'r1',
      final_text: 'before after',
      exit_reason: 'done'
    })

    const items = chatItems.get()
    expect(items[0]).toMatchObject({
      kind: 'assistant',
      runId: 'r1',
      text: 'before ',
      streaming: false
    })
    expect(items[1]).toMatchObject({
      kind: 'tool-call',
      toolCallId: 'tc1'
    })
    expect(items[2]).toMatchObject({
      kind: 'assistant',
      runId: 'r1',
      text: 'after',
      streaming: false
    })
  })

  it('updates activity todos from todo_write tool calls', () => {
    applyGatewayEvent({
      type: 'tool.call',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'todo1',
      tool_name: 'todo_write',
      arguments: {
        todos: [
          { id: '1', content: 'Implement registry', status: 'in_progress' },
          { id: '2', content: 'Wire gateway', status: 'pending' }
        ]
      },
      iteration: 1
    })

    expect(activityState.get().todos).toEqual([
      { id: '1', content: 'Implement registry', status: 'in_progress' },
      { id: '2', content: 'Wire gateway', status: 'pending' }
    ])

    applyGatewayEvent({
      type: 'tool.call',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'todo2',
      tool_name: 'todo_write',
      arguments: {
        todos: [
          { id: '1', content: 'Implement registry', status: 'completed' },
          { id: '2', content: 'Wire gateway', status: 'completed' }
        ]
      },
      iteration: 1
    })

    expect(activityState.get().todos).toEqual([])
  })

  it('appends the final assistant suffix below interleaved shell output even without a post-tool delta', () => {
    applyGatewayEvent({
      type: 'text.delta',
      session_id: 's1',
      run_id: 'r1',
      text: 'before ',
      sequence: 0
    })
    applyGatewayEvent({
      type: 'tool.call',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'tc1',
      tool_name: 'shell',
      arguments: { command: 'echo hi' },
      iteration: 1
    })
    applyGatewayEvent({
      type: 'tool.result',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'tc1',
      tool_name: 'shell',
      content: 'hi',
      is_error: false,
      iteration: 1
    })
    applyGatewayEvent({
      type: 'done',
      session_id: 's1',
      run_id: 'r1',
      final_text: 'before after',
      exit_reason: 'done'
    })

    const items = chatItems.get()
    expect(items[0]).toMatchObject({
      kind: 'assistant',
      runId: 'r1',
      text: 'before ',
      streaming: false
    })
    expect(items[1]).toMatchObject({
      kind: 'tool-call',
      toolCallId: 'tc1'
    })
    expect(items[2]).toMatchObject({
      kind: 'assistant',
      runId: 'r1',
      text: 'after',
      streaming: false
    })
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
