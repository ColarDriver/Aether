// @vitest-environment jsdom

import { describe, expect, it } from 'vitest'
import type { TranscriptMessage } from '../api/types'
import { transcriptToChatState } from './chatStore'

describe('transcriptToChatState', () => {
  it('reconstructs persisted tool blocks from assistant tool calls and tool results', () => {
    const transcript: TranscriptMessage[] = [
      { role: 'user', text: 'edit file' },
      {
        role: 'assistant',
        text: 'I will edit it.',
        tool_calls: [
          { id: 'call-1', name: 'file_edit', arguments: { path: 'app.py' } },
        ],
      },
      {
        role: 'tool',
        tool_call_id: 'call-1',
        text: 'updated app.py',
        metadata: { diff: '-old\n+new' },
      },
    ]

    const result = transcriptToChatState('session-1', transcript)

    expect(result.messages).toHaveLength(2)
    expect(result.messages[1]).toMatchObject({ role: 'assistant', text: 'I will edit it.' })
    expect(result.tools).toHaveLength(1)
    expect(result.tools[0]).toMatchObject({
      sessionId: 'session-1',
      toolCallId: 'call-1',
      toolName: 'file_edit',
      arguments: { path: 'app.py' },
      status: 'finished',
      content: 'updated app.py',
      metadata: { diff: '-old\n+new' },
    })
  })

  it('keeps orphan tool result messages visible as tool blocks', () => {
    const result = transcriptToChatState('session-2', [
      { role: 'tool', name: 'shell', tool_call_id: 'missing', text: 'done', is_error: true },
    ])

    expect(result.messages).toHaveLength(0)
    expect(result.tools).toHaveLength(1)
    expect(result.tools[0]).toMatchObject({ toolName: 'shell', status: 'finished', isError: true })
  })
})
