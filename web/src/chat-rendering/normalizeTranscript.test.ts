import { describe, expect, it } from 'vitest'
import type { TranscriptMessage } from '../api/types'
import { normalizeTranscript } from './normalizeTranscript'

describe('normalizeTranscript', () => {
  it('normalizes text, tool calls, tool results, and diffs into ordered blocks', () => {
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
        name: 'file_edit',
        tool_call_id: 'call-1',
        text: 'updated app.py',
        metadata: { diff: '-old\n+new', path: 'app.py' },
      },
    ]

    const blocks = normalizeTranscript('session-1', transcript)

    expect(blocks.map((block) => block.kind)).toEqual([
      'user_message',
      'assistant_message',
      'tool_call',
      'tool_result',
      'diff',
    ])
    expect(blocks[2]).toMatchObject({
      kind: 'tool_call',
      toolCallId: 'call-1',
      toolName: 'file_edit',
      arguments: { path: 'app.py' },
    })
    expect(blocks[4]).toMatchObject({
      kind: 'diff',
      origin: 'tool_result',
      path: 'app.py',
      diff: '-old\n+new',
    })
  })

  it('keeps orphan tool results visible', () => {
    const blocks = normalizeTranscript('session-2', [
      { role: 'tool', name: 'shell', tool_call_id: 'missing', text: 'done', is_error: true },
    ])

    expect(blocks).toHaveLength(1)
    expect(blocks[0]).toMatchObject({
      kind: 'tool_result',
      toolName: 'shell',
      toolCallId: 'missing',
      isError: true,
    })
  })

  it('normalizes ask_user_question calls as interaction blocks', () => {
    const blocks = normalizeTranscript('session-3', [
      {
        role: 'assistant',
        tool_calls: [
          {
            id: 'ask-1',
            name: 'ask_user_question',
            arguments: { question: 'Continue?', options: ['Yes', 'No'] },
          },
        ],
      },
    ])

    expect(blocks).toHaveLength(1)
    expect(blocks[0]).toMatchObject({
      kind: 'ask_user_question',
      toolCallId: 'ask-1',
      questions: [{ question: 'Continue?', options: [{ label: 'Yes' }, { label: 'No' }] }],
    })
  })
})
