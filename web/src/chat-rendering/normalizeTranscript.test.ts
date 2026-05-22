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
      {
        role: 'tool',
        name: 'ask_user_question',
        tool_call_id: 'ask-1',
        text: 'answered',
        metadata: { answer_pairs: [{ label: 'Continue?', value: 'Yes' }] },
      },
    ])

    expect(blocks).toHaveLength(1)
    expect(blocks[0]).toMatchObject({
      kind: 'ask_user_question',
      toolCallId: 'ask-1',
      questions: [{ question: 'Continue?', options: [{ label: 'Yes' }, { label: 'No' }] }],
      state: 'answered',
      answers: { 'Continue?': 'Yes' },
    })
  })

  it('keeps user attachments from transcript fields and metadata', () => {
    const blocks = normalizeTranscript('session-4', [
      {
        role: 'user',
        text: 'inspect this',
        attachments: [
          { type: 'file', name: 'app.ts', path: 'src/app.ts', line_start: 3, line_end: 5 },
        ],
        metadata: {
          displayAttachments: [
            { type: 'image', name: 'chart.png', data: 'data:image/png;base64,abc', mimeType: 'image/png' },
          ],
        },
      },
    ])

    expect(blocks[0]).toMatchObject({
      kind: 'user_message',
      content: 'inspect this',
      attachments: [
        { type: 'file', name: 'app.ts', path: 'src/app.ts', lineStart: 3, lineEnd: 5 },
        { type: 'image', name: 'chart.png', data: 'data:image/png;base64,abc', mimeType: 'image/png' },
      ],
    })
  })

  it('normalizes diagnostics user turns without exposing raw XML as a user prompt', () => {
    const blocks = normalizeTranscript('session-6', [
      {
        role: 'user',
        text: '<diagnostics>\n## src/app.py\n  ERROR   4:8  pyright [reportGeneralTypeIssues]: bad type\n</diagnostics>',
        metadata: { source: 'diagnostics' },
      },
    ])

    expect(blocks).toHaveLength(1)
    expect(blocks[0]).toMatchObject({
      kind: 'diagnostics',
      files: [
        {
          path: 'src/app.py',
          diagnostics: [
            { severity: 'error', line: 4, column: 8, source: 'pyright', code: 'reportGeneralTypeIssues', message: 'bad type' },
          ],
        },
      ],
    })
  })

  it('normalizes task-notification user turns without exposing raw XML', () => {
    const blocks = normalizeTranscript('session-5', [
      {
        role: 'user',
        text: '<task-notification>\n  <task_id>task-1</task_id>\n  <subagent_type>explorer</subagent_type>\n  <status>completed</status>\n  <duration_seconds>4.0</duration_seconds>\n  <summary>Done</summary>\n</task-notification>',
        metadata: { source: 'task_notification' },
      },
    ])

    expect(blocks).toHaveLength(1)
    expect(blocks[0]).toMatchObject({
      kind: 'task_notification',
      taskId: 'task-1',
      subagentType: 'explorer',
      status: 'completed',
      durationSeconds: 4,
      summary: 'Done',
    })
  })
})
