import { describe, expect, it } from 'vitest'
import type { ChatBlock } from './blocks'
import {
  extractDiffFromMetadata,
  firstNonEmptyLine,
  jsonPreview,
  parseAskUserQuestions,
  recordFromUnknown,
  stringFromUnknown,
} from './content'
import { isBlockKind, isPromptBlock, isToolCallBlock } from './blockGuards'

const base = {
  id: 'block-1',
  sessionId: 'session-1',
  runId: 'run-1',
  timestamp: 1,
  source: 'live' as const,
}

describe('chat rendering blocks', () => {
  it('supports every required block kind as a discriminated union', () => {
    const blocks: ChatBlock[] = [
      { ...base, kind: 'user_message', content: 'hello' },
      { ...base, id: 'a', kind: 'assistant_message', content: 'hi', isStreaming: true },
      { ...base, id: 't', kind: 'thinking', content: 'thinking', isActive: true },
      { ...base, id: 'tc', kind: 'tool_call', toolCallId: 'call-1', toolName: 'read_file', arguments: {}, status: 'running' },
      { ...base, id: 'tr', kind: 'tool_result', toolCallId: 'call-1', content: 'done', isError: false, metadata: {} },
      { ...base, id: 'd', kind: 'diff', origin: 'tool_result', path: 'app.py', diff: '-old\n+new' },
      { ...base, id: 'p', kind: 'permission_request', promptId: 'permission-1', toolName: 'write_file', arguments: {}, state: 'pending' },
      { ...base, id: 'ap', kind: 'approval_request', promptId: 'approval-1', approvalKind: 'plan', questions: [], state: 'pending' },
      { ...base, id: 'q', kind: 'ask_user_question', questions: [{ question: 'Continue?' }], state: 'pending' },
      { ...base, id: 's', kind: 'streaming_status', state: 'thinking', tokens: { output_tokens: 12 } },
      { ...base, id: 'sys', kind: 'system_notice', content: 'cancelled' },
      { ...base, id: 'e', kind: 'error', message: 'failed', code: 'run_failed' },
    ]

    expect(blocks.map((block) => block.kind)).toEqual([
      'user_message',
      'assistant_message',
      'thinking',
      'tool_call',
      'tool_result',
      'diff',
      'permission_request',
      'approval_request',
      'ask_user_question',
      'streaming_status',
      'system_notice',
      'error',
    ])
    expect(isToolCallBlock(blocks[3])).toBe(true)
    expect(isPromptBlock(blocks[6])).toBe(true)
    expect(isBlockKind(blocks[1], 'assistant_message')).toBe(true)
  })
})

describe('chat rendering content helpers', () => {
  it('normalizes unknown values into readable strings', () => {
    expect(stringFromUnknown('hello')).toBe('hello')
    expect(stringFromUnknown(null)).toBe('')
    expect(stringFromUnknown({ text: 'from text field' })).toBe('from text field')
    expect(stringFromUnknown({ ok: true })).toContain('"ok": true')
    expect(stringFromUnknown(['a', { message: 'b' }])).toBe('a\nb')
  })

  it('returns records only for plain object-like values', () => {
    expect(recordFromUnknown({ ok: true })).toEqual({ ok: true })
    expect(recordFromUnknown(null)).toEqual({})
    expect(recordFromUnknown(['x'])).toEqual({})
    expect(recordFromUnknown('x')).toEqual({})
  })

  it('creates bounded JSON previews', () => {
    expect(jsonPreview({ value: 'abcdef' }, { maxChars: 10 })).toHaveLength(10)
    expect(jsonPreview('short', { maxChars: 10 })).toBe('short')
  })

  it('returns the first meaningful line', () => {
    expect(firstNonEmptyLine('\n  \n first \n second')).toBe('first')
  })

  it('extracts diff content from common metadata shapes', () => {
    expect(extractDiffFromMetadata({ diff: '-old\n+new', path: 'app.py' })).toEqual({
      path: 'app.py',
      diff: '-old\n+new',
      oldText: null,
      newText: null,
      language: null,
    })
    expect(extractDiffFromMetadata({ old_string: 'old', new_string: 'new', file_path: 'a.ts' })).toEqual({
      path: 'a.ts',
      diff: null,
      oldText: 'old',
      newText: 'new',
      language: null,
    })
    expect(extractDiffFromMetadata({ path: 'no-diff.py' })).toBeNull()
  })

  it('parses ask_user_question array input', () => {
    expect(parseAskUserQuestions({
      questions: [
        {
          header: 'Mode',
          question: 'Which mode?',
          options: [{ label: 'Fast', description: 'Less detail' }, 'Careful'],
          multi_select: true,
        },
      ],
    })).toEqual([
      {
        header: 'Mode',
        question: 'Which mode?',
        options: [
          { label: 'Fast', description: 'Less detail' },
          { label: 'Careful' },
        ],
        multiSelect: true,
      },
    ])
  })

  it('parses ask_user_question single-question input', () => {
    expect(parseAskUserQuestions({ question: 'Continue?', options: ['Yes', 'No'] })).toEqual([
      {
        question: 'Continue?',
        options: [{ label: 'Yes' }, { label: 'No' }],
      },
    ])
    expect(parseAskUserQuestions({ questions: [] })).toEqual([])
    expect(parseAskUserQuestions(null)).toEqual([])
  })
})
