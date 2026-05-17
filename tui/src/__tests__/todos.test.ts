import { describe, expect, it } from 'vitest'

import { formatTodoPreviewLines } from '../lib/todos.js'

describe('formatTodoPreviewLines', () => {
  it('uses ascii-safe checklist glyphs when requested', () => {
    expect(
      formatTodoPreviewLines(
        [{ id: '1', content: 'Verify fallback', status: 'in_progress' }],
        { ascii: true }
      )
    ).toEqual(['- [*] Verify fallback'])
  })

  it('summarizes hidden todos instead of silently dropping them', () => {
    const lines = formatTodoPreviewLines(
      [
        { id: '1', content: 'Visible', status: 'in_progress' },
        { id: '2', content: 'Hidden pending', status: 'pending' },
        { id: '3', content: 'Hidden completed', status: 'completed' }
      ],
      { limit: 1 }
    )
    expect(lines).toEqual([
      '└ ■ Visible',
      '  … +1 pending, 1 completed'
    ])
  })

  it('truncates long todo content to the requested display width', () => {
    const lines = formatTodoPreviewLines(
      [{ id: '1', content: 'A very long todo item label', status: 'pending' }],
      { width: 14 }
    )
    expect(lines[0]).toBe('└ □ A very lo…')
  })
})
