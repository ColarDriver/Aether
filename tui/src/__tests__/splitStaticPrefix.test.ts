import { describe, expect, it } from 'vitest'

import { splitStaticPrefix } from '../components/ChatTranscript.js'
import type { ChatItem } from '../store/chatStore.js'

function note(id: string, rows: number): ChatItem {
  return {
    kind: 'note',
    id,
    text: Array.from({ length: rows }, (_, index) => `${id}-${index}`).join('\n'),
    level: 'info',
    ts: Number(id.replace(/\D/g, '') || 0)
  }
}

describe('splitStaticPrefix', () => {
  it('promotes all stable items to static scrollback', () => {
    const items = [note('n1', 20), note('n2', 5), note('n3', 2)]

    const { staticItems, liveItems } = splitStaticPrefix(items, true, 80, 12)

    expect(staticItems.map((item) => item.id)).toEqual(['n1', 'n2', 'n3'])
    expect(liveItems).toEqual([])
  })

  it('does not duplicate the just-submitted user echo in the live region while streaming', () => {
    const items: ChatItem[] = [
      note('n1', 2),
      { kind: 'user', id: 'u1', text: '还有吗?', ts: 2 },
      {
        kind: 'assistant',
        id: 'a1',
        runId: 'r1',
        text: '还有。',
        streaming: true,
        ts: 3
      }
    ]

    const { staticItems, liveItems } = splitStaticPrefix(items, true, 80, 12)

    expect(staticItems.map((item) => item.id)).toEqual(['n1', 'u1'])
    expect(liveItems.map((item) => item.id)).toEqual(['a1'])
  })

  it('keeps a completed assistant turn live with its footer', () => {
    const items: ChatItem[] = [
      note('n1', 20),
      {
        kind: 'assistant',
        id: 'a1',
        runId: 'r1',
        text: ['line one', 'line two', 'final line'].join('\n'),
        streaming: false,
        ts: 2
      },
      {
        kind: 'note',
        id: 'n2',
        text: '✓ done · 1.20s',
        level: 'info',
        ts: 3
      }
    ]

    const { staticItems, liveItems } = splitStaticPrefix(items, true, 80, 1)

    expect(staticItems.map((item) => item.id)).toEqual(['n1', 'a1', 'n2'])
    expect(liveItems).toEqual([])
  })

  it('leaves all items live when static scrollback is disabled', () => {
    const items = [note('n1', 1), note('n2', 1), note('n3', 1)]

    const { staticItems, liveItems } = splitStaticPrefix(items, false, 80, 1)

    expect(staticItems).toEqual([])
    expect(liveItems).toEqual(items)
  })

  it('keeps only the incomplete streaming assistant suffix live', () => {
    const items: ChatItem[] = [
      note('n1', 20),
      {
        kind: 'assistant',
        id: 'a1',
        runId: 'r1',
        text: 'line one\nline two\npartial',
        streaming: true,
        ts: 2
      }
    ]
    const chunkedRunIds = new Set<string>()

    const { staticItems, liveItems } = splitStaticPrefix(items, true, 80, 1, chunkedRunIds)

    expect(staticItems.map((item) => item.id)).toEqual(['n1', 'a1', 'a1:chunk:1'])
    expect(liveItems).toHaveLength(1)
    expect(liveItems[0]).toMatchObject({ id: 'a1:chunk:2', text: 'partial' })
    expect(chunkedRunIds.has('r1')).toBe(true)
  })
})
