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
  it('promotes stable prefix by rendered row budget instead of item count', () => {
    const items = [note('n1', 20), note('n2', 5), note('n3', 2), note('n4', 3), note('n5', 4)]

    const { staticItems, liveItems } = splitStaticPrefix(items, true, 80, 12)

    expect(staticItems.map((item) => item.id)).toEqual(['n1', 'n2'])
    expect(liveItems.map((item) => item.id)).toEqual(['n3', 'n4', 'n5'])
  })

  it('keeps at least one stable item live even when it exceeds the row budget', () => {
    const items = [note('n1', 100)]

    const { staticItems, liveItems } = splitStaticPrefix(items, true, 80, 12)

    expect(staticItems).toEqual([])
    expect(liveItems.map((item) => item.id)).toEqual(['n1'])
  })

  it('leaves all items live when static scrollback is disabled', () => {
    const items = [note('n1', 1), note('n2', 1), note('n3', 1)]

    const { staticItems, liveItems } = splitStaticPrefix(items, false, 80, 1)

    expect(staticItems).toEqual([])
    expect(liveItems).toEqual(items)
  })

  it('does not promote unstable streaming tail into static scrollback', () => {
    const items: ChatItem[] = [
      note('n1', 20),
      {
        kind: 'assistant',
        id: 'a1',
        runId: 'r1',
        text: 'streaming',
        streaming: true,
        ts: 2
      }
    ]

    const { staticItems, liveItems } = splitStaticPrefix(items, true, 80, 1)

    expect(staticItems).toEqual([])
    expect(liveItems.map((item) => item.id)).toEqual(['n1', 'a1'])
  })
})
