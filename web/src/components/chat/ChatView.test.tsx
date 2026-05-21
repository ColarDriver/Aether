// @vitest-environment jsdom

import { describe, expect, it } from 'vitest'
import { isNearChatBottom } from './ChatView'

describe('isNearChatBottom', () => {
  it('treats the scroll position as bottom only within the follow threshold', () => {
    expect(isNearChatBottom({ scrollHeight: 1000, scrollTop: 540, clientHeight: 420 })).toBe(true)
    expect(isNearChatBottom({ scrollHeight: 1000, scrollTop: 500, clientHeight: 420 })).toBe(false)
  })
})
