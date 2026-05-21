// @vitest-environment jsdom

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MessageList } from './MessageList'

describe('MessageList', () => {
  it('renders chat messages and streaming state', () => {
    render(
      <MessageList
        messages={[
          { id: 'u1', role: 'user', text: 'hello' },
          { id: 'a1', role: 'assistant', text: 'hi', isStreaming: true },
        ]}
      />,
    )

    expect(screen.getByText('hello')).toBeTruthy()
    expect(screen.getByText('hi')).toBeTruthy()
    expect(document.querySelector('.streaming-caret')).toBeTruthy()
  })
})
