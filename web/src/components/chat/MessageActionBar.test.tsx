// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { MessageActionBar } from './MessageActionBar'

afterEach(cleanup)

describe('MessageActionBar', () => {
  it('renders a copy action when copy text is available', () => {
    render(<MessageActionBar copyText="reply" copyLabel="Copy reply" align="end" />)

    expect(screen.getByRole('button', { name: 'Copy reply' })).toBeTruthy()
    expect(document.querySelector('[data-message-actions]')?.className).toContain('message-action-bar-end')
  })

  it('renders explicit message actions without copy text', () => {
    const onQuote = vi.fn()
    render(<MessageActionBar copyText="   " copyLabel="Copy prompt" actions={[{ kind: 'quote', label: 'Quote prompt', onClick: onQuote }]} />)

    fireEvent.click(screen.getByRole('button', { name: 'Quote prompt' }))

    expect(screen.queryByRole('button', { name: 'Copy prompt' })).toBeNull()
    expect(onQuote).toHaveBeenCalledTimes(1)
  })

  it('does not render an empty action bar', () => {
    render(<MessageActionBar copyText="   " copyLabel="Copy prompt" />)

    expect(screen.queryByRole('button', { name: 'Copy prompt' })).toBeNull()
    expect(document.querySelector('[data-message-actions]')).toBeNull()
  })
})
