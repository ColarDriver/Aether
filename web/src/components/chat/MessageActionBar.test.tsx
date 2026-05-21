// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { MessageActionBar } from './MessageActionBar'

afterEach(cleanup)

describe('MessageActionBar', () => {
  it('renders a copy action when copy text is available', () => {
    render(<MessageActionBar copyText="reply" copyLabel="Copy reply" align="end" />)

    expect(screen.getByRole('button', { name: 'Copy reply' })).toBeTruthy()
    expect(document.querySelector('[data-message-actions]')?.className).toContain('message-action-bar-end')
  })

  it('does not render an empty copy action', () => {
    render(<MessageActionBar copyText="   " copyLabel="Copy prompt" />)

    expect(screen.queryByRole('button', { name: 'Copy prompt' })).toBeNull()
  })
})
