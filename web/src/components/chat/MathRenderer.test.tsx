// @vitest-environment jsdom

import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { MathRenderer } from './MathRenderer'

afterEach(cleanup)

describe('MathRenderer', () => {
  it('renders inline KaTeX output', async () => {
    const { container } = render(<MathRenderer source={'x^2 + y^2'} />)

    expect(container.querySelector('.math-renderer-loading')).toBeTruthy()
    await waitFor(() => expect(container.querySelector('.katex')).toBeTruthy())
    expect(container.querySelector('.math-renderer-inline')).toBeTruthy()
  })

  it('renders display KaTeX output', async () => {
    const { container } = render(<MathRenderer display source={'E = mc^2'} />)

    expect(container.querySelector('.math-renderer-loading')).toBeTruthy()
    await waitFor(() => expect(container.querySelector('.katex-display')).toBeTruthy())
    expect(container.querySelector('.math-renderer-display')).toBeTruthy()
  })
})
