// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { CodeBlock } from './CodeBlock'

afterEach(() => cleanup())

describe('CodeBlock', () => {
  it('renders a label and lightweight JSON syntax spans', () => {
    render(<CodeBlock code={'{"ok": true, "count": 2}'} language="json" />)

    expect(screen.getByText('json')).toBeTruthy()
    expect(document.querySelector('.syntax-string')?.textContent).toBe('"ok"')
    expect(document.querySelector('.syntax-boolean')?.textContent).toBe('true')
    expect(document.querySelector('.syntax-number')?.textContent).toBe('2')
  })

  it('colors TypeScript functions and types', () => {
    render(<CodeBlock code={'const ref = useRef<HTMLDivElement>(null)'} language="tsx" />)

    expect(document.querySelector('.syntax-function')?.textContent).toBe('useRef')
    expect(document.querySelector('.syntax-type')?.textContent).toBe('HTMLDivElement')
    expect(document.querySelector('.syntax-null')?.textContent).toBe('null')
  })

  it('supports custom titles for shell and tool previews', () => {
    render(<CodeBlock code="python app.py" language="shell" title="Command" />)

    expect(screen.getByText('Command')).toBeTruthy()
    expect(document.querySelector('.code-block-body')?.textContent).toContain('python app.py')
  })
})
