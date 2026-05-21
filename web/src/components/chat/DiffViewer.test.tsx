// @vitest-environment jsdom

import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { DiffViewer } from './DiffViewer'

describe('DiffViewer', () => {
  it('marks added and removed lines', () => {
    render(<DiffViewer diff={'@@ file\n-old\n+new'} />)

    expect(document.querySelector('.diff-line-remove')?.textContent).toContain('-old')
    expect(document.querySelector('.diff-line-add')?.textContent).toContain('+new')
  })
})
