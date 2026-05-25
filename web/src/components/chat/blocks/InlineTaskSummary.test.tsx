// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { InlineTaskSummary } from './InlineTaskSummary'

afterEach(cleanup)

describe('InlineTaskSummary', () => {
  it('shimmers active subagent task statuses', () => {
    render(<InlineTaskSummary title="Inspect renderer" status="running" taskId="task-1" />)

    expect(screen.getByText('running').className).toContain('aether-shimmer-text')
  })

  it('does not shimmer completed subagent task statuses', () => {
    render(<InlineTaskSummary title="Inspect renderer" status="completed" taskId="task-1" />)

    expect(screen.getByText('completed').className).not.toContain('aether-shimmer-text')
  })
})
