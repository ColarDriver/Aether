// @vitest-environment jsdom

import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { PermissionDialog } from './PermissionDialog'

describe('PermissionDialog', () => {
  it('sends allow once', () => {
    const onAllow = vi.fn()
    render(
      <PermissionDialog
        prompt={{
          promptId: 'p1',
          runId: 'r1',
          request: {
            tool_name: 'shell',
            arguments: { command: 'echo ok' },
            risk: 'medium',
            allow_session: true,
            preview: { title: 'Run command', command: 'echo ok' },
          },
        }}
        onAllow={onAllow}
        onAllowSession={() => undefined}
        onDeny={() => undefined}
      />,
    )

    fireEvent.click(screen.getByText('Allow once'))
    expect(onAllow).toHaveBeenCalledOnce()
  })
})
