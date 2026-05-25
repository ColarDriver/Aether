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
          sessionId: 's1',
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

  it('uses the shared permission preview renderer for diff and arguments', () => {
    render(
      <PermissionDialog
        prompt={{
          promptId: 'p1',
          sessionId: 's1',
          runId: 'r1',
          request: {
            tool_name: 'write_file',
            arguments: { path: 'app.py' },
            preview: { title: 'Edit file', diff: '@@ -1 +1 @@\n-old\n+new' },
          },
        }}
        onAllow={() => undefined}
        onAllowSession={() => undefined}
        onDeny={() => undefined}
      />,
    )

    expect(screen.getByRole('table', { name: 'Code diff' })).toBeTruthy()
    expect(screen.getByText(/app.py/)).toBeTruthy()
  })
})
