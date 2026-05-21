// @vitest-environment jsdom

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { DiagnosticsView } from './DiagnosticsView'

describe('DiagnosticsView', () => {
  it('renders runtime health and service availability', () => {
    render(
      <DiagnosticsView
        health={{
          status: 'ok',
          runtime: { python_version: '3.13', platform: 'linux', implementation: 'CPython' },
          diagnostics: { enabled: true, pending_count: 0 },
          services: [{ name: 'sessions', available: true, status: 'ok', detail: 'ready' }],
        }}
      />,
    )

    expect(screen.getByRole('heading', { name: 'Diagnostics' })).toBeTruthy()
    expect(screen.getByText('3.13')).toBeTruthy()
    expect(screen.getByText('sessions')).toBeTruthy()
    expect(screen.getByText('available')).toBeTruthy()
  })
})
