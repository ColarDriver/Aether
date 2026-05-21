// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { LogsView } from './LogsView'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('LogsView', () => {
  it('renders log lines and refetches with selected filters', async () => {
    const logFiles = vi.spyOn(api, 'logFiles').mockResolvedValue({
      files: [{ key: 'gateway', name: 'gateway_crash.log', path: '/tmp/gateway_crash.log', exists: true, size_bytes: 20 }],
    })
    const logs = vi.spyOn(api, 'logs').mockResolvedValue({
      file: 'gateway',
      path: '/tmp/gateway_crash.log',
      exists: true,
      lines: ['INFO gateway ready', 'ERROR gateway failed'],
      available_files: [],
    })

    render(<LogsView />)

    expect(await screen.findByText('ERROR gateway failed')).toBeTruthy()
    fireEvent.change(screen.getByLabelText('Level'), { target: { value: 'ERROR' } })

    await waitFor(() => expect(logs).toHaveBeenLastCalledWith({ file: 'gateway', lines: 100, level: 'ERROR', search: undefined }))
    expect(logFiles).toHaveBeenCalled()
  })
})
