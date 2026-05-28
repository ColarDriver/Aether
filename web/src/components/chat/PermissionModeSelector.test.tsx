// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { PermissionModeSelector } from './PermissionModeSelector'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('PermissionModeSelector', () => {
  it('selects ordinary permission modes directly', async () => {
    const onChange = vi.fn()
    render(<PermissionModeSelector value="default" onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'Permission mode: Ask before tools' }))
    fireEvent.click(screen.getByRole('menuitemradio', { name: /Auto-accept edits/ }))

    await waitFor(() => expect(onChange).toHaveBeenCalledWith('acceptEdits'))
  })

  it('requires confirmation before bypassing permissions', async () => {
    const onChange = vi.fn()
    render(<PermissionModeSelector value="default" onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'Permission mode: Ask before tools' }))
    fireEvent.click(screen.getByRole('menuitemradio', { name: /Bypass permissions/ }))

    expect(onChange).not.toHaveBeenCalled()
    const dialog = screen.getByRole('dialog', { name: 'Bypass permissions?' })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Bypass' }))

    await waitFor(() => expect(onChange).toHaveBeenCalledWith('bypassPermissions'))
  })
})
