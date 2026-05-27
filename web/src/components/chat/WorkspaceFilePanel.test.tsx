// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { WorkspaceFilePanel } from './WorkspaceFilePanel'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('WorkspaceFilePanel', () => {
  it('renders markdown file content in the dedicated preview panel', () => {
    render(
      <WorkspaceFilePanel
        preview={{
          path: 'README.md',
          loading: false,
          error: null,
          file: {
            root: '/workspace/Aether',
            path: 'README.md',
            name: 'README.md',
            content: '# Aether\n\nWorkspace preview body.',
            size_bytes: 32,
            updated_at: 1,
            language: 'markdown',
            truncated: false,
            binary: false,
          },
        }}
        onClose={vi.fn()}
      />,
    )

    expect(screen.getByLabelText('Workspace file preview')).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Aether' })).toBeTruthy()
    expect(screen.getByText('Workspace preview body.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Close file preview' })).toBeTruthy()
  })

  it('saves edited text files through the panel action', async () => {
    const onSave = vi.fn().mockResolvedValue({
      root: '/workspace/Aether',
      path: 'src/app.ts',
      name: 'app.ts',
      content: 'const answer = 43\n',
      size_bytes: 18,
      updated_at: 2,
      language: 'typescript',
      truncated: false,
      binary: false,
    })

    render(
      <WorkspaceFilePanel
        preview={{
          path: 'src/app.ts',
          loading: false,
          error: null,
          file: {
            root: '/workspace/Aether',
            path: 'src/app.ts',
            name: 'app.ts',
            content: 'const answer = 42\n',
            size_bytes: 18,
            updated_at: 1,
            language: 'typescript',
            truncated: false,
            binary: false,
          },
        }}
        onSave={onSave}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit workspace file' }))
    const editor = screen.getByLabelText('Workspace file editor') as HTMLTextAreaElement
    fireEvent.change(editor, { target: { value: 'const answer = 43\n' } })
    expect(screen.getByText('modified')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Save workspace file' }))

    await waitFor(() => expect(onSave).toHaveBeenCalledWith('src/app.ts', 'const answer = 43\n'))
    await waitFor(() => expect(screen.queryByLabelText('Workspace file editor')).toBeNull())
  })

  it('does not offer editing for truncated files', () => {
    render(
      <WorkspaceFilePanel
        preview={{
          path: 'uv.lock',
          loading: false,
          error: null,
          file: {
            root: '/workspace/Aether',
            path: 'uv.lock',
            name: 'uv.lock',
            content: 'partial',
            size_bytes: 300000,
            updated_at: 1,
            language: 'text',
            truncated: true,
            binary: false,
          },
        }}
        onSave={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: 'Edit workspace file' })).toBeNull()
    expect(screen.getByText('read only')).toBeTruthy()
  })

  it('previews image files as read-only media', async () => {
    const createObjectURL = vi.fn(() => 'blob:workspace-logo')
    const revokeObjectURL = vi.fn()
    vi.stubGlobal('URL', { ...URL, createObjectURL, revokeObjectURL })
    vi.spyOn(api, 'workspaceFileBlob').mockResolvedValue(new Blob(['png'], { type: 'image/png' }))

    render(
      <WorkspaceFilePanel
        preview={{
          path: 'assets/logo.png',
          loading: false,
          error: null,
          file: {
            root: '/workspace/Aether',
            path: 'assets/logo.png',
            name: 'logo.png',
            content: '',
            size_bytes: 8,
            updated_at: 1,
            language: 'image',
            mime_type: 'image/png',
            truncated: false,
            binary: true,
          },
        }}
        onSave={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: 'Edit workspace file' })).toBeNull()
    expect(screen.getByText('read only')).toBeTruthy()
    expect(screen.getByText('image/png')).toBeTruthy()
    await waitFor(() => expect(api.workspaceFileBlob).toHaveBeenCalledWith('assets/logo.png'))
    const image = await screen.findByRole('img', { name: 'logo.png' })
    expect(image.getAttribute('src')).toBe('blob:workspace-logo')
  })

  it('renders code and loading states without requiring a chat session', () => {
    const { rerender } = render(
      <WorkspaceFilePanel
        preview={{ path: 'src/app.ts', file: null, loading: true, error: null }}
      />,
    )
    expect(screen.getByRole('status').textContent).toContain('Loading file')

    rerender(
      <WorkspaceFilePanel
        preview={{
          path: 'src/app.ts',
          loading: false,
          error: null,
          file: {
            root: '/workspace/Aether',
            path: 'src/app.ts',
            name: 'app.ts',
            content: 'const answer = 42\n',
            size_bytes: 18,
            updated_at: 1,
            language: 'typescript',
            truncated: false,
            binary: false,
          },
        }}
      />,
    )

    expect(screen.getByText('const')).toBeTruthy()
    expect(screen.getByText(/answer/)).toBeTruthy()
  })
})
