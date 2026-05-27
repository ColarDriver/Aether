// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { WorkspaceView } from './WorkspaceView'

const rootTree = {
  root: '/workspace/Aether',
  path: '',
  parent_path: null,
  entries: [
    { path: 'aether', name: 'aether', kind: 'directory' as const, updated_at: 1 },
    { path: 'README.md', name: 'README.md', kind: 'file' as const, size_bytes: 18, updated_at: 2 },
  ],
}

const nestedTree = {
  root: '/workspace/Aether',
  path: 'aether',
  parent_path: '',
  entries: [
    { path: 'aether/app.py', name: 'app.py', kind: 'file' as const, size_bytes: 11, updated_at: 3 },
  ],
}

const readme = {
  root: '/workspace/Aether',
  path: 'README.md',
  name: 'README.md',
  content: '# Aether\n\nHello.',
  size_bytes: 18,
  updated_at: 2,
  language: 'markdown',
  truncated: false,
  binary: false,
}

const savedReadme = {
  ...readme,
  content: '# Aether\n\nSaved.',
  size_bytes: 18,
  updated_at: 4,
}

const appFile = {
  root: '/workspace/Aether',
  path: 'aether/app.py',
  name: 'app.py',
  content: 'print("hi")\n',
  size_bytes: 12,
  updated_at: 3,
  language: 'python',
  truncated: false,
  binary: false,
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('WorkspaceView', () => {
  it('browses, searches, previews, and edits workspace files through the shared panels', async () => {
    vi.spyOn(api, 'workspaceTree').mockImplementation(async (path = '') => (path === 'aether' ? nestedTree : rootTree))
    vi.spyOn(api, 'workspaceFile').mockImplementation(async (path: string) => (path === 'aether/app.py' ? appFile : readme))
    const saveFile = vi.spyOn(api, 'workspaceSaveFile').mockResolvedValue(savedReadme)
    const search = vi.spyOn(api, 'workspaceSearch').mockResolvedValue({ root: '/workspace/Aether', query: 'app', entries: nestedTree.entries })

    render(<WorkspaceView />)

    const workspaceFiles = screen.getByLabelText('Workspace files')
    expect(await within(workspaceFiles).findByTitle('README.md')).toBeTruthy()
    expect(await screen.findByText('Hello.')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Edit workspace file' }))
    const editor = screen.getByLabelText('Workspace file editor')
    fireEvent.change(editor, { target: { value: '# Aether\n\nSaved.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save workspace file' }))

    await waitFor(() => expect(saveFile).toHaveBeenCalledWith('README.md', '# Aether\n\nSaved.'))
    expect(await screen.findByText('Saved.')).toBeTruthy()

    fireEvent.click(screen.getByTitle('aether'))
    expect(await screen.findByTitle('aether/app.py')).toBeTruthy()

    fireEvent.change(screen.getByPlaceholderText('Search files'), { target: { value: 'app' } })
    fireEvent.click(screen.getByRole('button', { name: 'Search' }))

    await waitFor(() => expect(search).toHaveBeenCalledWith('app', 80))
    fireEvent.click(screen.getByTitle('aether/app.py'))
    expect(await screen.findByText(/print/)).toBeTruthy()
  })

  it('closes the preview when the selected workspace file is deleted', async () => {
    vi.spyOn(api, 'workspaceTree').mockResolvedValue(rootTree)
    vi.spyOn(api, 'workspaceFile').mockResolvedValue(readme)
    const deletePath = vi.spyOn(api, 'workspaceDeletePath').mockResolvedValue(undefined)

    render(<WorkspaceView />)

    const workspaceFiles = screen.getByLabelText('Workspace files')
    fireEvent.click(await within(workspaceFiles).findByTitle('README.md'))
    expect(await screen.findByText('Hello.')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Delete README.md' }))
    const dialog = await screen.findByRole('dialog', { name: 'Delete workspace path' })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Delete' }))

    await waitFor(() => expect(deletePath).toHaveBeenCalledWith('README.md', false))
    expect(await screen.findByText('No file selected.')).toBeTruthy()
  })
})
