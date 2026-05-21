// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { WorkspaceRail } from './WorkspaceRail'

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

describe('WorkspaceRail', () => {
  it('browses directories, previews files, and searches paths', async () => {
    vi.spyOn(api, 'workspaceTree').mockImplementation(async (path = '') => (path === 'aether' ? nestedTree : rootTree))
    vi.spyOn(api, 'workspaceFile').mockImplementation(async (path: string) => (path === 'aether/app.py' ? appFile : readme))
    const search = vi.spyOn(api, 'workspaceSearch').mockResolvedValue({ root: '/workspace/Aether', query: 'app', entries: nestedTree.entries })

    render(<WorkspaceRail />)

    expect(await screen.findByTitle('README.md')).toBeTruthy()

    fireEvent.click(screen.getByTitle('README.md'))
    expect(await screen.findByText('Hello.')).toBeTruthy()

    fireEvent.click(screen.getByTitle('aether'))
    expect(await screen.findByTitle('aether/app.py')).toBeTruthy()

    fireEvent.change(screen.getByPlaceholderText('Search files'), { target: { value: 'app' } })
    fireEvent.click(screen.getByRole('button', { name: 'Search' }))

    await waitFor(() => expect(search).toHaveBeenCalledWith('app', 80))
    fireEvent.click(screen.getByTitle('aether/app.py'))
    expect(await screen.findByText(/print/)).toBeTruthy()
  })
})
