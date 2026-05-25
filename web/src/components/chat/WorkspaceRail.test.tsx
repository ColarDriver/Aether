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

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('WorkspaceRail', () => {
  it('browses directories, selects files, and searches paths', async () => {
    vi.spyOn(api, 'workspaceTree').mockImplementation(async (path = '') => (path === 'aether' ? nestedTree : rootTree))
    const search = vi.spyOn(api, 'workspaceSearch').mockResolvedValue({ root: '/workspace/Aether', query: 'app', entries: nestedTree.entries })
    const onSelectFile = vi.fn()

    render(<WorkspaceRail selectedFilePath="README.md" onSelectFile={onSelectFile} />)

    expect(await screen.findByTitle('README.md')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'root' })).toBeTruthy()
    expect(screen.queryByText('Select a file')).toBeNull()
    expect(screen.getByText(/2 items.*1 dir.*1 file/)).toBeTruthy()

    fireEvent.click(screen.getByTitle('README.md'))
    expect(onSelectFile).toHaveBeenCalledWith('README.md')

    fireEvent.click(screen.getByTitle('aether'))
    expect(await screen.findByTitle('aether/app.py')).toBeTruthy()

    fireEvent.change(screen.getByPlaceholderText('Search files'), { target: { value: 'app' } })
    fireEvent.click(screen.getByRole('button', { name: 'Search' }))

    await waitFor(() => expect(search).toHaveBeenCalledWith('app', 80))
    expect(screen.getByText('Search "app"')).toBeTruthy()
    fireEvent.click(screen.getByTitle('aether/app.py'))
    expect(onSelectFile).toHaveBeenCalledWith('aether/app.py')
  })
})
