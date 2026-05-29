// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
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

beforeEach(() => {
  vi.spyOn(api, 'workspaceRoot').mockResolvedValue({
    root: '/workspace/Aether',
    name: 'Aether',
    exists: true,
    readable: true,
    git_root: '/workspace/Aether',
    is_git: true,
    recent_roots: ['/workspace/Aether', '/workspace/Other'],
  })
  vi.spyOn(api, 'workspaceGitStatus').mockResolvedValue({
    root: '/workspace/Aether',
    git_root: '/workspace/Aether',
    available: true,
    branch: 'feature/web',
    upstream: 'origin/feature/web',
    ahead: 0,
    behind: 0,
    clean: true,
    files: [],
  })
  vi.spyOn(api, 'workspaceCheckpoints').mockResolvedValue({
    root: '/workspace/Aether',
    checkpoints: [],
  })
})

describe('WorkspaceRail', () => {
  it('browses directories, selects files, and searches paths', async () => {
    vi.spyOn(api, 'workspaceTree').mockImplementation(async (path = '') => (path === 'aether' ? nestedTree : rootTree))
    const search = vi.spyOn(api, 'workspaceSearch').mockResolvedValue({ root: '/workspace/Aether', query: 'app', entries: nestedTree.entries })
    const onSelectFile = vi.fn()

    render(<WorkspaceRail selectedFilePath="README.md" onSelectFile={onSelectFile} />)

    expect(await screen.findByTitle('README.md')).toBeTruthy()
    expect(screen.getByLabelText('Workspace root').textContent).toContain('Aether')
    expect(screen.getByRole('tab', { name: 'Files' }).getAttribute('aria-selected')).toBe('true')
    expect(screen.getByRole('tab', { name: 'Source Control' }).getAttribute('aria-selected')).toBe('false')
    expect(screen.getByLabelText('Workspace file tree')).toBeTruthy()
    expect(screen.queryByText('Select a file')).toBeNull()
    expect(screen.queryByText('18 B')).toBeNull()
    expect(screen.queryByText(/\bdir\b/i)).toBeNull()
    expect(document.querySelector('.workspace-rail-entry-kind-directory')).toBeTruthy()
    expect(document.querySelector('.workspace-rail-entry-kind-file')).toBeTruthy()

    fireEvent.click(screen.getByTitle('README.md'))
    expect(onSelectFile).toHaveBeenCalledWith('README.md')

    fireEvent.click(screen.getByTitle('aether'))
    expect(await screen.findByTitle('aether/app.py')).toBeTruthy()

    fireEvent.change(screen.getByPlaceholderText('Search files...'), { target: { value: 'app' } })
    fireEvent.click(screen.getByRole('button', { name: 'Search' }))

    await waitFor(() => expect(search).toHaveBeenCalledWith('app', 80))
    expect(screen.getByText('Search "app"')).toBeTruthy()
    fireEvent.click(screen.getByTitle('aether/app.py'))
    expect(onSelectFile).toHaveBeenCalledWith('aether/app.py')
  })

  it('switches workspace roots and reloads workspace state', async () => {
    vi.spyOn(api, 'workspaceTree').mockResolvedValue(rootTree)
    const switchRoot = vi.spyOn(api, 'switchWorkspaceRoot').mockResolvedValue({
      root: '/workspace/Other',
      name: 'Other',
      exists: true,
      readable: true,
      git_root: null,
      is_git: false,
      recent_roots: ['/workspace/Other', '/workspace/Aether'],
    })
    const onWorkspaceRootChanged = vi.fn()

    render(<WorkspaceRail sessionId="ses-root" onWorkspaceRootChanged={onWorkspaceRootChanged} />)

    expect(await screen.findByTitle('README.md')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Switch workspace root' }))
    const dialog = await screen.findByRole('dialog', { name: 'Switch workspace root' })
    fireEvent.change(within(dialog).getByLabelText('Root path'), { target: { value: '/workspace/Other' } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Switch' }))

    await waitFor(() => expect(switchRoot).toHaveBeenCalledWith({
      path: '/workspace/Other',
      session_id: 'ses-root',
      remember: true,
    }))
    expect(onWorkspaceRootChanged).toHaveBeenCalledWith(expect.objectContaining({ root: '/workspace/Other' }))
    await waitFor(() => expect(screen.getByLabelText('Workspace root').textContent).toContain('Other'))
  })

  it('creates, renames, and deletes workspace paths', async () => {
    vi.spyOn(api, 'workspaceTree').mockResolvedValue(rootTree)
    const createFile = vi.spyOn(api, 'workspaceCreateFile').mockResolvedValue({
      root: '/workspace/Aether',
      path: 'notes.md',
      name: 'notes.md',
      content: '',
      size_bytes: 0,
      updated_at: 4,
      language: 'markdown',
      truncated: false,
      binary: false,
    })
    const createDirectory = vi.spyOn(api, 'workspaceCreateDirectory').mockResolvedValue({
      path: 'docs',
      name: 'docs',
      kind: 'directory',
      updated_at: 5,
    })
    const renamePath = vi.spyOn(api, 'workspaceRenamePath').mockResolvedValue({
      path: 'README-next.md',
      name: 'README-next.md',
      kind: 'file',
      size_bytes: 18,
      updated_at: 6,
    })
    const deletePath = vi.spyOn(api, 'workspaceDeletePath').mockResolvedValue(undefined)
    const onSelectFile = vi.fn()
    const onDeletedPath = vi.fn()
    const onRenamedPath = vi.fn()

    render(<WorkspaceRail selectedFilePath="README.md" onSelectFile={onSelectFile} onDeletedPath={onDeletedPath} onRenamedPath={onRenamedPath} />)

    expect(await screen.findByTitle('README.md')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'New workspace file' }))
    const newFileDialog = await screen.findByRole('dialog', { name: 'New file' })
    fireEvent.change(within(newFileDialog).getByLabelText('File name'), { target: { value: 'notes.md' } })
    fireEvent.click(within(newFileDialog).getByRole('button', { name: 'Apply' }))

    await waitFor(() => expect(createFile).toHaveBeenCalledWith('notes.md', ''))
    expect(onSelectFile).toHaveBeenCalledWith('notes.md')

    fireEvent.click(screen.getByRole('button', { name: 'New workspace folder' }))
    const newFolderDialog = await screen.findByRole('dialog', { name: 'New folder' })
    fireEvent.change(within(newFolderDialog).getByLabelText('Folder name'), { target: { value: 'docs' } })
    fireEvent.click(within(newFolderDialog).getByRole('button', { name: 'Apply' }))

    await waitFor(() => expect(createDirectory).toHaveBeenCalledWith('docs'))

    fireEvent.click(screen.getByRole('button', { name: 'Rename README.md' }))
    const renameDialog = await screen.findByRole('dialog', { name: 'Rename path' })
    const input = within(renameDialog).getByLabelText('New path')
    fireEvent.change(input, { target: { value: 'README-next.md' } })
    fireEvent.click(within(renameDialog).getByRole('button', { name: 'Apply' }))

    await waitFor(() => expect(renamePath).toHaveBeenCalledWith('README.md', 'README-next.md'))
    expect(onRenamedPath).toHaveBeenCalledWith('README.md', 'README-next.md', 'file')

    fireEvent.click(screen.getByRole('button', { name: 'Delete README.md' }))
    const deleteDialog = await screen.findByRole('dialog', { name: 'Delete workspace path' })
    fireEvent.click(within(deleteDialog).getByRole('button', { name: 'Delete' }))

    await waitFor(() => expect(deletePath).toHaveBeenCalledWith('README.md', false))
    expect(onDeletedPath).toHaveBeenCalledWith('README.md')
  })

  it('shows git changes, previews diffs, creates checkpoints, and restores files or checkpoints', async () => {
    vi.spyOn(api, 'workspaceTree').mockResolvedValue(rootTree)
    const dirtyStatus = {
      root: '/workspace/Aether',
      git_root: '/workspace/Aether',
      available: true,
      branch: 'feature/web',
      upstream: 'origin/feature/web',
      ahead: 1,
      behind: 0,
      clean: false,
      files: [
        {
          path: 'src/app.py',
          status: 'modified',
          index_status: ' ',
          worktree_status: 'M',
          staged: false,
          unstaged: true,
          untracked: false,
        },
      ],
    }
    const cleanStatus = {
      ...dirtyStatus,
      clean: true,
      files: [],
    }
    const existingCheckpoint = {
      checkpoint_id: 'cp-1',
      label: 'Before auth edit',
      created_at: 1_800_000_000,
      root: '/workspace/Aether',
      files: [
        { path: 'src/app.py', exists: true, size_bytes: 12, binary: false },
      ],
    }
    const createdCheckpoint = {
      checkpoint_id: 'cp-2',
      label: 'Manual workspace checkpoint',
      created_at: 1_800_000_100,
      root: '/workspace/Aether',
      files: [
        { path: 'src/app.py', exists: true, size_bytes: 12, binary: false },
      ],
    }
    vi.spyOn(api, 'workspaceGitStatus').mockResolvedValue(dirtyStatus)
    vi.spyOn(api, 'workspaceCheckpoints').mockResolvedValue({
      root: '/workspace/Aether',
      checkpoints: [existingCheckpoint],
    })
    const gitDiff = vi.spyOn(api, 'workspaceGitDiff').mockResolvedValue({
      root: '/workspace/Aether',
      path: 'src/app.py',
      staged: false,
      truncated: false,
      diff: '--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n',
    })
    const createCheckpoint = vi.spyOn(api, 'createWorkspaceCheckpoint').mockResolvedValue(createdCheckpoint)
    const restoreFile = vi.spyOn(api, 'workspaceGitRestore').mockResolvedValue(cleanStatus)
    const restoreCheckpoint = vi.spyOn(api, 'restoreWorkspaceCheckpoint').mockResolvedValue(existingCheckpoint)

    render(<WorkspaceRail />)

    fireEvent.click(await screen.findByRole('tab', { name: /Source Control/ }))
    const repo = await screen.findByLabelText('Repository status')
    expect(within(repo).getByText('feature/web')).toBeTruthy()
    expect(within(repo).getByText('1 changed')).toBeTruthy()
    expect(within(repo).getByText('Before auth edit')).toBeTruthy()
    expect(within(repo).getByText('src/app.py')).toBeTruthy()
    const changedFile = within(repo).getByText('src/app.py').closest('.workspace-git-file')
    expect(changedFile).toBeTruthy()

    fireEvent.click(within(repo).getByRole('button', { name: 'Diff' }))
    await waitFor(() => expect(gitDiff).toHaveBeenCalledWith('src/app.py'))
    expect(await screen.findByLabelText('Workspace git diff')).toBeTruthy()
    expect(within(repo).getAllByText('src/app.py').length).toBeGreaterThan(0)

    fireEvent.click(within(repo).getByRole('button', { name: 'Checkpoint' }))
    await waitFor(() => expect(createCheckpoint).toHaveBeenCalledWith({ label: 'Manual workspace checkpoint' }))
    expect(await screen.findByText('Checkpoint cp-2 captured 1 file.')).toBeTruthy()

    fireEvent.click(within(changedFile as HTMLElement).getByRole('button', { name: 'Restore' }))
    const restoreFileDialog = await screen.findByRole('dialog', { name: 'Restore git file' })
    fireEvent.click(within(restoreFileDialog).getByRole('button', { name: 'Restore' }))
    await waitFor(() => expect(restoreFile).toHaveBeenCalledWith('src/app.py'))

    const checkpointRow = within(repo).getByText('Before auth edit').closest('.workspace-git-checkpoint')
    expect(checkpointRow).toBeTruthy()
    fireEvent.click(within(checkpointRow as HTMLElement).getByRole('button', { name: 'Restore' }))
    const restoreCheckpointDialog = await screen.findByRole('dialog', { name: 'Restore checkpoint' })
    expect(within(restoreCheckpointDialog).getByText(/Before auth edit/)).toBeTruthy()
    fireEvent.click(within(restoreCheckpointDialog).getByRole('button', { name: 'Restore' }))
    await waitFor(() => expect(restoreCheckpoint).toHaveBeenCalledWith('cp-1'))
    expect(await screen.findByText('Restored checkpoint cp-1.')).toBeTruthy()
  })
})
