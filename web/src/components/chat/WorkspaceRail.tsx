import { ChevronLeft, ChevronRight, ExternalLink, File, FilePlus, Folder, FolderOpen, FolderPlus, FolderTree, GitBranch, GitCompare, History, Pencil, RefreshCw, RotateCcw, Search, Trash2, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { WorkspaceCheckpoint, WorkspaceEntry, WorkspaceGitDiff, WorkspaceGitFile, WorkspaceGitStatus, WorkspaceRootInfo, WorkspaceTree } from '../../api/types'
import { Button } from '../shared/Button'
import { ConfirmDialog } from '../shared/ConfirmDialog'
import { Spinner } from '../shared/Spinner'
import { DiffViewer } from './DiffViewer'

type Props = {
  side?: 'left' | 'right'
  sessionId?: string | null
  selectedFilePath?: string | null
  onSelectFile?: (path: string) => void
  onClose?: () => void
  onOpenWorkspace?: () => void
  onDeletedPath?: (path: string) => void
  onRenamedPath?: (path: string, newPath: string, kind: WorkspaceEntry['kind']) => void
  onTreeLoaded?: (tree: WorkspaceTree) => void
  onWorkspaceRootChanged?: (info: WorkspaceRootInfo) => void
}

type WorkspaceActionDialog =
  | { kind: 'new-file'; title: string; description: string; label: string; initialValue: string }
  | { kind: 'new-folder'; title: string; description: string; label: string; initialValue: string }
  | { kind: 'rename'; title: string; description: string; label: string; initialValue: string; entry: WorkspaceEntry }

export function WorkspaceRail({
  side = 'right',
  sessionId = null,
  selectedFilePath = null,
  onSelectFile,
  onClose,
  onOpenWorkspace,
  onDeletedPath,
  onRenamedPath,
  onTreeLoaded,
  onWorkspaceRootChanged,
}: Props) {
  const [rootInfo, setRootInfo] = useState<WorkspaceRootInfo | null>(null)
  const [rootLoading, setRootLoading] = useState(false)
  const [rootError, setRootError] = useState<string | null>(null)
  const [rootDialogOpen, setRootDialogOpen] = useState(false)
  const [tree, setTree] = useState<WorkspaceTree | null>(null)
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<WorkspaceEntry[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [mutating, setMutating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [actionDialog, setActionDialog] = useState<WorkspaceActionDialog | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<WorkspaceEntry | null>(null)
  const [gitStatus, setGitStatus] = useState<WorkspaceGitStatus | null>(null)
  const [gitLoading, setGitLoading] = useState(false)
  const [gitError, setGitError] = useState<string | null>(null)
  const [activeDiff, setActiveDiff] = useState<WorkspaceGitDiff | null>(null)
  const [diffLoadingPath, setDiffLoadingPath] = useState<string | null>(null)
  const [restoreTarget, setRestoreTarget] = useState<WorkspaceGitFile | null>(null)
  const [checkpointRestoreTarget, setCheckpointRestoreTarget] = useState<WorkspaceCheckpoint | null>(null)
  const [checkpoints, setCheckpoints] = useState<WorkspaceCheckpoint[]>([])
  const [checkpointsLoading, setCheckpointsLoading] = useState(false)
  const [checkpointMessage, setCheckpointMessage] = useState<string | null>(null)
  const onTreeLoadedRef = useRef(onTreeLoaded)
  const onWorkspaceRootChangedRef = useRef(onWorkspaceRootChanged)

  const visibleEntries = searchResults ?? tree?.entries ?? []
  const title = tree?.path ? tree.path : 'Project root'
  const browserTitle = searchResults ? 'Search results' : title
  const currentRoot = rootInfo?.root ?? tree?.root ?? ''
  const rootLabel = useMemo(() => shortenPath(currentRoot), [currentRoot])
  const breadcrumbs = useMemo(() => buildBreadcrumbs(tree?.path ?? ''), [tree?.path])
  const currentPath = tree?.path ?? ''

  useEffect(() => {
    onTreeLoadedRef.current = onTreeLoaded
  }, [onTreeLoaded])

  useEffect(() => {
    onWorkspaceRootChangedRef.current = onWorkspaceRootChanged
  }, [onWorkspaceRootChanged])

  const loadRoot = useCallback(() => {
    setRootLoading(true)
    setRootError(null)
    api.workspaceRoot()
      .then((info) => setRootInfo(info))
      .catch((err: unknown) => setRootError(err instanceof Error ? err.message : String(err)))
      .finally(() => setRootLoading(false))
  }, [])

  const loadTree = useCallback((path = '') => {
    setLoading(true)
    setError(null)
    setSearchResults(null)
    api.workspaceTree(path)
      .then((nextTree) => {
        setTree(nextTree)
        onTreeLoadedRef.current?.(nextTree)
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }, [])

  const loadGitStatus = useCallback(() => {
    setGitLoading(true)
    setGitError(null)
    api.workspaceGitStatus()
      .then((status) => setGitStatus(status))
      .catch((err: unknown) => {
        setGitStatus(null)
        setGitError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => setGitLoading(false))
  }, [])

  const loadCheckpoints = useCallback(() => {
    setCheckpointsLoading(true)
    api.workspaceCheckpoints()
      .then((result) => setCheckpoints(result.checkpoints ?? []))
      .catch((err: unknown) => setGitError(err instanceof Error ? err.message : String(err)))
      .finally(() => setCheckpointsLoading(false))
  }, [])

  useEffect(() => {
    loadRoot()
    loadTree('')
    loadGitStatus()
    loadCheckpoints()
  }, [loadCheckpoints, loadGitStatus, loadRoot, loadTree])

  const reloadCurrentTree = useCallback(() => {
    loadRoot()
    loadTree(currentPath)
    loadGitStatus()
    loadCheckpoints()
  }, [currentPath, loadCheckpoints, loadGitStatus, loadRoot, loadTree])

  const switchWorkspaceRoot = useCallback(async (path: string) => {
    const normalized = path.trim()
    if (!normalized || mutating) return
    setMutating(true)
    setRootError(null)
    setError(null)
    try {
      const info = await api.switchWorkspaceRoot({
        path: normalized,
        ...(sessionId ? { session_id: sessionId } : {}),
        remember: true,
      })
      setRootInfo(info)
      setRootDialogOpen(false)
      setSearchResults(null)
      setQuery('')
      setActiveDiff(null)
      setCheckpointMessage(null)
      onWorkspaceRootChangedRef.current?.(info)
      loadTree('')
      loadGitStatus()
      loadCheckpoints()
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setRootError(message)
      setError(message)
    } finally {
      setMutating(false)
    }
  }, [loadCheckpoints, loadGitStatus, loadTree, mutating, sessionId])

  const runSearch = () => {
    const value = query.trim()
    if (!value) {
      setSearchResults(null)
      return
    }
    setLoading(true)
    setError(null)
    api.workspaceSearch(value, 80)
      .then((result) => setSearchResults(result.entries))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }

  const openEntry = (entry: WorkspaceEntry) => {
    if (entry.kind === 'directory') {
      loadTree(entry.path)
      return
    }
    onSelectFile?.(entry.path)
  }

  const openNewFileDialog = () => {
    setActionDialog({
      kind: 'new-file',
      title: 'New file',
      description: 'Create a text file in the current workspace directory.',
      label: 'File name',
      initialValue: '',
    })
  }

  const openNewFolderDialog = () => {
    setActionDialog({
      kind: 'new-folder',
      title: 'New folder',
      description: 'Create a directory in the current workspace directory.',
      label: 'Folder name',
      initialValue: '',
    })
  }

  const openRenameDialog = (entry: WorkspaceEntry) => {
    setActionDialog({
      kind: 'rename',
      title: 'Rename path',
      description: 'Move or rename this workspace path inside the project root.',
      label: 'New path',
      initialValue: entry.path,
      entry,
    })
  }

  const submitActionDialog = async (value: string) => {
    if (!actionDialog || mutating) return
    const normalized = normalizeInputPath(value)
    if (!normalized) {
      setError('Path is required.')
      return
    }
    setMutating(true)
    setError(null)
    try {
      if (actionDialog.kind === 'new-file') {
        const created = await api.workspaceCreateFile(joinWorkspacePath(currentPath, normalized), '')
        setActionDialog(null)
        loadTree(parentPath(created.path))
        onSelectFile?.(created.path)
      } else if (actionDialog.kind === 'new-folder') {
        const created = await api.workspaceCreateDirectory(joinWorkspacePath(currentPath, normalized))
        setActionDialog(null)
        loadTree(parentPath(created.path))
      } else {
        const renamed = await api.workspaceRenamePath(actionDialog.entry.path, normalized)
        setActionDialog(null)
        loadTree(parentPath(renamed.path))
        onRenamedPath?.(actionDialog.entry.path, renamed.path, actionDialog.entry.kind)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setMutating(false)
    }
  }

  const confirmDelete = async () => {
    if (!deleteTarget || mutating) return
    const target = deleteTarget
    setDeleteTarget(null)
    setMutating(true)
    setError(null)
    try {
      await api.workspaceDeletePath(target.path, target.kind === 'directory')
      reloadCurrentTree()
      onDeletedPath?.(target.path)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setMutating(false)
    }
  }

  const openGitDiff = (file: WorkspaceGitFile) => {
    setDiffLoadingPath(file.path)
    setGitError(null)
    api.workspaceGitDiff(file.path)
      .then((diff) => setActiveDiff(diff))
      .catch((err: unknown) => setGitError(err instanceof Error ? err.message : String(err)))
      .finally(() => setDiffLoadingPath(null))
  }

  const createCheckpoint = async () => {
    if (mutating || gitLoading) return
    setMutating(true)
    setCheckpointMessage(null)
    setGitError(null)
    try {
      const checkpoint = await api.createWorkspaceCheckpoint({ label: 'Manual workspace checkpoint' })
      setCheckpointMessage('Checkpoint ' + checkpoint.checkpoint_id + ' captured ' + checkpoint.files.length.toLocaleString() + ' file' + (checkpoint.files.length === 1 ? '' : 's') + '.')
      setCheckpoints((current) => [checkpoint, ...current.filter((item) => item.checkpoint_id !== checkpoint.checkpoint_id)])
      loadGitStatus()
    } catch (err) {
      setGitError(err instanceof Error ? err.message : String(err))
    } finally {
      setMutating(false)
    }
  }

  const confirmCheckpointRestore = async () => {
    if (!checkpointRestoreTarget || mutating) return
    const target = checkpointRestoreTarget
    setCheckpointRestoreTarget(null)
    setMutating(true)
    setGitError(null)
    try {
      const checkpoint = await api.restoreWorkspaceCheckpoint(target.checkpoint_id)
      setCheckpointMessage('Restored checkpoint ' + checkpoint.checkpoint_id + '.')
      setActiveDiff(null)
      reloadCurrentTree()
    } catch (err) {
      setGitError(err instanceof Error ? err.message : String(err))
    } finally {
      setMutating(false)
    }
  }

  const confirmGitRestore = async () => {
    if (!restoreTarget || mutating) return
    const target = restoreTarget
    setRestoreTarget(null)
    setMutating(true)
    setGitError(null)
    try {
      const status = await api.workspaceGitRestore(target.path)
      setGitStatus(status)
      if (activeDiff?.path === target.path) setActiveDiff(null)
      reloadCurrentTree()
    } catch (err) {
      setGitError(err instanceof Error ? err.message : String(err))
    } finally {
      setMutating(false)
    }
  }

  return (
    <>
      <aside className={'workspace-rail workspace-rail-side-' + side} aria-label="Workspace files">
        <header className="workspace-rail-header">
          <div className="workspace-rail-title">
            <div>
              <strong>Workspace</strong>
              <span title={currentRoot}>{rootLoading ? 'Loading root' : rootLabel || 'No root'}</span>
            </div>
          </div>
          <div className="workspace-rail-actions">
            <Button title="Switch root" aria-label="Switch workspace root" onClick={() => setRootDialogOpen(true)} disabled={rootLoading || mutating}>
              <FolderOpen size={15} />
            </Button>
            <Button title="New file" aria-label="New workspace file" onClick={openNewFileDialog} disabled={loading || mutating}>
              <FilePlus size={15} />
            </Button>
            <Button title="New folder" aria-label="New workspace folder" onClick={openNewFolderDialog} disabled={loading || mutating}>
              <FolderPlus size={15} />
            </Button>
            <Button title="Refresh workspace" aria-label="Refresh workspace" onClick={() => loadTree(tree?.path ?? '')} disabled={loading || mutating}>
              <RefreshCw size={15} />
            </Button>
            {onOpenWorkspace ? (
              <Button title="Open workspace page" aria-label="Open workspace page" onClick={onOpenWorkspace}>
                <ExternalLink size={15} />
              </Button>
            ) : null}
            {onClose ? (
              <Button title="Close workspace panel" aria-label="Close workspace panel" onClick={onClose}>
                <X size={15} />
              </Button>
            ) : null}
          </div>
        </header>

        <WorkspaceRootStrip
          info={rootInfo}
          error={rootError}
          loading={rootLoading}
          mutating={mutating}
          onSwitch={switchWorkspaceRoot}
        />

        <nav className="workspace-rail-breadcrumb" aria-label="Workspace path">
          {searchResults ? (
            <span className="workspace-rail-search-crumb">
              <Search size={13} />
              Search &quot;{query.trim()}&quot;
            </span>
          ) : (
            <>
              <button type="button" onClick={() => loadTree('')} title={tree?.root ?? 'Workspace root'}>
                root
              </button>
              {breadcrumbs.map((crumb) => (
                <span key={crumb.path} className="workspace-rail-crumb-segment">
                  <ChevronRight size={12} />
                  <button type="button" onClick={() => loadTree(crumb.path)} title={crumb.path}>
                    {crumb.name}
                  </button>
                </span>
              ))}
            </>
          )}
        </nav>

        <WorkspaceGitPanel
          status={gitStatus}
          loading={gitLoading}
          error={gitError}
          activeDiff={activeDiff}
          diffLoadingPath={diffLoadingPath}
          checkpoints={checkpoints}
          checkpointsLoading={checkpointsLoading}
          checkpointMessage={checkpointMessage}
          mutating={mutating}
          onCheckpoint={createCheckpoint}
          onOpenDiff={openGitDiff}
          onRefresh={loadGitStatus}
          onRestore={setRestoreTarget}
          onRestoreCheckpoint={setCheckpointRestoreTarget}
        />

        <div className="workspace-rail-browser" aria-label="Workspace browser">
          <div className="workspace-rail-path">
            <div>
              <strong>{browserTitle}</strong>
              <span>{visibleEntries.length} item{visibleEntries.length === 1 ? '' : 's'}</span>
            </div>
            {tree?.parent_path !== null && tree?.parent_path !== undefined ? (
              <button type="button" onClick={() => loadTree(tree.parent_path ?? '')}>
                <ChevronLeft size={14} /> Parent
              </button>
            ) : null}
          </div>
          <form className="workspace-rail-search" onSubmit={(event) => { event.preventDefault(); runSearch() }}>
            <Search size={14} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search files" />
            {searchResults ? (
              <button type="button" aria-label="Clear workspace search" onClick={() => { setSearchResults(null); setQuery('') }}>
                <X size={14} />
              </button>
            ) : (
              <button type="submit">Search</button>
            )}
          </form>
          {error ? <div className="workspace-rail-error">{error}</div> : null}
          <div className="workspace-rail-list">
            {loading && visibleEntries.length === 0 ? <Spinner label="Loading workspace" /> : null}
            {visibleEntries.length === 0 && !loading ? <div className="empty-chat">No files found.</div> : null}
            {visibleEntries.map((entry) => (
              <div
                key={entry.path || '__root__'}
                className={'workspace-rail-entry workspace-rail-entry-kind-' + entry.kind + (selectedFilePath === entry.path ? ' active' : '')}
              >
                <button
                  type="button"
                  className="workspace-rail-entry-main"
                  onClick={() => openEntry(entry)}
                  title={entry.path || '.'}
                >
                  {entry.kind === 'directory' ? <Folder size={15} /> : <File size={15} />}
                  <span>{entry.name}</span>
                  {searchResults ? <small>{entry.path || '.'}</small> : null}
                </button>
                <div className="workspace-rail-entry-actions" aria-label={'Actions for ' + entry.name}>
                  <button type="button" aria-label={'Rename ' + entry.name} title="Rename" onClick={() => openRenameDialog(entry)} disabled={mutating}>
                    <Pencil size={13} />
                  </button>
                  <button type="button" aria-label={'Delete ' + entry.name} title="Delete" onClick={() => setDeleteTarget(entry)} disabled={mutating}>
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </aside>
      {actionDialog ? (
        <WorkspaceNameDialog
          title={actionDialog.title}
          description={actionDialog.description}
          label={actionDialog.label}
          initialValue={actionDialog.initialValue}
          submitting={mutating}
          onCancel={() => setActionDialog(null)}
          onSubmit={submitActionDialog}
        />
      ) : null}
      {rootDialogOpen ? (
        <WorkspaceRootDialog
          currentRoot={currentRoot}
          recentRoots={rootInfo?.recent_roots ?? []}
          submitting={mutating}
          error={rootError}
          onCancel={() => setRootDialogOpen(false)}
          onSubmit={switchWorkspaceRoot}
        />
      ) : null}
      {deleteTarget ? (
        <ConfirmDialog
          title="Delete workspace path"
          description={'Delete "' + deleteTarget.path + '"? ' + (deleteTarget.kind === 'directory' ? 'This removes the directory and its contents.' : 'This removes the file from disk.')}
          confirmLabel="Delete"
          cancelLabel="Cancel"
          onCancel={() => setDeleteTarget(null)}
          onConfirm={confirmDelete}
        />
      ) : null}
      {restoreTarget ? (
        <ConfirmDialog
          title="Restore git file"
          description={'Restore "' + restoreTarget.path + '" from git? Untracked files are removed; tracked files are reset to HEAD.'}
          confirmLabel="Restore"
          cancelLabel="Cancel"
          onCancel={() => setRestoreTarget(null)}
          onConfirm={confirmGitRestore}
        />
      ) : null}
      {checkpointRestoreTarget ? (
        <ConfirmDialog
          title="Restore checkpoint"
          description={'Restore checkpoint "' + checkpointLabel(checkpointRestoreTarget) + '"? This writes captured file contents back to disk and removes files that did not exist when the checkpoint was created.'}
          confirmLabel="Restore"
          cancelLabel="Cancel"
          onCancel={() => setCheckpointRestoreTarget(null)}
          onConfirm={confirmCheckpointRestore}
        />
      ) : null}
    </>
  )
}

function WorkspaceRootStrip({
  info,
  error,
  loading,
  mutating,
  onSwitch,
}: {
  info: WorkspaceRootInfo | null
  error: string | null
  loading: boolean
  mutating: boolean
  onSwitch: (path: string) => void
}) {
  if (error) {
    return <div className="workspace-root-strip workspace-root-strip-error">{error}</div>
  }
  if (!info) {
    return loading ? <div className="workspace-root-strip">Loading workspace root</div> : null
  }
  const recent = info.recent_roots.filter((root) => root !== info.root).slice(0, 3)
  return (
    <div className="workspace-root-strip" aria-label="Workspace root">
      <span title={info.root}>
        <FolderOpen size={13} aria-hidden="true" />
        {info.name || shortenPath(info.root)}
      </span>
      <em>{info.is_git ? 'git' : 'no git'}</em>
      {recent.map((root) => (
        <button key={root} type="button" title={root} onClick={() => onSwitch(root)} disabled={mutating || loading}>
          {shortenPath(root)}
        </button>
      ))}
    </div>
  )
}

function WorkspaceRootDialog({
  currentRoot,
  recentRoots,
  submitting,
  error,
  onCancel,
  onSubmit,
}: {
  currentRoot: string
  recentRoots: string[]
  submitting: boolean
  error: string | null
  onCancel: () => void
  onSubmit: (path: string) => void
}) {
  const [value, setValue] = useState(currentRoot)
  const canSubmit = value.trim().length > 0 && !submitting
  return (
    <div className="modal-backdrop" role="presentation">
      <form
        className="prompt-modal workspace-action-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Switch workspace root"
        onSubmit={(event) => {
          event.preventDefault()
          if (canSubmit) onSubmit(value)
        }}
      >
        <header>
          <span className="prompt-modal-icon" aria-hidden="true"><FolderOpen size={17} /></span>
          <div className="prompt-modal-title">
            <strong>Switch workspace root</strong>
            <span>Choose a server-local project directory for browsing and future runs.</span>
          </div>
        </header>
        <div className="prompt-body workspace-action-modal-body">
          <label>
            <span>Root path</span>
            <input autoFocus value={value} onChange={(event) => setValue(event.target.value)} />
          </label>
          {error ? <div className="workspace-rail-error">{error}</div> : null}
          {recentRoots.length > 0 ? (
            <div className="workspace-root-recent" aria-label="Recent workspace roots">
              {recentRoots.map((root) => (
                <button key={root} type="button" title={root} onClick={() => setValue(root)}>
                  {shortenPath(root)}
                </button>
              ))}
            </div>
          ) : null}
        </div>
        <footer>
          <button type="button" className="prompt-action" onClick={onCancel} disabled={submitting}>Cancel</button>
          <button type="submit" className="prompt-action prompt-action-primary" disabled={!canSubmit}>{submitting ? 'Switching' : 'Switch'}</button>
        </footer>
      </form>
    </div>
  )
}

type WorkspaceNameDialogProps = {
  title: string
  description: string
  label: string
  initialValue: string
  submitting: boolean
  onCancel: () => void
  onSubmit: (value: string) => void
}

function WorkspaceGitPanel({
  status,
  loading,
  error,
  activeDiff,
  diffLoadingPath,
  checkpoints,
  checkpointsLoading,
  checkpointMessage,
  mutating,
  onCheckpoint,
  onOpenDiff,
  onRefresh,
  onRestore,
  onRestoreCheckpoint,
}: {
  status: WorkspaceGitStatus | null
  loading: boolean
  error: string | null
  activeDiff: WorkspaceGitDiff | null
  diffLoadingPath: string | null
  checkpoints: WorkspaceCheckpoint[]
  checkpointsLoading: boolean
  checkpointMessage: string | null
  mutating: boolean
  onCheckpoint: () => void
  onOpenDiff: (file: WorkspaceGitFile) => void
  onRefresh: () => void
  onRestore: (file: WorkspaceGitFile) => void
  onRestoreCheckpoint: (checkpoint: WorkspaceCheckpoint) => void
}) {
  if (loading && !status) {
    return (
      <section className="workspace-git-panel" aria-label="Repository status">
        <span className="workspace-git-loading"><RefreshCw size={13} /> Loading repository status</span>
      </section>
    )
  }
  if (!status && !error) return null
  if (error) {
    return (
      <section className="workspace-git-panel workspace-git-panel-error" aria-label="Repository status">
        <span>{error}</span>
        <button type="button" onClick={onRefresh}>Retry</button>
      </section>
    )
  }
  if (!status?.available) {
    return (
      <section className="workspace-git-panel workspace-git-panel-muted" aria-label="Repository status">
        <span><GitBranch size={13} /> {status?.message || 'No git repository detected.'}</span>
      </section>
    )
  }

  const changedFiles = status.files.slice(0, 8)
  const recentCheckpoints = checkpoints.slice(0, 4)
  return (
    <section className="workspace-git-panel" aria-label="Repository status">
      <header className="workspace-git-header">
        <span>
          <GitBranch size={14} aria-hidden="true" />
          <strong>{status.branch || 'detached'}</strong>
          {status.upstream ? <small>{status.upstream}</small> : null}
        </span>
        <em className={status.clean ? 'workspace-git-clean' : 'workspace-git-dirty'}>
          {status.clean ? 'clean' : status.files.length.toLocaleString() + ' changed'}
        </em>
      </header>
      {(status.ahead > 0 || status.behind > 0) ? (
        <div className="workspace-git-sync">
          {status.ahead > 0 ? <span>{status.ahead} ahead</span> : null}
          {status.behind > 0 ? <span>{status.behind} behind</span> : null}
        </div>
      ) : null}
      <div className="workspace-git-actions">
        <button type="button" onClick={onRefresh} disabled={loading || mutating}>
          <RefreshCw size={13} />
          Refresh
        </button>
        <button type="button" onClick={onCheckpoint} disabled={mutating || status.files.length === 0}>
          <History size={13} />
          Checkpoint
        </button>
      </div>
      {checkpointMessage ? <div className="workspace-git-message">{checkpointMessage}</div> : null}
      {checkpointsLoading && recentCheckpoints.length === 0 ? (
        <span className="workspace-git-loading"><RefreshCw size={13} /> Loading checkpoints</span>
      ) : null}
      {recentCheckpoints.length > 0 ? (
        <div className="workspace-git-checkpoints" aria-label="Workspace checkpoints">
          <header>
            <span><History size={13} aria-hidden="true" /> Recent checkpoints</span>
            {checkpointsLoading ? <em>refreshing</em> : null}
          </header>
          {recentCheckpoints.map((checkpoint) => (
            <div className="workspace-git-checkpoint" key={checkpoint.checkpoint_id}>
              <span>
                <strong title={checkpoint.checkpoint_id}>{checkpointLabel(checkpoint)}</strong>
                <small>{formatCheckpointMeta(checkpoint)}</small>
              </span>
              <button type="button" onClick={() => onRestoreCheckpoint(checkpoint)} disabled={mutating}>
                <RotateCcw size={12} />
                Restore
              </button>
            </div>
          ))}
          {checkpoints.length > recentCheckpoints.length ? (
            <div className="workspace-git-more">{checkpoints.length - recentCheckpoints.length} older checkpoint{checkpoints.length - recentCheckpoints.length === 1 ? '' : 's'}</div>
          ) : null}
        </div>
      ) : null}
      {changedFiles.length > 0 ? (
        <div className="workspace-git-files" aria-label="Changed git files">
          {changedFiles.map((file) => (
            <div className="workspace-git-file" key={file.path}>
              <span>
                <strong>{file.path}</strong>
                <small>{file.status}{file.staged ? ' / staged' : ''}{file.untracked ? ' / untracked' : ''}</small>
              </span>
              <div>
                <button type="button" onClick={() => onOpenDiff(file)} disabled={diffLoadingPath === file.path}>
                  <GitCompare size={12} />
                  {diffLoadingPath === file.path ? 'Loading' : 'Diff'}
                </button>
                <button type="button" onClick={() => onRestore(file)} disabled={mutating}>
                  <RotateCcw size={12} />
                  Restore
                </button>
              </div>
            </div>
          ))}
          {status.files.length > changedFiles.length ? (
            <div className="workspace-git-more">{status.files.length - changedFiles.length} more changed file{status.files.length - changedFiles.length === 1 ? '' : 's'}</div>
          ) : null}
        </div>
      ) : null}
      {activeDiff ? (
        <article className="workspace-git-diff" aria-label="Workspace git diff">
          <header>
            <strong>{activeDiff.path || 'Workspace diff'}</strong>
            {activeDiff.truncated ? <em>truncated</em> : null}
          </header>
          {activeDiff.diff.trim() ? <DiffViewer diff={activeDiff.diff} /> : <p>No diff available for this path.</p>}
        </article>
      ) : null}
    </section>
  )
}

function WorkspaceNameDialog({ title, description, label, initialValue, submitting, onCancel, onSubmit }: WorkspaceNameDialogProps) {
  const [value, setValue] = useState(initialValue)
  const canSubmit = value.trim().length > 0 && !submitting

  return (
    <div className="modal-backdrop" role="presentation">
      <form
        className="prompt-modal workspace-action-modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onSubmit={(event) => {
          event.preventDefault()
          if (canSubmit) onSubmit(value)
        }}
      >
        <header>
          <span className="prompt-modal-icon" aria-hidden="true"><FolderTree size={17} /></span>
          <div className="prompt-modal-title">
            <strong>{title}</strong>
            <span>{description}</span>
          </div>
        </header>
        <div className="prompt-body workspace-action-modal-body">
          <label>
            <span>{label}</span>
            <input autoFocus value={value} onChange={(event) => setValue(event.target.value)} />
          </label>
        </div>
        <footer>
          <button type="button" className="prompt-action" onClick={onCancel} disabled={submitting}>Cancel</button>
          <button type="submit" className="prompt-action prompt-action-primary" disabled={!canSubmit}>{submitting ? 'Saving' : 'Apply'}</button>
        </footer>
      </form>
    </div>
  )
}


function shortenPath(path: string): string {
  const parts = path.split(/[\/]+/).filter(Boolean)
  if (parts.length <= 2) return path
  return parts.slice(-2).join('/')
}

function buildBreadcrumbs(path: string): Array<{ name: string; path: string }> {
  return path
    .split(/[\/]+/)
    .filter(Boolean)
    .map((name, index, parts) => ({
      name,
      path: parts.slice(0, index + 1).join('/'),
    }))
}

function normalizeInputPath(path: string): string {
  return path.trim().replace(/\\/g, '/').replace(/^\/+/, '').replace(/\/+$/, '')
}

function joinWorkspacePath(base: string, child: string): string {
  const normalizedChild = normalizeInputPath(child)
  if (!base) return normalizedChild
  return [normalizeInputPath(base), normalizedChild].filter(Boolean).join('/')
}

function parentPath(path: string): string {
  const parts = normalizeInputPath(path).split('/').filter(Boolean)
  parts.pop()
  return parts.join('/')
}

function checkpointLabel(checkpoint: WorkspaceCheckpoint): string {
  return checkpoint.label?.trim() || checkpoint.checkpoint_id
}

function formatCheckpointMeta(checkpoint: WorkspaceCheckpoint): string {
  const parts = [
    checkpoint.files.length.toLocaleString() + ' file' + (checkpoint.files.length === 1 ? '' : 's'),
    formatCheckpointTime(checkpoint.created_at),
  ].filter(Boolean)
  return parts.join(' · ')
}

function formatCheckpointTime(value: number): string | null {
  if (!Number.isFinite(value) || value <= 0) return null
  const millis = value > 10_000_000_000 ? value : value * 1000
  return new Date(millis).toLocaleString()
}
