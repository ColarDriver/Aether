import { ChevronRight, ExternalLink, File, FilePlus, Folder, FolderOpen, FolderPlus, FolderTree, GitBranch, GitCompare, History, Pencil, RefreshCw, RotateCcw, Search, Trash2, X } from 'lucide-react'
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

type WorkspaceRailTab = 'files' | 'git'

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
  const [treeCache, setTreeCache] = useState<Record<string, WorkspaceTree>>({})
  const [expandedPaths, setExpandedPaths] = useState<string[]>([])
  const [treeLoadingPaths, setTreeLoadingPaths] = useState<string[]>([])
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
  const [activeTab, setActiveTab] = useState<WorkspaceRailTab>('files')
  const onTreeLoadedRef = useRef(onTreeLoaded)
  const onWorkspaceRootChangedRef = useRef(onWorkspaceRootChanged)

  const rootTree = treeCache[''] ?? (tree?.path ? null : tree)
  const rootEntries = rootTree?.entries ?? []
  const visibleEntries = searchResults ?? rootEntries
  const currentRoot = rootInfo?.root ?? tree?.root ?? ''
  const rootLabel = useMemo(() => shortenPath(currentRoot), [currentRoot])
  const currentPath = tree?.path ?? ''
  const gitChangeCount = gitStatus?.available ? gitStatus.files.length : 0

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

  const loadTree = useCallback((path = '', options: { setCurrent?: boolean; preserveSearch?: boolean; expand?: boolean } = {}) => {
    const normalizedPath = normalizeTreePath(path)
    if (normalizedPath === '') setLoading(true)
    setTreeLoadingPaths((current) => current.includes(normalizedPath) ? current : [...current, normalizedPath])
    setError(null)
    if (!options.preserveSearch) setSearchResults(null)
    api.workspaceTree(normalizedPath)
      .then((nextTree) => {
        const cacheKey = normalizeTreePath(nextTree.path ?? normalizedPath)
        setTreeCache((current) => ({ ...current, [cacheKey]: nextTree }))
        if (options.setCurrent !== false) setTree(nextTree)
        if (options.expand && cacheKey) {
          setExpandedPaths((current) => current.includes(cacheKey) ? current : [...current, cacheKey])
        }
        onTreeLoadedRef.current?.(nextTree)
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => {
        if (normalizedPath === '') setLoading(false)
        setTreeLoadingPaths((current) => current.filter((item) => item !== normalizedPath))
      })
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
    loadTree('', { setCurrent: currentPath === '' })
    if (currentPath) loadTree(currentPath, { preserveSearch: true, expand: true })
    loadGitStatus()
    loadCheckpoints()
  }, [currentPath, loadCheckpoints, loadGitStatus, loadRoot, loadTree])

  const refreshActiveTab = useCallback(() => {
    if (activeTab === 'git') {
      loadGitStatus()
      loadCheckpoints()
      return
    }
    loadRoot()
    loadTree('', { setCurrent: currentPath === '' })
    if (currentPath) loadTree(currentPath, { preserveSearch: true, expand: true })
  }, [activeTab, currentPath, loadCheckpoints, loadGitStatus, loadRoot, loadTree])

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
      setTreeCache({})
      setExpandedPaths([])
      setTreeLoadingPaths([])
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
    if (entry.kind !== 'directory') {
      onSelectFile?.(entry.path)
      return
    }
    const path = normalizeTreePath(entry.path)
    const expanded = expandedPaths.includes(path)
    if (searchResults) {
      setSearchResults(null)
      setQuery('')
    }
    if (expanded) {
      setExpandedPaths((current) => current.filter((item) => item !== path))
      if (treeCache[path]) setTree(treeCache[path])
      return
    }
    setExpandedPaths((current) => current.includes(path) ? current : [...current, path])
    if (treeCache[path]) {
      setTree(treeCache[path])
      return
    }
    loadTree(path, { preserveSearch: true, expand: true })
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
        loadTree(parentPath(created.path), { expand: Boolean(parentPath(created.path)) })
        onSelectFile?.(created.path)
      } else if (actionDialog.kind === 'new-folder') {
        const created = await api.workspaceCreateDirectory(joinWorkspacePath(currentPath, normalized))
        setActionDialog(null)
        loadTree(parentPath(created.path), { expand: Boolean(parentPath(created.path)) })
      } else {
        const renamed = await api.workspaceRenamePath(actionDialog.entry.path, normalized)
        setActionDialog(null)
        loadTree(parentPath(renamed.path), { expand: Boolean(parentPath(renamed.path)) })
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
            <Button title="Refresh workspace" aria-label="Refresh workspace" onClick={refreshActiveTab} disabled={loading || gitLoading || mutating}>
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

        <nav className="workspace-rail-tabs" role="tablist" aria-label="Workspace panels">
          <button
            type="button"
            role="tab"
            id="workspace-rail-tab-files"
            aria-controls="workspace-rail-panel-files"
            aria-selected={activeTab === 'files'}
            className={activeTab === 'files' ? 'active' : ''}
            onClick={() => setActiveTab('files')}
          >
            <FolderTree size={15} />
            <span>Files</span>
          </button>
          <button
            type="button"
            role="tab"
            id="workspace-rail-tab-git"
            aria-controls="workspace-rail-panel-git"
            aria-selected={activeTab === 'git'}
            className={activeTab === 'git' ? 'active' : ''}
            onClick={() => setActiveTab('git')}
          >
            <GitBranch size={15} />
            <span>Source Control</span>
            {gitChangeCount > 0 ? <em>{gitChangeCount > 99 ? '99+' : gitChangeCount}</em> : null}
          </button>
        </nav>

        {activeTab === 'files' ? (
          <section
            id="workspace-rail-panel-files"
            className="workspace-rail-tab-panel workspace-rail-tab-panel-files"
            role="tabpanel"
            aria-labelledby="workspace-rail-tab-files"
          >
            <WorkspaceRootStrip
              info={rootInfo}
              error={rootError}
              loading={rootLoading}
              mutating={mutating}
              onSwitch={switchWorkspaceRoot}
            />

            <div className="workspace-rail-browser workspace-rail-browser-tree" aria-label="Workspace browser">
              <form className="workspace-rail-search" onSubmit={(event) => { event.preventDefault(); runSearch() }}>
                <Search size={14} />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search files..." />
                {searchResults ? (
                  <button type="button" aria-label="Clear workspace search" onClick={() => { setSearchResults(null); setQuery('') }}>
                    <X size={14} />
                  </button>
                ) : (
                  <button type="submit">Search</button>
                )}
              </form>
              {error ? <div className="workspace-rail-error">{error}</div> : null}
              <div className="workspace-rail-tree" aria-label={searchResults ? 'Workspace search results' : 'Workspace file tree'}>
                {loading && rootEntries.length === 0 && !searchResults ? <Spinner label="Loading workspace" /> : null}
                {searchResults ? (
                  <>
                    <div className="workspace-rail-tree-search-heading">Search &quot;{query.trim()}&quot;</div>
                    {visibleEntries.length === 0 && !loading ? <div className="empty-chat">No files found.</div> : null}
                    <WorkspaceTreeRows
                      entries={visibleEntries}
                      level={0}
                      selectedFilePath={selectedFilePath}
                      expandedPaths={expandedPaths}
                      treeCache={treeCache}
                      treeLoadingPaths={treeLoadingPaths}
                      mutating={mutating}
                      searchMode
                      onOpen={openEntry}
                      onRename={openRenameDialog}
                      onDelete={setDeleteTarget}
                    />
                  </>
                ) : (
                  <>
                    <div className="workspace-rail-tree-root" title={currentRoot || 'Workspace root'}>
                      <FolderOpen size={15} aria-hidden="true" />
                      <span>{rootInfo?.name || rootLabel || 'Project root'}</span>
                    </div>
                    {rootEntries.length === 0 && !loading ? <div className="empty-chat">No files found.</div> : null}
                    <WorkspaceTreeRows
                      entries={rootEntries}
                      level={1}
                      selectedFilePath={selectedFilePath}
                      expandedPaths={expandedPaths}
                      treeCache={treeCache}
                      treeLoadingPaths={treeLoadingPaths}
                      mutating={mutating}
                      onOpen={openEntry}
                      onRename={openRenameDialog}
                      onDelete={setDeleteTarget}
                    />
                  </>
                )}
              </div>
            </div>
          </section>
        ) : (
          <section
            id="workspace-rail-panel-git"
            className="workspace-rail-tab-panel workspace-rail-tab-panel-git"
            role="tabpanel"
            aria-labelledby="workspace-rail-tab-git"
          >
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
          </section>
        )}
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


function WorkspaceTreeRows({
  entries,
  level,
  selectedFilePath,
  expandedPaths,
  treeCache,
  treeLoadingPaths,
  mutating,
  searchMode = false,
  onOpen,
  onRename,
  onDelete,
}: {
  entries: WorkspaceEntry[]
  level: number
  selectedFilePath?: string | null
  expandedPaths: string[]
  treeCache: Record<string, WorkspaceTree>
  treeLoadingPaths: string[]
  mutating: boolean
  searchMode?: boolean
  onOpen: (entry: WorkspaceEntry) => void
  onRename: (entry: WorkspaceEntry) => void
  onDelete: (entry: WorkspaceEntry) => void
}) {
  return (
    <>
      {entries.map((entry) => {
        const isDirectory = entry.kind === 'directory'
        const path = normalizeTreePath(entry.path)
        const expanded = isDirectory && expandedPaths.includes(path)
        const children = treeCache[path]?.entries ?? []
        const loading = isDirectory && treeLoadingPaths.includes(path)
        const hasLoaded = Boolean(treeCache[path])
        return (
          <div className="workspace-rail-tree-node" key={entry.path || '__root__'}>
            <div
              className={'workspace-rail-tree-row workspace-rail-entry-kind-' + entry.kind + (selectedFilePath === entry.path ? ' active' : '') + (expanded ? ' expanded' : '')}
              style={{ paddingLeft: 8 + level * 16 }}
            >
              <button
                type="button"
                className="workspace-rail-tree-main"
                onClick={() => onOpen(entry)}
                title={entry.path || '.'}
                aria-expanded={isDirectory ? expanded : undefined}
              >
                {isDirectory ? (
                  <ChevronRight className="workspace-rail-tree-chevron" size={13} aria-hidden="true" />
                ) : (
                  <span className="workspace-rail-tree-chevron workspace-rail-tree-chevron-empty" aria-hidden="true" />
                )}
                {isDirectory ? (expanded ? <FolderOpen size={15} /> : <Folder size={15} />) : <File size={15} />}
                <span>{entry.name}</span>
                {searchMode ? <small>{entry.path || '.'}</small> : null}
              </button>
              <div className="workspace-rail-entry-actions" aria-label={'Actions for ' + entry.name}>
                <button type="button" aria-label={'Rename ' + entry.name} title="Rename" onClick={() => onRename(entry)} disabled={mutating}>
                  <Pencil size={13} />
                </button>
                <button type="button" aria-label={'Delete ' + entry.name} title="Delete" onClick={() => onDelete(entry)} disabled={mutating}>
                  <Trash2 size={13} />
                </button>
              </div>
            </div>
            {isDirectory && expanded ? (
              <div className="workspace-rail-tree-children">
                {loading ? <div className="workspace-rail-tree-loading" style={{ paddingLeft: 24 + (level + 1) * 16 }}><Spinner label={'Loading ' + entry.name} /></div> : null}
                {!loading && hasLoaded && children.length === 0 ? <div className="workspace-rail-tree-empty" style={{ paddingLeft: 24 + (level + 1) * 16 }}>Empty folder</div> : null}
                {children.length > 0 ? (
                  <WorkspaceTreeRows
                    entries={children}
                    level={level + 1}
                    selectedFilePath={selectedFilePath}
                    expandedPaths={expandedPaths}
                    treeCache={treeCache}
                    treeLoadingPaths={treeLoadingPaths}
                    mutating={mutating}
                    onOpen={onOpen}
                    onRename={onRename}
                    onDelete={onDelete}
                  />
                ) : null}
              </div>
            ) : null}
          </div>
        )
      })}
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
  const hasSyncState = status.ahead > 0 || status.behind > 0
  return (
    <section className={'workspace-git-panel' + (status.clean ? ' workspace-git-panel-clean' : ' workspace-git-panel-dirty')} aria-label="Repository status">
      <header className="workspace-git-summary">
        <span className="workspace-git-branch-icon" aria-hidden="true"><GitBranch size={15} /></span>
        <span className="workspace-git-branch-text">
          <strong>{status.branch || 'detached'}</strong>
          {status.upstream ? <small>{status.upstream}</small> : null}
        </span>
        <em className={status.clean ? 'workspace-git-clean' : 'workspace-git-dirty'}>
          {status.clean ? 'clean' : status.files.length.toLocaleString() + ' changed'}
        </em>
        {hasSyncState ? (
          <span className="workspace-git-sync">
            {status.ahead > 0 ? <span>{status.ahead} ahead</span> : null}
            {status.behind > 0 ? <span>{status.behind} behind</span> : null}
          </span>
        ) : null}
      </header>
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
      <section className="workspace-git-section workspace-git-section-changes" aria-label="Changed git files">
        <header className="workspace-git-section-header">
          <span><GitCompare size={13} aria-hidden="true" /> Changes</span>
          <em>{status.files.length.toLocaleString()}</em>
        </header>
        {changedFiles.length > 0 ? (
          <div className="workspace-git-files">
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
        ) : (
          <div className="workspace-git-empty-state">
            <GitBranch size={14} aria-hidden="true" />
            Working tree clean
          </div>
        )}
      </section>
      {activeDiff ? (
        <article className="workspace-git-diff" aria-label="Workspace git diff">
          <header>
            <strong>{activeDiff.path || 'Workspace diff'}</strong>
            {activeDiff.truncated ? <em>truncated</em> : null}
          </header>
          {activeDiff.diff.trim() ? <DiffViewer diff={activeDiff.diff} /> : <p>No diff available for this path.</p>}
        </article>
      ) : null}
      <section className="workspace-git-section workspace-git-checkpoints" aria-label="Workspace checkpoints">
        <header className="workspace-git-section-header">
          <span><History size={13} aria-hidden="true" /> Recent checkpoints</span>
          {checkpointsLoading ? <em>refreshing</em> : recentCheckpoints.length > 0 ? <em>{checkpoints.length.toLocaleString()}</em> : null}
        </header>
        {checkpointsLoading && recentCheckpoints.length === 0 ? (
          <span className="workspace-git-loading"><RefreshCw size={13} /> Loading checkpoints</span>
        ) : null}
        {recentCheckpoints.length > 0 ? (
          <div className="workspace-git-checkpoint-list">
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
        ) : !checkpointsLoading ? (
          <div className="workspace-git-empty-state">
            <History size={14} aria-hidden="true" />
            No checkpoints yet
          </div>
        ) : null}
      </section>
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


function normalizeTreePath(path: string): string {
  return normalizeInputPath(path)
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
