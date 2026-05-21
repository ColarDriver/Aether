import { ChevronLeft, ExternalLink, File, Folder, RefreshCw, Search, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../../api/client'
import type { WorkspaceEntry, WorkspaceFile, WorkspaceTree } from '../../api/types'
import { Button } from '../shared/Button'
import { Spinner } from '../shared/Spinner'
import { MarkdownRenderer } from './MarkdownRenderer'

type Props = {
  onClose?: () => void
  onOpenWorkspace?: () => void
}

export function WorkspaceRail({ onClose, onOpenWorkspace }: Props) {
  const [tree, setTree] = useState<WorkspaceTree | null>(null)
  const [activeFile, setActiveFile] = useState<WorkspaceFile | null>(null)
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<WorkspaceEntry[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const visibleEntries = searchResults ?? tree?.entries ?? []
  const title = tree?.path ? tree.path : 'Project root'
  const rootLabel = useMemo(() => shortenPath(tree?.root ?? ''), [tree?.root])

  const openFile = useCallback((path: string) => {
    setLoading(true)
    setError(null)
    return api.workspaceFile(path)
      .then(setActiveFile)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }, [])

  const loadTree = useCallback((path = '') => {
    setLoading(true)
    setError(null)
    setSearchResults(null)
    api.workspaceTree(path)
      .then((nextTree) => setTree(nextTree))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    loadTree('')
  }, [loadTree])

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
    void openFile(entry.path)
  }

  return (
    <aside className="workspace-rail" aria-label="Workspace files">
      <header className="workspace-rail-header">
        <div>
          <strong>Workspace</strong>
          <span title={tree?.root ?? ''}>{rootLabel || 'Loading root'}</span>
        </div>
        <div className="workspace-rail-actions">
          <Button title="Refresh workspace" aria-label="Refresh workspace" onClick={() => loadTree(tree?.path ?? '')} disabled={loading}>
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

      <div className="workspace-rail-browser">
        <div className="workspace-rail-path">
          <strong>{title}</strong>
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
            <button
              type="button"
              key={entry.path || '__root__'}
              className={activeFile?.path === entry.path ? 'workspace-rail-entry active' : 'workspace-rail-entry'}
              onClick={() => openEntry(entry)}
              title={entry.path || '.'}
            >
              {entry.kind === 'directory' ? <Folder size={15} /> : <File size={15} />}
              <span>{entry.name}</span>
              <small>{entry.path || '.'}</small>
            </button>
          ))}
        </div>
      </div>

      <section className="workspace-rail-preview" aria-label="Workspace preview">
        {loading && activeFile ? <div className="workspace-rail-loading">Updating preview...</div> : null}
        {activeFile ? (
          <>
            <div className="workspace-rail-preview-header">
              <strong>{activeFile.path}</strong>
              <span>{activeFile.language} · {formatBytes(activeFile.size_bytes)}</span>
            </div>
            {activeFile.binary ? (
              <div className="empty-chat">Binary file preview is disabled.</div>
            ) : activeFile.language === 'markdown' ? (
              <div className="workspace-rail-markdown"><MarkdownRenderer text={activeFile.content} /></div>
            ) : (
              <pre className="workspace-rail-code">{activeFile.content}</pre>
            )}
            {activeFile.truncated ? <div className="workspace-rail-note">Preview truncated.</div> : null}
          </>
        ) : (
          <div className="empty-chat">Select a file to preview.</div>
        )}
      </section>
    </aside>
  )
}

function formatBytes(value: number): string {
  if (value >= 1_000_000) return (value / 1_000_000).toFixed(1) + ' MB'
  if (value >= 1_000) return (value / 1_000).toFixed(1) + ' KB'
  return value + ' B'
}

function shortenPath(path: string): string {
  const parts = path.split(/[\\/]+/).filter(Boolean)
  if (parts.length <= 2) return path
  return parts.slice(-2).join('/')
}
