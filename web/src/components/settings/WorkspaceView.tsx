import { ChevronLeft, File, Folder, RefreshCw, Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../../api/client'
import type { WorkspaceEntry, WorkspaceFile, WorkspaceTree } from '../../api/types'
import { MarkdownRenderer } from '../chat/MarkdownRenderer'
import { Spinner } from '../shared/Spinner'

export function WorkspaceView() {
  const [tree, setTree] = useState<WorkspaceTree | null>(null)
  const [activeFile, setActiveFile] = useState<WorkspaceFile | null>(null)
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<WorkspaceEntry[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const visibleEntries = searchResults ?? tree?.entries ?? []
  const title = tree?.path ? tree.path : 'Project root'
  const rootLabel = tree?.root ?? ''

  const loadTree = (path = tree?.path ?? '') => {
    setLoading(true)
    setError(null)
    setSearchResults(null)
    api.workspaceTree(path)
      .then((nextTree) => {
        setTree(nextTree)
        if (!activeFile && nextTree.entries.length > 0) {
          const firstFile = nextTree.entries.find((entry) => entry.kind === 'file')
          if (firstFile) void openFile(firstFile.path)
        }
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadTree('')
  }, [])

  const runSearch = () => {
    const value = query.trim()
    if (!value) {
      setSearchResults(null)
      return
    }
    setLoading(true)
    setError(null)
    api.workspaceSearch(value, 150)
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

  const openFile = (path: string) => {
    setLoading(true)
    setError(null)
    return api.workspaceFile(path)
      .then(setActiveFile)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }

  const lineCount = useMemo(() => activeFile?.content.split('\n').length ?? 0, [activeFile])

  return (
    <div className="settings-panel workspace-panel">
      <header className="panel-header">
        <div>
          <h2>Workspace</h2>
          <p>{rootLabel}</p>
        </div>
        <button type="button" onClick={() => loadTree(tree?.path ?? '')} disabled={loading}>
          <RefreshCw size={15} /> Refresh
        </button>
      </header>

      <div className="workspace-layout">
        <aside className="workspace-browser" aria-label="Workspace browser">
          <div className="workspace-browser-header">
            <strong>{title}</strong>
            {tree?.parent_path !== null && tree?.parent_path !== undefined ? (
              <button type="button" onClick={() => loadTree(tree.parent_path ?? '')} aria-label="Open parent directory">
                <ChevronLeft size={14} /> Parent
              </button>
            ) : null}
          </div>
          <form className="workspace-search" onSubmit={(event) => { event.preventDefault(); runSearch() }}>
            <Search size={14} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search files" />
            <button type="submit">Search</button>
          </form>
          {searchResults ? (
            <button type="button" className="workspace-clear-search" onClick={() => { setSearchResults(null); setQuery('') }}>
              Clear search
            </button>
          ) : null}
          <div className="workspace-entry-list">
            {visibleEntries.length === 0 && !loading ? <div className="empty-chat">No files found.</div> : null}
            {visibleEntries.map((entry) => (
              <button
                type="button"
                key={entry.path || '__root__'}
                className={activeFile?.path === entry.path ? 'active' : ''}
                onClick={() => openEntry(entry)}
              >
                {entry.kind === 'directory' ? <Folder size={15} /> : <File size={15} />}
                <span>{entry.name}</span>
                <small>{entry.path || '.'}</small>
              </button>
            ))}
          </div>
        </aside>

        <section className="workspace-preview" aria-label="Workspace file preview">
          {loading ? <Spinner label="Loading workspace" /> : null}
          {error ? <div className="notice notice-error">{error}</div> : null}
          {activeFile ? (
            <>
              <div className="workspace-preview-header">
                <div>
                  <strong>{activeFile.path}</strong>
                  <span>{activeFile.language} · {formatBytes(activeFile.size_bytes)} · {lineCount} lines</span>
                </div>
                {activeFile.truncated ? <span className="workspace-pill">truncated</span> : null}
                {activeFile.binary ? <span className="workspace-pill">binary</span> : null}
              </div>
              {activeFile.binary ? (
                <div className="empty-chat">Binary file preview is disabled.</div>
              ) : activeFile.language === 'markdown' ? (
                <div className="workspace-markdown"><MarkdownRenderer text={activeFile.content} /></div>
              ) : (
                <pre className="workspace-code">{activeFile.content}</pre>
              )}
            </>
          ) : !loading ? (
            <div className="empty-chat">Select a file to preview.</div>
          ) : null}
        </section>
      </div>
    </div>
  )
}

function formatBytes(value: number): string {
  if (value >= 1_000_000) return (value / 1_000_000).toFixed(1) + ' MB'
  if (value >= 1_000) return (value / 1_000).toFixed(1) + ' KB'
  return value + ' B'
}
