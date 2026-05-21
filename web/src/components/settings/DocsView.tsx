import { BookOpen, RefreshCw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { DocContent, DocSummary } from '../../api/types'
import { MarkdownRenderer } from '../chat/MarkdownRenderer'
import { Spinner } from '../shared/Spinner'

export function DocsView() {
  const [documents, setDocuments] = useState<DocSummary[]>([])
  const [activePath, setActivePath] = useState<string | null>(null)
  const [content, setContent] = useState<DocContent | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadIndex = (preferredPath: string | null = activePath) => {
    setLoading(true)
    setError(null)
    api.docs()
      .then((index) => {
        setDocuments(index.documents)
        const nextPath = preferredPath || index.default_path || index.documents[0]?.path || null
        setActivePath(nextPath)
        if (nextPath) return api.doc(nextPath)
        setContent(null)
        return null
      })
      .then((doc) => {
        if (doc) setContent(doc)
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadIndex(null)
  }, [])

  const openDocument = (path: string) => {
    setActivePath(path)
    setLoading(true)
    setError(null)
    api.doc(path)
      .then(setContent)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }

  return (
    <div className="settings-panel docs-panel">
      <header className="panel-header">
        <div>
          <h2>Docs</h2>
          <p>Browse local Aether markdown documentation from the project docs directory.</p>
        </div>
        <button type="button" onClick={() => loadIndex(activePath)} disabled={loading}>
          <RefreshCw size={15} /> Refresh
        </button>
      </header>

      {loading ? <Spinner label="Loading docs" /> : null}
      {error ? <div className="notice notice-error">{error}</div> : null}

      <div className="docs-layout">
        <aside className="docs-list" aria-label="Documentation files">
          {documents.length === 0 && !loading ? <div className="empty-chat">No markdown docs found.</div> : null}
          {documents.map((document) => (
            <button
              type="button"
              key={document.path}
              className={document.path === activePath ? 'active' : ''}
              onClick={() => openDocument(document.path)}
            >
              <BookOpen size={14} />
              <span>{document.title}</span>
              <small>{document.path}</small>
            </button>
          ))}
        </aside>

        <article className="docs-content" aria-label="Documentation content">
          {content ? (
            <>
              <div className="docs-meta">
                <span>{content.path}</span>
                <span>{formatBytes(content.size_bytes)}</span>
              </div>
              <MarkdownRenderer text={content.content} />
            </>
          ) : documents.length > 0 && !loading ? (
            <div className="empty-chat">Select a document.</div>
          ) : null}
        </article>
      </div>
    </div>
  )
}

function formatBytes(value: number): string {
  if (value >= 1_000_000) return (value / 1_000_000).toFixed(1) + ' MB'
  if (value >= 1_000) return (value / 1_000).toFixed(1) + ' KB'
  return value + ' B'
}
