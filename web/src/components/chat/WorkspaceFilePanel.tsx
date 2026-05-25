import { FileCode, Image as ImageIcon, Pencil, Save, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { WorkspaceFile } from '../../api/types'
import { Spinner } from '../shared/Spinner'
import { MarkdownRenderer } from './MarkdownRenderer'
import { highlightCode } from './blocks/CodeBlock'

export type WorkspaceFilePreviewState = {
  path: string | null
  file: WorkspaceFile | null
  loading: boolean
  error: string | null
}

type Props = {
  preview: WorkspaceFilePreviewState
  onClose?: () => void
  onSave?: (path: string, content: string) => Promise<WorkspaceFile>
}

export function WorkspaceFilePanel({ preview, onClose, onSave }: Props) {
  const path = preview.file?.path || preview.path || ''
  const name = preview.file?.name || basename(path)
  const language = preview.file?.language || languageFromPath(path)
  const mimeType = preview.file?.mime_type || mimeTypeFromPath(path)
  const fileContent = preview.file?.content ?? ''
  const isMarkdown = Boolean(preview.file && isMarkdownFile(preview.file))
  const isImage = Boolean(preview.file && isImageFile(preview.file, mimeType))
  const lineCount = preview.file && !preview.file.binary && !isImage ? countLines(fileContent) : null
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(fileContent)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageError, setImageError] = useState<string | null>(null)
  const canEdit = Boolean(onSave && preview.file && !preview.loading && !preview.error && !preview.file.binary && !preview.file.truncated && !isImage)
  const dirty = Boolean(preview.file && draft !== fileContent)

  useEffect(() => {
    setEditing(false)
    setDraft(fileContent)
    setSaveError(null)
  }, [preview.file?.path, fileContent])

  useEffect(() => {
    let cancelled = false
    let objectUrl: string | null = null
    setImageUrl(null)
    setImageError(null)

    if (!isImage || !preview.file || preview.loading || preview.error) {
      return () => undefined
    }

    api.workspaceFileBlob(preview.file.path)
      .then((blob) => {
        if (cancelled) return
        if (typeof URL.createObjectURL !== 'function') {
          setImageError('Image preview is unavailable in this browser.')
          return
        }
        objectUrl = URL.createObjectURL(blob)
        setImageUrl(objectUrl)
      })
      .catch((error: unknown) => {
        if (!cancelled) setImageError(error instanceof Error ? error.message : String(error))
      })

    return () => {
      cancelled = true
      if (objectUrl && typeof URL.revokeObjectURL === 'function') URL.revokeObjectURL(objectUrl)
    }
  }, [isImage, preview.file?.path, preview.loading, preview.error])

  const beginEdit = () => {
    if (!canEdit || !preview.file) return
    setDraft(fileContent)
    setSaveError(null)
    setEditing(true)
  }

  const cancelEdit = () => {
    setDraft(fileContent)
    setSaveError(null)
    setEditing(false)
  }

  const saveEdit = async () => {
    if (!canEdit || !preview.file || saving) return
    setSaving(true)
    setSaveError(null)
    try {
      const saved = await onSave?.(preview.file.path, draft)
      if (saved) setDraft(saved.content ?? '')
      setEditing(false)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : String(error))
    } finally {
      setSaving(false)
    }
  }

  return (
    <aside className="workspace-file-panel" aria-label="Workspace file preview">
      <header className="workspace-file-panel-header">
        <span className="workspace-file-panel-icon" aria-hidden="true">{isImage ? <ImageIcon size={16} /> : <FileCode size={16} />}</span>
        <div className="workspace-file-panel-title">
          <strong>{name || 'Workspace file'}</strong>
          <span title={path}>{path || 'No file selected'}</span>
        </div>
        <div className="workspace-file-panel-actions">
          {editing ? (
            <>
              <button type="button" aria-label="Cancel workspace file edit" disabled={saving} onClick={cancelEdit}>
                <X size={14} />
                <span>Cancel</span>
              </button>
              <button type="button" aria-label="Save workspace file" disabled={saving || !dirty} onClick={saveEdit}>
                <Save size={14} />
                <span>{saving ? 'Saving' : 'Save'}</span>
              </button>
            </>
          ) : canEdit ? (
            <button type="button" aria-label="Edit workspace file" onClick={beginEdit}>
              <Pencil size={14} />
              <span>Edit</span>
            </button>
          ) : null}
          {onClose ? (
            <button type="button" aria-label="Close workspace file preview" onClick={onClose}>
              <X size={15} />
            </button>
          ) : null}
        </div>
      </header>

      <div className="workspace-file-panel-meta">
        <span>{language || 'text'}</span>
        {mimeType ? <span>{mimeType}</span> : null}
        {preview.file ? <span>{formatBytes(preview.file.size_bytes)}</span> : null}
        {lineCount != null ? <span>{lineCount} line{lineCount === 1 ? '' : 's'}</span> : null}
        {preview.file?.truncated ? <span>truncated</span> : null}
        {preview.file && !canEdit ? <span>read only</span> : null}
        {editing && dirty ? <span>modified</span> : null}
      </div>

      <div className="workspace-file-panel-body">
        {preview.loading ? <Spinner label="Loading file" /> : null}
        {preview.error ? <div className="workspace-file-panel-state workspace-file-panel-error">{preview.error}</div> : null}
        {saveError ? <div className="workspace-file-panel-state workspace-file-panel-error">{saveError}</div> : null}
        {!preview.loading && !preview.error && !preview.file ? (
          <div className="workspace-file-panel-state workspace-file-panel-state-fill">No file selected.</div>
        ) : null}
        {!preview.loading && !preview.error && preview.file?.binary && !isImage ? (
          <div className="workspace-file-panel-state workspace-file-panel-state-fill">Binary preview unavailable.</div>
        ) : null}
        {!preview.loading && !preview.error && preview.file && isImage ? (
          <figure className="workspace-file-panel-image">
            {imageUrl ? <img src={imageUrl} alt={name || path || 'Workspace image'} /> : null}
            {!imageUrl && !imageError ? <Spinner label="Loading image" /> : null}
            {imageError ? <div className="workspace-file-panel-state workspace-file-panel-error">{imageError}</div> : null}
          </figure>
        ) : null}
        {!preview.loading && !preview.error && preview.file && !preview.file.binary && !isImage && editing ? (
          <textarea
            aria-label="Workspace file editor"
            className="workspace-file-panel-editor"
            disabled={saving}
            spellCheck={false}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
          />
        ) : null}
        {!preview.loading && !preview.error && preview.file && !preview.file.binary && !isImage && !editing && isMarkdown ? (
          <div className="workspace-file-panel-markdown">
            <MarkdownRenderer text={fileContent} />
          </div>
        ) : null}
        {!preview.loading && !preview.error && preview.file && !preview.file.binary && !isImage && !editing && !isMarkdown ? (
          <pre className="workspace-file-panel-code">
            <code>{highlightCode(fileContent, language)}</code>
          </pre>
        ) : null}
      </div>
    </aside>
  )
}

function basename(path: string): string {
  const parts = path.split(/[\\/]+/).filter(Boolean)
  return parts.at(-1) ?? path
}

function languageFromPath(path: string): string {
  const extension = basename(path).split('.').pop()?.toLowerCase() ?? ''
  if (extension === 'md' || extension === 'mdx') return 'markdown'
  if (extension === 'py') return 'python'
  if (extension === 'ts' || extension === 'tsx') return 'typescript'
  if (extension === 'js' || extension === 'jsx') return 'javascript'
  if (extension === 'json') return 'json'
  if (extension === 'sh' || extension === 'bash' || extension === 'zsh') return 'bash'
  if (isImageExtension(extension)) return extension === 'svg' ? 'svg' : 'image'
  return extension || 'text'
}

function mimeTypeFromPath(path: string): string | null {
  const extension = basename(path).split('.').pop()?.toLowerCase() ?? ''
  if (extension === 'png') return 'image/png'
  if (extension === 'jpg' || extension === 'jpeg') return 'image/jpeg'
  if (extension === 'gif') return 'image/gif'
  if (extension === 'webp') return 'image/webp'
  if (extension === 'avif') return 'image/avif'
  if (extension === 'bmp') return 'image/bmp'
  if (extension === 'ico') return 'image/x-icon'
  if (extension === 'svg') return 'image/svg+xml'
  if (extension === 'md' || extension === 'mdx') return 'text/markdown'
  if (extension === 'txt') return 'text/plain'
  return null
}

function isMarkdownFile(file: WorkspaceFile): boolean {
  const language = (file.language || '').toLowerCase()
  return language === 'markdown' || language === 'md' || /\.(md|mdx)$/i.test(file.path || '')
}

function isImageFile(file: WorkspaceFile, mimeType: string | null): boolean {
  if (mimeType?.toLowerCase().startsWith('image/')) return true
  const extension = basename(file.path || file.name || '').split('.').pop()?.toLowerCase() ?? ''
  return isImageExtension(extension)
}

function isImageExtension(extension: string): boolean {
  return ['avif', 'bmp', 'gif', 'ico', 'jpeg', 'jpg', 'png', 'svg', 'webp'].includes(extension)
}

function countLines(content: string): number {
  if (!content) return 0
  return content.split(/\r\n|\r|\n/).length
}

function formatBytes(value: number): string {
  if (value >= 1_000_000) return (value / 1_000_000).toFixed(1) + ' MB'
  if (value >= 1_000) return (value / 1_000).toFixed(1) + ' KB'
  return value + ' B'
}
