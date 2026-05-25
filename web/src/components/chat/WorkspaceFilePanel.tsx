import { FileCode, X } from 'lucide-react'
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
}

export function WorkspaceFilePanel({ preview, onClose }: Props) {
  const path = preview.file?.path || preview.path || ''
  const name = preview.file?.name || basename(path)
  const language = preview.file?.language || languageFromPath(path)
  const lineCount = preview.file ? countLines(preview.file.content) : null
  const isMarkdown = Boolean(preview.file && isMarkdownFile(preview.file))

  return (
    <aside className="workspace-file-panel" aria-label="Workspace file preview">
      <header className="workspace-file-panel-header">
        <span className="workspace-file-panel-icon" aria-hidden="true"><FileCode size={16} /></span>
        <div>
          <strong>{name || 'Workspace file'}</strong>
          <span title={path}>{path || 'No file selected'}</span>
        </div>
        {onClose ? (
          <button type="button" aria-label="Close workspace file preview" onClick={onClose}>
            <X size={15} />
          </button>
        ) : null}
      </header>

      <div className="workspace-file-panel-meta">
        <span>{language || 'text'}</span>
        {preview.file ? <span>{formatBytes(preview.file.size_bytes)}</span> : null}
        {lineCount != null ? <span>{lineCount} line{lineCount === 1 ? '' : 's'}</span> : null}
        {preview.file?.truncated ? <span>truncated</span> : null}
      </div>

      <div className="workspace-file-panel-body">
        {preview.loading ? <Spinner label="Loading file" /> : null}
        {preview.error ? <div className="workspace-file-panel-state workspace-file-panel-error">{preview.error}</div> : null}
        {!preview.loading && !preview.error && preview.file?.binary ? (
          <div className="workspace-file-panel-state">Binary preview unavailable.</div>
        ) : null}
        {!preview.loading && !preview.error && preview.file && !preview.file.binary && isMarkdown ? (
          <div className="workspace-file-panel-markdown">
            <MarkdownRenderer text={preview.file.content} />
          </div>
        ) : null}
        {!preview.loading && !preview.error && preview.file && !preview.file.binary && !isMarkdown ? (
          <pre className="workspace-file-panel-code">
            <code>{highlightCode(preview.file.content, language)}</code>
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
  return extension || 'text'
}

function isMarkdownFile(file: WorkspaceFile): boolean {
  const language = file.language.toLowerCase()
  return language === 'markdown' || language === 'md' || /\.(md|mdx)$/i.test(file.path)
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
