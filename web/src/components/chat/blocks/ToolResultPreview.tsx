import { BookOpen, ChevronLeft, ChevronRight, ExternalLink, FileArchive, FileCode2, FileText, Globe, ImageIcon, Monitor, Pencil, Search, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { DiffViewer } from '../DiffViewer'
import { CopyButton } from '../../shared/CopyButton'
import { CodeBlock } from './CodeBlock'
import { InlineTaskSummary } from './InlineTaskSummary'

type Props = {
  block: ToolResult
  toolArguments?: Record<string, unknown>
}

type PreviewProps = {
  block: ToolResult
  toolArguments: Record<string, unknown>
}

type SearchMatch = {
  path?: string | null
  line?: number | null
  text: string
}

type WebResult = {
  title: string
  url: string | null
  snippet: string | null
}

type ToolImage = {
  src: string
  name: string
  caption: string | null
  href: string | null
}

type ToolArtifact = {
  name: string
  href: string | null
  path: string | null
  kind: string | null
  mimeType: string | null
  size: number | null
  note: string | null
}

export function canPreviewToolResult(block: ToolResult): boolean {
  const toolName = (block.toolName || '').toLowerCase()
  if (parseToolImages(block).length > 0) return true
  if (parseToolArtifacts(block).length > 0) return true
  if (isReadTool(toolName) || isTaskTool(toolName) || isEditTool(toolName) || isNotebookTool(toolName) || isLspTool(toolName) || isBrowserTool(toolName)) return true
  if (isSearchTool(toolName)) return parseSearchMatches(block.content).length > 0
  if (isWebTool(toolName)) return parseWebResults(block).length > 0
  return false
}

export function ToolResultPreview({ block, toolArguments = {} }: Props) {
  const toolName = (block.toolName || '').toLowerCase()
  if (isBrowserTool(toolName)) return <BrowserPreview block={block} toolArguments={toolArguments} />

  const toolImages = parseToolImages(block)
  if (toolImages.length > 0) return <ImagePreview images={toolImages} />

  if (isReadTool(toolName)) return <ReadFilePreview block={block} toolArguments={toolArguments} />
  if (isEditTool(toolName)) return <FileChangePreview block={block} toolArguments={toolArguments} />
  if (isNotebookTool(toolName)) return <NotebookPreview block={block} toolArguments={toolArguments} />
  if (isLspTool(toolName)) return <LspPreview block={block} toolArguments={toolArguments} />
  if (isSearchTool(toolName)) return <SearchPreview content={block.content} />
  if (isWebTool(toolName)) return <WebPreview block={block} toolName={block.toolName || 'web'} />
  if (isTaskTool(toolName)) return <TaskPreview block={block} toolArguments={toolArguments} />

  const toolArtifacts = parseToolArtifacts(block)
  if (toolArtifacts.length > 0) return <ArtifactPreview artifacts={toolArtifacts} />
  return null
}

function ReadFilePreview({ block, toolArguments }: PreviewProps) {
  const path = stringValue(toolArguments.path) || stringValue(toolArguments.file_path) || stringValue(toolArguments.filePath) || stringValue(block.metadata.path)
  const language = languageFromPath(path) || stringValue(block.metadata.language) || stringValue(block.metadata.lang) || 'text'
  return (
    <section className="tool-preview tool-preview-file" aria-label="File preview">
      <header>
        <span><FileText size={14} /><strong>{path || 'File content'}</strong></span>
        <em>{language}</em>
      </header>
      <CodeBlock code={block.content} language={language} wrap />
    </section>
  )
}

function FileChangePreview({ block, toolArguments }: PreviewProps) {
  const path = firstString(block.metadata.path, toolArguments.path, toolArguments.file_path, toolArguments.filePath)
    || firstStringFromArray(block.metadata.edited_paths)
    || 'File change'
  const additions = numberValue(block.metadata.lines_added) ?? numberValue(block.metadata.linesAdded)
  const removals = numberValue(block.metadata.lines_removed) ?? numberValue(block.metadata.linesRemoved)
  const hunks = numberValue(block.metadata.hunks)
  const changeCount = numberValue(block.metadata.change_count) ?? numberValue(block.metadata.changeCount)
  const stats = [
    additions != null ? { label: 'added', value: '+' + additions.toLocaleString(), tone: 'add' } : null,
    removals != null ? { label: 'removed', value: '-' + removals.toLocaleString(), tone: 'remove' } : null,
    hunks != null ? { label: 'hunks', value: hunks.toLocaleString() } : null,
    changeCount != null ? { label: 'changes', value: changeCount.toLocaleString() } : null,
    typeof block.metadata.existed === 'boolean' ? { label: 'mode', value: block.metadata.existed ? 'overwrite' : 'create' } : null,
  ].filter((item): item is { label: string; value: string; tone?: string } => Boolean(item))
  return (
    <section className="tool-preview tool-preview-edit" aria-label="File change">
      <header>
        <span><Pencil size={14} /><strong>{path}</strong></span>
        <em>{block.isError ? 'failed' : 'changed'}</em>
      </header>
      {stats.length > 0 ? <PreviewStats stats={stats} /> : null}
      {block.content ? <pre className="tool-preview-summary"><code>{block.content}</code></pre> : null}
    </section>
  )
}

function NotebookPreview({ block, toolArguments }: PreviewProps) {
  const path = firstString(block.metadata.path, toolArguments.notebook_path, toolArguments.path) || 'Notebook'
  const mode = firstString(block.metadata.edit_mode, toolArguments.edit_mode) || 'edit'
  const cellRef = firstString(toolArguments.cell_id, block.metadata.cell_id, block.metadata.cellId)
    || numberLabel('cell', numberValue(toolArguments.cell_idx) ?? numberValue(toolArguments.cellIdx))
  const cellType = firstString(toolArguments.cell_type, block.metadata.cell_type, block.metadata.cellType) || 'code'
  const cellCount = numberValue(block.metadata.cell_count) ?? numberValue(block.metadata.cellCount)
  const stats = [
    { label: 'mode', value: mode },
    cellRef ? { label: 'cell', value: cellRef } : null,
    cellType ? { label: 'type', value: cellType } : null,
    cellCount != null ? { label: 'cells', value: cellCount.toLocaleString() } : null,
  ].filter((item): item is { label: string; value: string } => Boolean(item))
  return (
    <section className="tool-preview tool-preview-notebook" aria-label="Notebook edit">
      <header>
        <span><BookOpen size={14} /><strong>{path}</strong></span>
        <em>{block.isError ? 'failed' : mode}</em>
      </header>
      <PreviewStats stats={stats} />
      <NotebookCellPreview block={block} toolArguments={toolArguments} cellType={cellType} mode={mode} />
      {block.content ? <pre className="tool-preview-summary"><code>{block.content}</code></pre> : null}
    </section>
  )
}

function NotebookCellPreview({ block, toolArguments, cellType, mode }: PreviewProps & { cellType: string; mode: string }) {
  const parsedContent = parseJson(block.content)
  const contentRecord = isRecord(parsedContent) ? parsedContent : {}
  const diff = firstString(block.metadata.diff, toolArguments.diff, contentRecord.diff)
  const oldSource = firstString(block.metadata.old_source, block.metadata.oldSource, toolArguments.old_source, toolArguments.oldSource, contentRecord.old_source, contentRecord.oldSource)
  const newSource = firstString(block.metadata.new_source, block.metadata.newSource, toolArguments.new_source, toolArguments.newSource, contentRecord.new_source, contentRecord.newSource)
  if (!diff && !oldSource && !newSource) return null

  if (diff || (oldSource && newSource)) {
    return (
      <div className="notebook-cell-preview" aria-label="Notebook cell diff">
        <header>
          <strong>Cell diff</strong>
          <span>{cellType}</span>
        </header>
        <DiffViewer diff={diff || diffFromOldNew(oldSource || '', newSource || '')} />
      </div>
    )
  }

  const title = mode === 'delete' ? 'Removed cell source' : 'Cell source'
  return (
    <div className="notebook-cell-preview" aria-label="Notebook cell source">
      <CodeBlock code={newSource || oldSource || ''} language={notebookLanguage(cellType)} title={title} wrap />
    </div>
  )
}

function notebookLanguage(cellType: string): string {
  return cellType === 'markdown' ? 'markdown' : 'python'
}

function diffFromOldNew(oldText: string, newText: string): string {
  const oldLines = oldText ? oldText.split('\n').map((line) => '-' + line) : []
  const newLines = newText ? newText.split('\n').map((line) => '+' + line) : []
  return [...oldLines, ...newLines].join('\n')
}

function LspPreview({ block, toolArguments }: PreviewProps) {
  const operation = firstString(block.metadata.operation, toolArguments.operation) || 'lsp'
  const target = firstString(block.metadata.file_path, block.metadata.filePath, toolArguments.filePath, toolArguments.query, block.metadata.query)
  const matches = parseMarkdownBullets(block.content)
  return (
    <section className="tool-preview tool-preview-lsp" aria-label="LSP result">
      <header>
        <span><FileCode2 size={14} /><strong>{operation}</strong></span>
        <em>{block.isError ? 'failed' : 'lsp'}</em>
      </header>
      {target ? <PreviewStats stats={[{ label: operation === 'workspaceSymbol' ? 'query' : 'target', value: target }]} /> : null}
      {matches.length > 0 ? (
        <ol className="tool-preview-lsp-list">
          {matches.slice(0, 30).map((match, index) => <li key={index}><code>{match}</code></li>)}
        </ol>
      ) : block.content ? (
        <pre className="tool-preview-summary"><code>{block.content}</code></pre>
      ) : null}
    </section>
  )
}

function BrowserPreview({ block, toolArguments }: PreviewProps) {
  const operation = firstString(block.metadata.operation, toolArguments.operation) || 'browser'
  const url = firstString(block.metadata.url, toolArguments.url)
  const selector = firstString(block.metadata.selector, toolArguments.selector)
  const screenshotPath = firstString(block.metadata.screenshot_path, block.metadata.screenshotPath)
  const stats = [
    url ? { label: 'url', value: url } : null,
    selector ? { label: 'selector', value: selector } : null,
    numberValue(block.metadata.bytes) != null ? { label: 'bytes', value: numberValue(block.metadata.bytes)!.toLocaleString() } : null,
  ].filter((item): item is { label: string; value: string } => Boolean(item))
  return (
    <section className="tool-preview tool-preview-browser" aria-label="Browser result">
      <header>
        <span><Monitor size={14} /><strong>{operation}</strong></span>
        <em>{block.isError ? 'failed' : 'browser'}</em>
      </header>
      {stats.length > 0 ? <PreviewStats stats={stats} /> : null}
      {screenshotPath ? (
        <div className="tool-preview-artifact">
          <ImageIcon size={14} aria-hidden="true" />
          <span>
            <strong>Screenshot saved</strong>
            <code>{screenshotPath}</code>
          </span>
        </div>
      ) : block.content ? (
        <pre className="tool-preview-summary"><code>{block.content}</code></pre>
      ) : null}
    </section>
  )
}

function PreviewStats({ stats }: { stats: Array<{ label: string; value: string; tone?: string }> }) {
  if (stats.length === 0) return null
  return (
    <div className="tool-preview-stats">
      {stats.map((stat) => (
        <span className={stat.tone ? 'tool-preview-stat-' + stat.tone : undefined} key={stat.label} title={stat.value}>
          <strong>{stat.value}</strong>
          <small>{stat.label}</small>
        </span>
      ))}
    </div>
  )
}

function SearchPreview({ content }: { content: string }) {
  const matches = parseSearchMatches(content)
  if (matches.length === 0) return null
  return (
    <section className="tool-preview tool-preview-search" aria-label="Search results">
      <header>
        <span><Search size={14} /><strong>Search results</strong></span>
        <em>{matches.length.toLocaleString()} matches</em>
      </header>
      <ol className="tool-preview-list">
        {matches.slice(0, 30).map((match, index) => (
          <li key={index}>
            <span className="tool-preview-location">
              {match.path ? <strong>{match.path}</strong> : null}
              {typeof match.line === 'number' ? <em>:{match.line}</em> : null}
            </span>
            <code>{match.text}</code>
          </li>
        ))}
      </ol>
      {matches.length > 30 ? <p className="tool-preview-note">Showing first 30 matches.</p> : null}
    </section>
  )
}

function WebPreview({ block, toolName }: { block: ToolResult; toolName: string }) {
  const results = parseWebResults(block)
  if (results.length === 0) return null
  return (
    <section className="tool-preview tool-preview-web" aria-label="Web results">
      <header>
        <span><Globe size={14} /><strong>{toolName}</strong></span>
        <em>{results.length.toLocaleString()} results</em>
      </header>
      <ol className="tool-preview-web-list">
        {results.slice(0, 8).map((result, index) => (
          <li key={index}>
            {result.url ? (
              <a href={result.url} target="_blank" rel="noreferrer">
                <strong>{result.title}</strong>
                <ExternalLink size={12} />
              </a>
            ) : (
              <strong>{result.title}</strong>
            )}
            {result.snippet ? <p>{result.snippet}</p> : null}
            {result.url ? <small>{result.url}</small> : null}
          </li>
        ))}
      </ol>
    </section>
  )
}

function TaskPreview({ block, toolArguments }: PreviewProps) {
  const parsedContent = parseJson(block.content)
  const contentRecord = isRecord(parsedContent) ? parsedContent : {}
  const usage = isRecord(block.metadata.usage) ? block.metadata.usage : isRecord(contentRecord.usage) ? contentRecord.usage : {}
  const prompt = firstString(
    toolArguments.prompt,
    toolArguments.description,
    toolArguments.task,
    block.metadata.prompt,
    contentRecord.prompt,
    contentRecord.description,
    contentRecord.task,
  )
  const status = block.isError ? 'failed' : firstString(block.metadata.status, contentRecord.status) || 'completed'
  const summary = firstString(block.metadata.summary, contentRecord.summary)
  const outputTail = firstString(block.metadata.output_tail, block.metadata.outputTail, contentRecord.output_tail, contentRecord.outputTail)
    || (summary ? null : block.content)
  return (
    <InlineTaskSummary
      className="tool-preview tool-preview-task"
      ariaLabel="Subagent result"
      title={prompt || 'Subagent task'}
      status={status}
      taskId={firstString(block.metadata.task_id, block.metadata.taskId, contentRecord.task_id, contentRecord.taskId, contentRecord.id)}
      subagentType={firstString(toolArguments.subagent_type, toolArguments.subagentType, toolArguments.agent_type, toolArguments.agentType, block.metadata.subagent_type, block.metadata.subagentType, contentRecord.subagent_type, contentRecord.subagentType)}
      model={firstString(toolArguments.model, block.metadata.model, contentRecord.model)}
      durationSeconds={durationSecondsFromMetadata(block.metadata) ?? durationSecondsFromMetadata(contentRecord)}
      inputTokens={numberValue(block.metadata.input_tokens) ?? numberValue(block.metadata.inputTokens) ?? numberValue(usage.input_tokens) ?? numberValue(usage.inputTokens)}
      outputTokens={numberValue(block.metadata.output_tokens) ?? numberValue(block.metadata.outputTokens) ?? numberValue(usage.output_tokens) ?? numberValue(usage.outputTokens)}
      summary={summary}
      error={firstString(block.metadata.error, contentRecord.error)}
      outputTail={outputTail}
      outputFile={firstString(block.metadata.output_file, block.metadata.outputFile, block.metadata.result_path, block.metadata.resultPath, contentRecord.output_file, contentRecord.outputFile, contentRecord.result_path, contentRecord.resultPath)}
    />
  )
}

function ArtifactPreview({ artifacts }: { artifacts: ToolArtifact[] }) {
  return (
    <section className="tool-preview tool-preview-artifacts" aria-label="Tool artifacts">
      <header>
        <span><FileArchive size={14} /><strong>Artifacts</strong></span>
        <em>{artifacts.length.toLocaleString()} item{artifacts.length === 1 ? '' : 's'}</em>
      </header>
      <div className="tool-preview-artifact-list">
        {artifacts.map((artifact, index) => (
          <article className="tool-preview-artifact-row" key={(artifact.href || artifact.path || artifact.name) + '-' + index}>
            <FileText size={14} aria-hidden="true" />
            <span>
              <strong>{artifact.name}</strong>
              <small>{artifactMeta(artifact)}</small>
              {artifact.note ? <em>{artifact.note}</em> : null}
            </span>
            <span className="tool-preview-artifact-actions">
              {artifactCopyText(artifact) ? (
                <CopyButton
                  text={artifactCopyText(artifact) || ''}
                  label={'Copy ' + artifact.name + ' path'}
                  displayLabel="Copy"
                  displayCopiedLabel="Copied"
                  className="tool-preview-artifact-copy"
                />
              ) : null}
              {artifact.href ? <a href={artifact.href} target="_blank" rel="noreferrer">Open</a> : null}
            </span>
          </article>
        ))}
      </div>
    </section>
  )
}

function ImagePreview({ images }: { images: ToolImage[] }) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null)
  return (
    <>
      <section className="tool-preview tool-preview-images" aria-label="Tool image results">
        <header>
          <span><ImageIcon size={14} /><strong>Images</strong></span>
          <em>{images.length.toLocaleString()} image{images.length === 1 ? '' : 's'}</em>
        </header>
        <div className="tool-preview-image-grid">
          {images.map((image, index) => (
            <figure className="tool-preview-image-card" key={image.src + '-' + index}>
              <button type="button" aria-label={'Open image preview ' + image.name} onClick={() => setActiveIndex(index)}>
                <img src={image.src} alt={image.name} loading="lazy" />
              </button>
              <figcaption>
                <strong>{image.name}</strong>
                {image.caption ? <span>{image.caption}</span> : null}
                {image.href ? <a href={image.href} target="_blank" rel="noreferrer">Open source</a> : null}
              </figcaption>
            </figure>
          ))}
        </div>
      </section>
      {activeIndex !== null ? (
        <ToolImageModal images={images} activeIndex={activeIndex} onClose={() => setActiveIndex(null)} onSelect={setActiveIndex} />
      ) : null}
    </>
  )
}

function ToolImageModal({ images, activeIndex, onClose, onSelect }: { images: ToolImage[]; activeIndex: number; onClose: () => void; onSelect: (index: number) => void }) {
  const activeImage = images[activeIndex]

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
      if (images.length <= 1) return
      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        onSelect((activeIndex - 1 + images.length) % images.length)
      }
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        onSelect((activeIndex + 1) % images.length)
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [activeIndex, images.length, onClose, onSelect])

  if (!activeImage) return null

  return (
    <div className="tool-image-backdrop" role="dialog" aria-modal="true" aria-label={activeImage.name}>
      <div className="tool-image-modal">
        <header>
          <div>
            <strong>{activeImage.name}</strong>
            <span>{activeIndex + 1} / {images.length}</span>
          </div>
          <button type="button" aria-label="Close image preview" onClick={onClose}>
            <X aria-hidden="true" size={16} />
          </button>
        </header>
        <div className="tool-image-stage">
          {images.length > 1 ? (
            <button type="button" aria-label="Previous image" onClick={() => onSelect((activeIndex - 1 + images.length) % images.length)}>
              <ChevronLeft aria-hidden="true" size={18} />
            </button>
          ) : null}
          <img src={activeImage.src} alt={activeImage.name} />
          {images.length > 1 ? (
            <button type="button" aria-label="Next image" onClick={() => onSelect((activeIndex + 1) % images.length)}>
              <ChevronRight aria-hidden="true" size={18} />
            </button>
          ) : null}
        </div>
      </div>
    </div>
  )
}

function parseSearchMatches(content: string): SearchMatch[] {
  return content
    .split('\n')
    .map((line) => line.trimEnd())
    .filter(Boolean)
    .map((line) => {
      const match = line.match(/^(.+?):(\d+)(?::\d+)?:\s?(.*)$/)
      if (match) return { path: match[1], line: Number(match[2]), text: match[3] || '' }
      return { text: line }
    })
    .filter((match) => match.text.trim().length > 0)
}

function parseWebResults(block: ToolResult): WebResult[] {
  const results: WebResult[] = []
  const seen = new Set<string>()
  const add = (result: WebResult | null) => {
    if (!result) return
    const key = (result.url || result.title).toLowerCase()
    if (seen.has(key)) return
    seen.add(key)
    results.push(result)
  }

  collectWebResults(parseJson(block.content), add)
  collectWebResults(block.metadata, add, block.content)
  for (const result of webResultsFromMarkdown(block.content)) add(result)
  return results
}

function collectWebResults(value: unknown, add: (result: WebResult | null) => void, fallbackSnippet = '', depth = 0): void {
  if (value == null || depth > 5) return
  if (Array.isArray(value)) {
    for (const item of value) collectWebResults(item, add, fallbackSnippet, depth + 1)
    return
  }
  if (!isRecord(value)) return

  add(webResultFromRecord(value, fallbackSnippet))
  for (const nested of Object.values(value)) collectWebResults(nested, add, fallbackSnippet, depth + 1)
}

function webResultFromRecord(record: Record<string, unknown>, fallbackSnippet = ''): WebResult | null {
  const url = firstString(record.url, record.link, record.href, record.uri)
  const title = firstString(record.title, record.name, record.heading, record.url, record.link, record.href)
  if (!title && !url) return null
  const snippet = firstString(
    record.snippet,
    record.description,
    record.summary,
    record.content,
    record.text,
    record.raw_content,
    record.rawContent,
    record.page_content,
    record.pageContent,
    fallbackSnippet,
  )
  return {
    title: title || url || 'Web result',
    url,
    snippet: snippet ? compactSnippet(snippet) : null,
  }
}

function webResultsFromMarkdown(content: string): WebResult[] {
  const results: WebResult[] = []
  const pattern = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)(?:\s*[-–—:]\s*([^\n]+))?/g
  let match: RegExpExecArray | null
  while ((match = pattern.exec(content)) !== null) {
    results.push({ title: match[1], url: match[2], snippet: match[3] ? compactSnippet(match[3]) : null })
  }
  return results
}

function compactSnippet(value: string): string {
  const compact = value.replace(/\s+/g, ' ').trim()
  return compact.length > 320 ? compact.slice(0, 317).trimEnd() + '...' : compact
}

function parseToolArtifacts(block: ToolResult): ToolArtifact[] {
  const artifacts: ToolArtifact[] = []
  const seen = new Set<string>()
  const add = (artifact: ToolArtifact | null) => {
    if (!artifact) return
    const key = artifact.href || artifact.path || artifact.name
    if (seen.has(key)) return
    seen.add(key)
    artifacts.push(artifact)
  }

  add(spillArtifactFromText(block.content))
  const parsed = parseJson(block.content)
  collectArtifactsFromRecord(parsed, add)
  collectArtifactsFromRecord(block.metadata, add)
  return artifacts.filter((artifact) => !artifact.href || !safeImageSrc(artifact.href))
}

function collectArtifactsFromRecord(value: unknown, add: (artifact: ToolArtifact | null) => void): void {
  if (!isRecord(value)) return
  for (const key of ['artifacts', 'attachments', 'files', 'outputs']) {
    const items = arrayValue(value[key])
    if (!items) continue
    for (const item of items) add(artifactFromRecord(recordOrNull(item)))
  }
  add(artifactFromRecord(pickArtifactRecord(value)))
}

function artifactFromRecord(record: Record<string, unknown> | null): ToolArtifact | null {
  if (!record) return null
  const path = firstString(record.path, record.file_path, record.filePath, record.result_path, record.resultPath, record.output_file, record.outputFile)
  const url = firstString(record.url, record.href, record.uri)
  const href = safeHref(url || path)
  if (!href && !path) return null
  const name = firstString(record.title, record.name, record.filename, record.file_name, record.label) || artifactNameFromPath(path || href || 'artifact')
  return {
    name,
    href,
    path: path || null,
    kind: firstString(record.kind, record.type) || null,
    mimeType: firstString(record.mime_type, record.mimeType, record.media_type, record.mediaType) || null,
    size: numberValue(record.size) ?? numberValue(record.size_bytes) ?? numberValue(record.sizeBytes) ?? numberValue(record.bytes),
    note: firstString(record.caption, record.description, record.summary) || null,
  }
}

function pickArtifactRecord(record: Record<string, unknown>): Record<string, unknown> | null {
  const path = firstString(record.result_path, record.resultPath, record.output_file, record.outputFile, record.artifact_path, record.artifactPath)
  if (!path) return null
  return { path, name: firstString(record.result_name, record.resultName, record.output_name, record.outputName), kind: 'result' }
}

function spillArtifactFromText(content: string): ToolArtifact | null {
  const match = content.match(/output truncated: ([^\]]*?) saved to (\S+)\s+[—-]\s+use read_file to retrieve/i)
  if (!match) return null
  const path = match[2]?.trim()
  if (!path) return null
  return {
    name: artifactNameFromPath(path),
    href: safeHref(path),
    path,
    kind: 'spilled result',
    mimeType: null,
    size: null,
    note: match[1] ? 'Full output: ' + match[1].trim() : 'Full output saved to disk',
  }
}

function artifactCopyText(artifact: ToolArtifact): string | null {
  return artifact.path || artifact.href || null
}

function artifactMeta(artifact: ToolArtifact): string {
  const parts = [artifact.kind, artifact.mimeType, artifact.size != null ? artifact.size.toLocaleString() + ' bytes' : null, artifact.path].filter(Boolean)
  return parts.join(' / ') || 'artifact'
}

function artifactNameFromPath(path: string): string {
  const clean = path.split(/[?#]/, 1)[0] || path
  return clean.split(/[\/]/).filter(Boolean).pop() || 'artifact'
}

function recordOrNull(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null
}

function parseToolImages(block: ToolResult): ToolImage[] {
  const images: ToolImage[] = []
  const seen = new Set<string>()
  const add = (image: ToolImage | null) => {
    if (!image || seen.has(image.src)) return
    seen.add(image.src)
    images.push(image)
  }

  for (const image of imageUrlsFromText(block.content)) add(image)
  collectImages(parseJson(block.content), add)
  collectImages(block.metadata, add)
  return images
}

function collectImages(value: unknown, add: (image: ToolImage | null) => void, depth = 0): void {
  if (depth > 4 || value == null) return
  if (typeof value === 'string') {
    add(imageFromString(value))
    return
  }
  if (Array.isArray(value)) {
    for (const item of value) collectImages(item, add, depth + 1)
    return
  }
  if (!isRecord(value)) return

  add(imageFromRecord(value))
  for (const item of Object.values(value)) collectImages(item, add, depth + 1)
}

function imageFromRecord(record: Record<string, unknown>): ToolImage | null {
  const mimeType = stringValue(record.mime_type) || stringValue(record.mimeType) || stringValue(record.media_type) || stringValue(record.mediaType)
  const src = imageSrcFromRecord(record, mimeType)
  if (!src) return null
  const name = stringValue(record.title)
    || stringValue(record.name)
    || stringValue(record.filename)
    || stringValue(record.file_name)
    || stringValue(record.alt)
    || imageNameFromSrc(src)
  return {
    src,
    name,
    caption: stringValue(record.caption) || stringValue(record.description) || stringValue(record.summary),
    href: safeHref(stringValue(record.href) || stringValue(record.url) || stringValue(record.path) || src),
  }
}

function imageSrcFromRecord(record: Record<string, unknown>, mimeType: string | null): string | null {
  const direct = stringValue(record.src)
    || stringValue(record.url)
    || stringValue(record.uri)
    || stringValue(record.href)
    || stringValue(record.image_url)
    || stringValue(record.imageUrl)
    || stringValue(record.screenshot_url)
    || stringValue(record.screenshotUrl)
    || stringValue(record.path)
    || stringValue(record.file_path)
    || stringValue(record.filePath)
  const safeDirect = direct ? safeImageSrc(direct) : null
  if (safeDirect) return safeDirect

  const data = stringValue(record.data) || stringValue(record.base64) || stringValue(record.content)
  if (!data) return null
  if (data.startsWith('data:')) return safeImageSrc(data)
  if (mimeType?.startsWith('image/')) return 'data:' + mimeType + ';base64,' + data
  return null
}

function imageFromString(value: string): ToolImage | null {
  const src = safeImageSrc(value)
  return src ? { src, name: imageNameFromSrc(src), caption: null, href: safeHref(src) } : null
}

function imageUrlsFromText(content: string): ToolImage[] {
  const images: ToolImage[] = []
  const markdownPattern = /!\[([^\]]*)\]\(([^)]+)\)|\bhttps?:\/\/[^\s<>)\]]+/g
  let match: RegExpExecArray | null
  while ((match = markdownPattern.exec(content)) !== null) {
    const raw = match[2] ?? match[0]
    const src = safeImageSrc(raw)
    if (!src) continue
    images.push({
      src,
      name: match[1] || imageNameFromSrc(src),
      caption: null,
      href: safeHref(src),
    })
  }
  return images
}

function parseJson(content: string): unknown {
  try {
    return JSON.parse(content)
  } catch {
    return null
  }
}

function isReadTool(toolName: string) {
  return ['read', 'read_file', 'view_file', 'open_file'].includes(toolName)
}

function isSearchTool(toolName: string) {
  return ['grep', 'rg', 'search', 'search_files', 'find'].includes(toolName)
}

function isWebTool(toolName: string) {
  return ['web_search', 'search_web', 'web_fetch', 'fetch_url'].includes(toolName)
}

function isTaskTool(toolName: string) {
  return ['task', 'agent', 'spawn_agent', 'delegate_task'].includes(toolName)
}

function isEditTool(toolName: string) {
  return ['write', 'write_file', 'create_file', 'edit', 'edit_file', 'file_edit', 'replace', 'apply_patch'].includes(toolName)
}

function isNotebookTool(toolName: string) {
  return toolName === 'notebook_edit'
}

function isLspTool(toolName: string) {
  return toolName === 'lsp'
}

function isBrowserTool(toolName: string) {
  return ['web_browser', 'browser', 'browser_tool'].includes(toolName)
}

function languageFromPath(path?: string | null): string | null {
  if (!path) return null
  const ext = path.split('.').pop()?.toLowerCase()
  if (!ext || ext === path) return null
  if (ext === 'py') return 'python'
  if (ext === 'js') return 'javascript'
  if (ext === 'ts') return 'typescript'
  if (ext === 'tsx') return 'tsx'
  if (ext === 'jsx') return 'jsx'
  if (ext === 'md') return 'markdown'
  if (ext === 'sh') return 'bash'
  return ext
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value : null
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    const text = stringValue(value)
    if (text) return text
  }
  return null
}

function numberValue(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

function durationSecondsFromMetadata(metadata: Record<string, unknown>): number | null {
  const seconds = numberValue(metadata.duration_seconds) ?? numberValue(metadata.durationSeconds)
  if (seconds != null) return seconds
  const millis = numberValue(metadata.duration_ms) ?? numberValue(metadata.durationMs)
  return millis != null ? millis / 1000 : null
}

function firstStringFromArray(value: unknown): string | null {
  if (!Array.isArray(value)) return null
  for (const item of value) {
    const text = stringValue(item)
    if (text) return text
  }
  return null
}

function numberLabel(prefix: string, value: number | null): string | null {
  return value == null ? null : prefix + ' ' + value.toLocaleString()
}

function parseMarkdownBullets(content: string): string[] {
  return content
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.startsWith('- '))
    .map((line) => line.slice(2).trim())
    .filter(Boolean)
}

function arrayValue(value: unknown): unknown[] | null {
  return Array.isArray(value) ? value : null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function safeImageSrc(src: string): string | null {
  const trimmed = src.trim()
  if (/^data:image\/(?:png|jpe?g|gif|webp|svg\+xml);base64,/i.test(trimmed)) return trimmed
  if (/^(https?:|\/(?!\/)|\.\/|\.\.\/)/i.test(trimmed) && isImageUrl(trimmed)) return trimmed
  return null
}

function safeHref(href: string | null): string | null {
  if (!href) return null
  const trimmed = href.trim()
  return /^(https?:|#|\/(?!\/)|\.\/|\.\.\/)/i.test(trimmed) ? trimmed : null
}

function isImageUrl(value: string): boolean {
  return /\.(?:png|jpe?g|gif|webp|svg)(?:[?#].*)?$/i.test(value) || /^data:image\//i.test(value)
}

function imageNameFromSrc(src: string): string {
  if (src.startsWith('data:')) return 'image'
  const clean = src.split(/[?#]/, 1)[0] || src
  return clean.split('/').filter(Boolean).pop() || 'image'
}
