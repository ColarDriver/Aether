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
  preview: string | null
  previewLanguage: string | null
  binary: boolean
}

type NotebookOutput = {
  kind: 'text' | 'error' | 'image'
  label: string
  text?: string
  src?: string
}

type NotebookLifecycleStep = {
  label: string
  value: string
  tone?: 'ok' | 'error' | 'active'
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
  if (isWebTool(toolName)) return <WebPreview block={block} toolArguments={toolArguments} toolName={block.toolName || 'web'} />
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
  const parsedContent = parseJson(block.content)
  const contentRecord = isRecord(parsedContent) ? parsedContent : {}
  const status = firstString(
    block.metadata.execution_status,
    block.metadata.executionStatus,
    block.metadata.status,
    toolArguments.execution_status,
    toolArguments.executionStatus,
    toolArguments.status,
    contentRecord.execution_status,
    contentRecord.executionStatus,
    contentRecord.status,
  )
  const cellRef = firstString(toolArguments.cell_id, block.metadata.cell_id, block.metadata.cellId)
    || numberLabel('cell', numberValue(toolArguments.cell_idx) ?? numberValue(toolArguments.cellIdx))
  const cellType = firstString(toolArguments.cell_type, block.metadata.cell_type, block.metadata.cellType) || 'code'
  const cellCount = numberValue(block.metadata.cell_count) ?? numberValue(block.metadata.cellCount)
  const executionCount = numberValue(block.metadata.execution_count) ?? numberValue(block.metadata.executionCount) ?? numberValue(toolArguments.execution_count) ?? numberValue(toolArguments.executionCount) ?? numberValue(contentRecord.execution_count) ?? numberValue(contentRecord.executionCount)
  const durationSeconds = durationSecondsFromMetadata(block.metadata) ?? durationSecondsFromMetadata(toolArguments) ?? durationSecondsFromMetadata(contentRecord)
  const kernel = firstString(block.metadata.kernel, block.metadata.kernel_name, block.metadata.kernelName, toolArguments.kernel, toolArguments.kernel_name, toolArguments.kernelName, contentRecord.kernel, contentRecord.kernel_name, contentRecord.kernelName)
  const outputsTruncated = booleanValue(block.metadata.outputs_truncated) ?? booleanValue(block.metadata.outputsTruncated) ?? booleanValue(toolArguments.outputs_truncated) ?? booleanValue(toolArguments.outputsTruncated) ?? booleanValue(contentRecord.outputs_truncated) ?? booleanValue(contentRecord.outputsTruncated)
  const outputs = parseNotebookOutputs(block, toolArguments)
  const lifecycle = notebookLifecycleSteps(block, toolArguments, contentRecord)
  const summaryContent = notebookSummaryContent(block.content, outputs.length > 0)
  const stats = [
    { label: 'mode', value: mode },
    status ? { label: 'status', value: status } : null,
    cellRef ? { label: 'cell', value: cellRef } : null,
    cellType ? { label: 'type', value: cellType } : null,
    cellCount != null ? { label: 'cells', value: cellCount.toLocaleString() } : null,
    executionCount != null ? { label: 'exec', value: '#' + executionCount.toLocaleString() } : null,
    durationSeconds != null ? { label: 'duration', value: formatDuration(durationSeconds) } : null,
    kernel ? { label: 'kernel', value: kernel } : null,
    outputs.length > 0 ? { label: 'outputs', value: outputs.length.toLocaleString() } : null,
    outputsTruncated ? { label: 'outputs', value: 'truncated' } : null,
  ].filter((item): item is { label: string; value: string } => Boolean(item))
  return (
    <section className="tool-preview tool-preview-notebook" aria-label="Notebook edit">
      <header>
        <span><BookOpen size={14} /><strong>{path}</strong></span>
        <em>{block.isError ? 'failed' : status || mode}</em>
      </header>
      <PreviewStats stats={stats} />
      <NotebookLifecycle steps={lifecycle} />
      <NotebookCellPreview block={block} toolArguments={toolArguments} cellType={cellType} mode={mode} />
      <NotebookOutputsPreview outputs={outputs} />
      {summaryContent ? <pre className="tool-preview-summary"><code>{summaryContent}</code></pre> : null}
    </section>
  )
}

function NotebookLifecycle({ steps }: { steps: NotebookLifecycleStep[] }) {
  if (steps.length === 0) return null
  return (
    <ol className="notebook-lifecycle" aria-label="Notebook lifecycle">
      {steps.map((step) => (
        <li className={step.tone ? 'notebook-lifecycle-' + step.tone : undefined} key={step.label}>
          <strong>{step.label}</strong>
          <span>{step.value}</span>
        </li>
      ))}
    </ol>
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

function NotebookOutputsPreview({ outputs }: { outputs: NotebookOutput[] }) {
  if (outputs.length === 0) return null
  return (
    <div className="notebook-output-preview" aria-label="Notebook outputs">
      <header>
        <strong>Cell outputs</strong>
        <span>{outputs.length.toLocaleString()} item{outputs.length === 1 ? '' : 's'}</span>
      </header>
      <div className="notebook-output-list">
        {outputs.map((output, index) => (
          <article className={'notebook-output-item notebook-output-' + output.kind} key={output.label + '-' + index}>
            <header>
              <strong>{output.label}</strong>
              <span>{output.kind}</span>
            </header>
            {output.kind === 'image' && output.src ? (
              <img src={output.src} alt={output.label} loading="lazy" />
            ) : output.text ? (
              <pre><code>{output.text}</code></pre>
            ) : null}
          </article>
        ))}
      </div>
    </div>
  )
}

function notebookLifecycleSteps(
  block: ToolResult,
  toolArguments: Record<string, unknown>,
  contentRecord: Record<string, unknown>,
): NotebookLifecycleStep[] {
  const timingRecords = notebookLifecycleTimingRecords(block.metadata, toolArguments, contentRecord)
  const queued = firstRecordDisplayValue(timingRecords, ['queued_at', 'queuedAt', 'queue_time', 'queueTime'])
  const started = firstRecordDisplayValue(timingRecords, ['started_at', 'startedAt', 'start_time', 'startTime', 'execution_started_at', 'executionStartedAt'])
  const finished = firstRecordDisplayValue(timingRecords, ['finished_at', 'finishedAt', 'completed_at', 'completedAt', 'execution_finished_at', 'executionFinishedAt'])
  const state = firstRecordString(timingRecords, ['lifecycle_state', 'lifecycleState', 'execution_state', 'executionState', 'state', 'status'])
  const durationSeconds = firstDurationSecondsFromRecords(timingRecords)
  const steps: NotebookLifecycleStep[] = []
  if (queued) steps.push({ label: 'queued', value: queued })
  if (started) steps.push({ label: 'started', value: started, tone: finished ? undefined : 'active' })
  if (finished) steps.push({ label: block.isError ? 'failed' : 'finished', value: finished, tone: block.isError ? 'error' : 'ok' })
  if (!finished && state) steps.push({ label: 'state', value: state, tone: block.isError ? 'error' : stateTone(state) })

  for (const step of notebookLifecycleEventSteps(block, notebookLifecycleEventSources(block.metadata, toolArguments, contentRecord))) {
    if (steps.some((existing) => existing.label === step.label && existing.value === step.value)) continue
    steps.push(step)
  }

  if ((steps.length > 0 || durationSeconds != null) && durationSeconds != null && !steps.some((step) => step.label === 'duration')) {
    steps.push({ label: 'duration', value: formatDuration(durationSeconds) })
  }
  return steps
}

function notebookLifecycleTimingRecords(...records: Record<string, unknown>[]): Record<string, unknown>[] {
  const result: Record<string, unknown>[] = []
  const seen = new Set<Record<string, unknown>>()
  const add = (record: Record<string, unknown>) => {
    if (seen.has(record)) return
    seen.add(record)
    result.push(record)
  }
  for (const record of records) {
    add(record)
    for (const key of ['lifecycle', 'execution', 'timing', 'timestamps']) {
      const nested = recordOrNull(record[key])
      if (nested) add(nested)
    }
  }
  return result
}

function firstRecordDisplayValue(records: Record<string, unknown>[], keys: string[]): string | null {
  for (const record of records) {
    for (const key of keys) {
      const value = firstDisplayValue(record[key])
      if (value) return value
    }
  }
  return null
}

function firstRecordString(records: Record<string, unknown>[], keys: string[]): string | null {
  for (const record of records) {
    for (const key of keys) {
      const value = firstString(record[key])
      if (value) return value
    }
  }
  return null
}

function firstDurationSecondsFromRecords(records: Record<string, unknown>[]): number | null {
  for (const record of records) {
    const value = durationSecondsFromMetadata(record)
    if (value != null) return value
  }
  return null
}

function notebookLifecycleEventSources(...records: Record<string, unknown>[]): unknown[] {
  const sources: unknown[] = []
  for (const record of records) {
    for (const key of ['lifecycle_events', 'lifecycleEvents', 'execution_events', 'executionEvents', 'events', 'timeline', 'steps']) {
      const value = record[key]
      if (Array.isArray(value)) sources.push(value)
    }
    for (const key of ['lifecycle', 'execution', 'timing']) {
      const nested = recordOrNull(record[key])
      if (!nested) continue
      for (const nestedKey of ['events', 'timeline', 'steps', 'lifecycle_events', 'lifecycleEvents']) {
        const value = nested[nestedKey]
        if (Array.isArray(value)) sources.push(value)
      }
    }
  }
  return sources
}

function notebookLifecycleEventSteps(block: ToolResult, sources: unknown[]): NotebookLifecycleStep[] {
  const steps: NotebookLifecycleStep[] = []
  for (const source of sources) {
    if (!Array.isArray(source)) continue
    for (const item of source) {
      const step = notebookLifecycleEventStep(block, item)
      if (step) steps.push(step)
    }
  }
  return steps
}

function notebookLifecycleEventStep(block: ToolResult, value: unknown): NotebookLifecycleStep | null {
  if (!isRecord(value)) return null
  const rawLabel = firstString(value.label, value.name, value.phase, value.event, value.type, value.state, value.status)
  if (!rawLabel) return null
  const label = normalizeLifecycleLabel(rawLabel)
  const explicitValue = firstDisplayValue(
    value.value,
    value.at,
    value.timestamp,
    value.ts,
    value.time,
    value.started_at,
    value.startedAt,
    value.finished_at,
    value.finishedAt,
    value.completed_at,
    value.completedAt,
  )
  const durationSeconds = durationSecondsFromMetadata(value)
  const status = firstString(value.status, value.state)
  const detail = firstString(value.message, value.detail, value.description)
  const fallbackValue = status && normalizeLifecycleLabel(status) !== label ? status : detail
  const displayValue = explicitValue || (durationSeconds != null ? formatDuration(durationSeconds) : null) || fallbackValue
  if (!displayValue) return null
  const toneText = [rawLabel, status, detail].filter(Boolean).join(' ')
  return { label, value: displayValue, tone: block.isError ? 'error' : stateTone(toneText) }
}

function normalizeLifecycleLabel(value: string): string {
  return value.trim().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').toLowerCase()
}


function stateTone(value: string): NotebookLifecycleStep['tone'] {
  const normalized = value.toLowerCase()
  if (['ok', 'done', 'success', 'succeeded', 'completed', 'finished'].some((item) => normalized.includes(item))) return 'ok'
  if (['error', 'failed', 'failure', 'cancelled', 'timeout'].some((item) => normalized.includes(item))) return 'error'
  if (['running', 'executing', 'started', 'pending', 'queued'].some((item) => normalized.includes(item))) return 'active'
  return undefined
}

function parseNotebookOutputs(block: ToolResult, toolArguments: Record<string, unknown>): NotebookOutput[] {
  const parsedContent = parseJson(block.content)
  const contentRecord = isRecord(parsedContent) ? parsedContent : {}
  const candidates = [
    block.metadata.outputs,
    block.metadata.cell_outputs,
    block.metadata.cellOutputs,
    toolArguments.outputs,
    toolArguments.cell_outputs,
    toolArguments.cellOutputs,
    contentRecord.outputs,
    contentRecord.cell_outputs,
    contentRecord.cellOutputs,
  ]
  const outputs: NotebookOutput[] = []
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) {
      for (const item of candidate) {
        const output = notebookOutputFromUnknown(item)
        if (output) outputs.push(output)
      }
    }
  }
  return outputs.slice(0, 12)
}

function notebookOutputFromUnknown(value: unknown): NotebookOutput | null {
  if (!isRecord(value)) return null
  const outputType = firstString(value.output_type, value.outputType, value.type, value.name) || 'output'
  const normalizedType = outputType.toLowerCase()
  if (normalizedType === 'error' || value.ename || value.evalue || value.traceback) {
    const label = firstString(value.ename, value.name) || 'error'
    const traceback = Array.isArray(value.traceback) ? value.traceback.map(notebookDataText).filter(Boolean).join('\\n') : null
    const text = traceback || [firstString(value.ename), firstString(value.evalue)].filter(Boolean).join(': ') || notebookDataText(value.text)
    return text ? { kind: 'error', label, text } : null
  }

  const data = isRecord(value.data) ? value.data : null
  const image = notebookImageFromData(data, value)
  if (image) return image

  const text = notebookDataText(data?.['text/plain'])
    || notebookDataText(data?.['text/markdown'])
    || notebookDataText(value.text)
    || notebookDataText(value.content)
    || notebookDataText(value.output)
  if (!text) return null
  const label = normalizedType === 'stream' ? firstString(value.name) || 'stream' : outputType
  return { kind: 'text', label, text }
}

function notebookImageFromData(data: Record<string, unknown> | null, record: Record<string, unknown>): NotebookOutput | null {
  const direct = firstString(record.src, record.url, record.image_url, record.imageUrl)
  const directSrc = direct ? safeImageSrc(direct) : null
  if (directSrc) return { kind: 'image', label: firstString(record.name, record.title) || imageNameFromSrc(directSrc), src: directSrc }
  if (!data) return null
  for (const mimeType of ['image/png', 'image/jpeg', 'image/webp', 'image/gif', 'image/svg+xml']) {
    const raw = notebookDataText(data[mimeType])
    if (!raw) continue
    const src = raw.startsWith('data:') ? safeImageSrc(raw) : safeImageSrc('data:' + mimeType + ';base64,' + raw)
    if (src) return { kind: 'image', label: mimeType, src }
  }
  return null
}

function notebookDataText(value: unknown): string | null {
  if (typeof value === 'string') return value.trim() ? value : null
  if (Array.isArray(value)) {
    const text = value.map((item) => typeof item === 'string' ? item : '').join('')
    return text.trim() ? text : null
  }
  return null
}

function notebookSummaryContent(content: string, hasOutputs: boolean): string {
  if (!content) return ''
  if (!hasOutputs) return content
  const parsed = parseJson(content)
  if (!isRecord(parsed)) return content
  if (Array.isArray(parsed.outputs) || Array.isArray(parsed.cell_outputs) || Array.isArray(parsed.cellOutputs)) {
    return firstString(parsed.summary, parsed.message, parsed.status) || ''
  }
  return content
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
  const screenshotPath = firstString(block.metadata.screenshot_path, block.metadata.screenshotPath, block.metadata.path)
  const screenshot = browserScreenshotImage(block, screenshotPath)
  const structuredImages = screenshot || screenshotPath ? [] : parseToolImages(block)
  const structuredArtifacts = parseToolArtifacts(block)
  const hasVisualOrArtifact = Boolean(screenshot || screenshotPath || structuredImages.length > 0 || structuredArtifacts.length > 0)
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
      {screenshot ? (
        <figure className="tool-preview-browser-shot">
          <img src={screenshot.src} alt={screenshot.name} loading="lazy" />
          <figcaption>
            <span>
              <strong>{screenshot.name}</strong>
              {screenshotPath ? <code>{screenshotPath}</code> : null}
            </span>
            {screenshot.href ? <a href={screenshot.href} target="_blank" rel="noreferrer">Open</a> : null}
          </figcaption>
        </figure>
      ) : structuredImages.length > 0 ? (
        <BrowserImageGallery images={structuredImages} />
      ) : screenshotPath ? (
        <div className="tool-preview-artifact">
          <ImageIcon size={14} aria-hidden="true" />
          <span>
            <strong>Screenshot saved</strong>
            <code>{screenshotPath}</code>
          </span>
        </div>
      ) : null}
      {structuredArtifacts.length > 0 ? <BrowserArtifactList artifacts={structuredArtifacts} /> : null}
      {!hasVisualOrArtifact && block.content ? (
        <pre className="tool-preview-summary"><code>{block.content}</code></pre>
      ) : null}
    </section>
  )
}

function BrowserImageGallery({ images }: { images: ToolImage[] }) {
  return (
    <div className="tool-preview-image-grid" aria-label="Browser images">
      {images.map((image, index) => (
        <figure className="tool-preview-image-card" key={image.src + '-' + index}>
          <img src={image.src} alt={image.name} loading="lazy" />
          <figcaption>
            <strong>{image.name}</strong>
            {image.caption ? <span>{image.caption}</span> : null}
            {image.href ? <a href={image.href} target="_blank" rel="noreferrer">Open</a> : null}
          </figcaption>
        </figure>
      ))}
    </div>
  )
}

function BrowserArtifactList({ artifacts }: { artifacts: ToolArtifact[] }) {
  return (
    <div className="tool-preview-artifact-list" aria-label="Browser artifacts">
      {artifacts.map((artifact, index) => {
        const Icon = artifactIcon(artifact)
        return (
          <article className={'tool-preview-artifact-row' + (artifact.binary ? ' tool-preview-artifact-binary' : '')} key={(artifact.href || artifact.path || artifact.name) + '-' + index}>
            <Icon size={14} aria-hidden="true" />
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
              {artifact.preview ? (
                <CopyButton
                  text={artifact.preview}
                  label={'Copy ' + artifact.name + ' contents'}
                  displayLabel="Copy contents"
                  displayCopiedLabel="Copied"
                  className="tool-preview-artifact-copy"
                />
              ) : null}
              {artifact.href ? <a href={artifact.href} target="_blank" rel="noreferrer">Open</a> : null}
            </span>
            {artifact.preview ? (
              <pre className="tool-preview-artifact-inline" aria-label={'Preview ' + artifact.name} data-language={artifact.previewLanguage || undefined}>
                <code>{artifact.preview}</code>
              </pre>
            ) : artifact.binary ? (
              <p className="tool-preview-artifact-unavailable">Binary preview unavailable. Copy the path or open the linked artifact if a URL is provided.</p>
            ) : null}
          </article>
        )
      })}
    </div>
  )
}

function browserScreenshotImage(block: ToolResult, screenshotPath: string | null): ToolImage | null {
  const direct = firstString(
    block.metadata.screenshot_url,
    block.metadata.screenshotUrl,
    block.metadata.screenshot_src,
    block.metadata.screenshotSrc,
    block.metadata.image_url,
    block.metadata.imageUrl,
    block.metadata.url && isImageUrl(String(block.metadata.url)) ? block.metadata.url : null,
  )
  const src = safeImageSrc(direct || '')
  if (!src) return null
  const name = firstString(block.metadata.screenshot_name, block.metadata.screenshotName, block.metadata.title) || imageNameFromSrc(src)
  return {
    src,
    name,
    caption: null,
    href: safeHref(src) || safeHref(screenshotPath),
  }
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

function WebPreview({ block, toolArguments, toolName }: { block: ToolResult; toolArguments: Record<string, unknown>; toolName: string }) {
  const results = parseWebResults(block)
  if (results.length === 0) return null
  const hosted = isRecord(block.metadata.hosted_web_search) ? block.metadata.hosted_web_search : {}
  const hostedSources = arrayValue(hosted.sources)
  const provider = firstString(block.metadata.provider, hosted.provider)
  const query = firstString(toolArguments.query, toolArguments.q, block.metadata.query, hosted.query, hostedWebSearchQuery(hosted))
  const sourceCount = numberValue(block.metadata.source_count)
    ?? numberValue(block.metadata.sourceCount)
    ?? numberValue(hosted.source_count)
    ?? numberValue(hosted.sourceCount)
    ?? (hostedSources ? hostedSources.length : null)
  const stats = [
    provider ? { label: 'provider', value: provider } : null,
    query ? { label: 'query', value: query } : null,
    sourceCount != null ? { label: 'sources', value: sourceCount.toLocaleString() } : null,
  ].filter((item): item is { label: string; value: string } => Boolean(item))
  return (
    <section className="tool-preview tool-preview-web" aria-label="Web results">
      <header>
        <span><Globe size={14} /><strong>{toolName}</strong></span>
        <em>{results.length.toLocaleString()} results</em>
      </header>
      {stats.length > 0 ? <PreviewStats stats={stats} /> : null}
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

function hostedWebSearchQuery(hosted: Record<string, unknown>): string | null {
  const calls = arrayValue(hosted.calls)
  if (!calls) return null
  for (const call of calls) {
    if (!isRecord(call)) continue
    const direct = firstString(call.query, call.q)
    if (direct) return direct
    const input = isRecord(call.input) ? firstString(call.input.query, call.input.q) : null
    if (input) return input
    const action = isRecord(call.action) ? firstString(call.action.query, call.action.q) : null
    if (action) return action
  }
  return null
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
        {artifacts.map((artifact, index) => {
          const Icon = artifactIcon(artifact)
          return (
          <article className={'tool-preview-artifact-row' + (artifact.binary ? ' tool-preview-artifact-binary' : '')} key={(artifact.href || artifact.path || artifact.name) + '-' + index}>
            <Icon size={14} aria-hidden="true" />
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
              {artifact.preview ? (
                <CopyButton
                  text={artifact.preview}
                  label={'Copy ' + artifact.name + ' contents'}
                  displayLabel="Copy contents"
                  displayCopiedLabel="Copied"
                  className="tool-preview-artifact-copy"
                />
              ) : null}
              {artifact.href ? <a href={artifact.href} target="_blank" rel="noreferrer">Open</a> : null}
            </span>
            {artifact.preview ? (
              <pre className="tool-preview-artifact-inline" aria-label={'Preview ' + artifact.name} data-language={artifact.previewLanguage || undefined}>
                <code>{artifact.preview}</code>
              </pre>
            ) : artifact.binary ? (
              <p className="tool-preview-artifact-unavailable">Binary preview unavailable. Copy the path or open the linked artifact if a URL is provided.</p>
            ) : null}
          </article>
          )
        })}
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
  const metadataFallback = ['web_fetch', 'fetch_url'].includes((block.toolName || '').toLowerCase()) ? block.content : ''
  collectWebResults(block.metadata, add, metadataFallback)
  for (const result of webResultsFromMarkdown(block.content)) add(result)
  return results
}

function collectWebResults(value: unknown, add: (result: WebResult | null) => void, fallbackSnippet = '', depth = 0): void {
  if (value == null || depth > 5) return
  if (typeof value === 'string') {
    add(webResultFromString(value))
    return
  }
  if (Array.isArray(value)) {
    for (const item of value) collectWebResults(item, add, fallbackSnippet, depth + 1)
    return
  }
  if (!isRecord(value)) return

  add(webResultFromRecord(value, fallbackSnippet))
  for (const nested of Object.values(value)) collectWebResults(nested, add, fallbackSnippet, depth + 1)
}

function webResultFromRecord(record: Record<string, unknown>, fallbackSnippet = ''): WebResult | null {
  const rawUrl = firstString(
    record.url,
    record.link,
    record.href,
    record.uri,
    record.source_url,
    record.sourceUrl,
    record.page_url,
    record.pageUrl,
  )
  const url = safeHref(rawUrl)
  const title = firstString(
    record.title,
    record.name,
    record.heading,
    record.label,
    record.source,
    record.site_name,
    record.siteName,
    record.url,
    record.link,
    record.href,
  )
  if (!title && !url) return null
  const snippet = webSnippetFromRecord(record, fallbackSnippet)
  return {
    title: title || url || 'Web result',
    url,
    snippet: snippet ? compactSnippet(snippet) : null,
  }
}

function webSnippetFromRecord(record: Record<string, unknown>, fallbackSnippet = ''): string | null {
  const direct = firstString(
    record.snippet,
    record.description,
    record.summary,
    record.content,
    record.text,
    record.raw_content,
    record.rawContent,
    record.page_content,
    record.pageContent,
    record.answer,
  )
  if (direct) return direct
  return firstJoinedStringArray(record.extra_snippets)
    || firstJoinedStringArray(record.extraSnippets)
    || firstJoinedStringArray(record.snippets)
    || firstJoinedStringArray(record.highlights)
    || firstJoinedStringArray(record.citations)
    || fallbackSnippet
    || null
}

function firstJoinedStringArray(value: unknown): string | null {
  if (!Array.isArray(value)) return null
  const parts = value
    .map((item) => {
      if (typeof item === 'string') return item.trim()
      if (isRecord(item)) return firstString(item.snippet, item.text, item.content, item.summary, item.description) || ''
      return ''
    })
    .filter(Boolean)
  return parts.length > 0 ? parts.join(' ') : null
}

function webResultFromString(value: string): WebResult | null {
  const match = value.match(/\bhttps?:\/\/[^\s<>)\]]+/)
  if (!match) return null
  const url = trimUrl(match[0])
  return { title: url, url, snippet: null }
}

function webResultsFromMarkdown(content: string): WebResult[] {
  const results: WebResult[] = []
  const pattern = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)(?:\s*[-–—:]\s*([^\n]+))?/g
  let match: RegExpExecArray | null
  while ((match = pattern.exec(content)) !== null) {
    results.push({ title: match[1], url: match[2], snippet: match[3] ? compactSnippet(match[3]) : null })
  }

  const lines = content.split('\n')
  for (let index = 0; index < lines.length; index += 1) {
    const titleMatch = lines[index]?.match(/^\s*\d+[.)]\s+\*\*(.+?)\*\*\s*$/)
    if (!titleMatch) continue
    let url: string | null = null
    const snippetLines: string[] = []
    for (let nextIndex = index + 1; nextIndex < lines.length; nextIndex += 1) {
      const nextLine = (lines[nextIndex] || '').trim()
      if (/^\d+[.)]\s+\*\*.+?\*\*\s*$/.test(nextLine)) break
      if (!nextLine) {
        if (url || snippetLines.length > 0) break
        continue
      }
      const urlMatch = nextLine.match(/https?:\/\/[^\s)]+/)
      if (!url && urlMatch) {
        url = trimUrl(urlMatch[0])
        continue
      }
      if (!nextLine.startsWith('#')) snippetLines.push(nextLine.replace(/^[-*]\s+/, ''))
    }
    if (url || snippetLines.length > 0) {
      results.push({
        title: titleMatch[1] || url || 'Web result',
        url,
        snippet: snippetLines.length > 0 ? compactSnippet(snippetLines.join(' ')) : null,
      })
    }
  }

  return results
}

function trimUrl(value: string): string {
  return value.replace(/[.,;:]+$/, '')
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
  const explicitHref = firstString(record.url, record.href, record.uri, record.download_url, record.downloadUrl, record.preview_url, record.previewUrl)
  const href = safeHref(explicitHref)
  if (!href && !path) return null
  const name = firstString(record.title, record.name, record.filename, record.file_name, record.label) || artifactNameFromPath(path || href || 'artifact')
  const mimeType = firstString(record.mime_type, record.mimeType, record.media_type, record.mediaType) || null
  const kind = firstString(record.kind, record.type) || null
  const binary = isBinaryArtifact(record, mimeType, kind, path || href || name)
  return {
    name,
    href,
    path: path || null,
    kind,
    mimeType,
    size: numberValue(record.size) ?? numberValue(record.size_bytes) ?? numberValue(record.sizeBytes) ?? numberValue(record.bytes),
    note: firstString(record.caption, record.description, record.summary) || null,
    preview: binary ? null : artifactPreviewText(record),
    previewLanguage: binary ? null : artifactPreviewLanguage(record, path || href || name),
    binary,
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
    href: null,
    path,
    kind: 'spilled result',
    mimeType: null,
    size: null,
    note: match[1] ? 'Full output: ' + match[1].trim() : 'Full output saved to disk',
    preview: null,
    previewLanguage: null,
    binary: false,
  }
}

function artifactPreviewText(record: Record<string, unknown>): string | null {
  const direct = firstString(record.preview, record.content, record.text, record.output, record.body)
  if (direct) return truncateArtifactPreview(direct)
  for (const key of ['preview', 'content', 'text', 'output', 'body']) {
    const value = record[key]
    if (value == null || typeof value === 'string') continue
    if (typeof value === 'number' || typeof value === 'boolean' || Array.isArray(value) || isRecord(value)) {
      return truncateArtifactPreview(JSON.stringify(value, null, 2))
    }
  }
  return null
}

function artifactPreviewLanguage(record: Record<string, unknown>, fallbackPath: string): string | null {
  return firstString(record.language, record.lang) || languageFromPath(fallbackPath) || null
}

function truncateArtifactPreview(value: string): string {
  const trimmed = value.trim()
  if (trimmed.length <= 2400) return trimmed
  return trimmed.slice(0, 2400).trimEnd() + '\n... preview truncated ...'
}

function artifactCopyText(artifact: ToolArtifact): string | null {
  return artifact.path || artifact.href || null
}

function artifactMeta(artifact: ToolArtifact): string {
  const parts = [artifact.kind, artifact.binary ? 'binary' : null, artifact.mimeType, artifact.size != null ? artifact.size.toLocaleString() + ' bytes' : null, artifact.path].filter(Boolean)
  return parts.join(' / ') || 'artifact'
}

function artifactIcon(artifact: ToolArtifact) {
  if (artifact.binary || artifact.mimeType?.includes('zip') || artifact.kind?.toLowerCase().includes('archive')) return FileArchive
  if (artifact.previewLanguage || artifact.mimeType?.includes('json') || artifact.mimeType?.startsWith('text/')) return FileCode2
  return FileText
}

function isBinaryArtifact(record: Record<string, unknown>, mimeType: string | null, kind: string | null, fallbackPath: string): boolean {
  const explicit = booleanValue(record.binary) ?? booleanValue(record.is_binary) ?? booleanValue(record.isBinary)
  if (explicit != null) return explicit
  const normalizedMime = (mimeType || '').toLowerCase()
  if (normalizedMime.startsWith('text/') || normalizedMime.includes('json') || normalizedMime.includes('xml') || normalizedMime.includes('yaml') || normalizedMime.includes('toml')) return false
  if (normalizedMime === 'application/octet-stream' || normalizedMime.includes('zip') || normalizedMime.includes('pdf') || normalizedMime.startsWith('image/')) return true
  const normalizedKind = (kind || '').toLowerCase()
  if (normalizedKind.includes('binary') || normalizedKind.includes('archive')) return true
  const language = languageFromPath(fallbackPath)
  return language == null && /\.(?:bin|pdf|zip|tar|gz|tgz|sqlite|db|parquet|png|jpe?g|gif|webp)$/i.test(fallbackPath)
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

function firstDisplayValue(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value.trim()
    if (typeof value === 'number' && Number.isFinite(value)) return formatTimestampValue(value)
  }
  return null
}

function formatTimestampValue(value: number): string {
  if (value > 1_000_000_000) {
    const millis = value > 10_000_000_000 ? value : value * 1000
    return new Date(millis).toISOString().replace('T', ' ').replace(/\.\d{3}Z$/, ' UTC')
  }
  return value.toLocaleString()
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

function booleanValue(value: unknown): boolean | null {
  if (typeof value === 'boolean') return value
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase()
    if (['true', 'yes', '1'].includes(normalized)) return true
    if (['false', 'no', '0'].includes(normalized)) return false
  }
  return null
}

function durationSecondsFromMetadata(metadata: Record<string, unknown>): number | null {
  const seconds = numberValue(metadata.duration_seconds) ?? numberValue(metadata.durationSeconds)
  if (seconds != null) return seconds
  const millis = numberValue(metadata.duration_ms) ?? numberValue(metadata.durationMs)
  return millis != null ? millis / 1000 : null
}

function formatDuration(seconds: number): string {
  if (seconds < 10) return seconds.toFixed(1) + 's'
  if (seconds < 60) return Math.round(seconds) + 's'
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  return minutes + 'm ' + rest + 's'
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
