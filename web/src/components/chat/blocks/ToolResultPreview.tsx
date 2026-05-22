import { Bot, ChevronLeft, ChevronRight, ExternalLink, FileText, Globe, ImageIcon, Search, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { CodeBlock } from './CodeBlock'

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

export function canPreviewToolResult(block: ToolResult): boolean {
  const toolName = (block.toolName || '').toLowerCase()
  if (parseToolImages(block).length > 0) return true
  if (isReadTool(toolName) || isTaskTool(toolName)) return true
  if (isSearchTool(toolName)) return parseSearchMatches(block.content).length > 0
  if (isWebTool(toolName)) return parseWebResults(block.content).length > 0
  return false
}

export function ToolResultPreview({ block, toolArguments = {} }: Props) {
  const toolImages = parseToolImages(block)
  if (toolImages.length > 0) return <ImagePreview images={toolImages} />

  const toolName = (block.toolName || '').toLowerCase()
  if (isReadTool(toolName)) return <ReadFilePreview block={block} toolArguments={toolArguments} />
  if (isSearchTool(toolName)) return <SearchPreview content={block.content} />
  if (isWebTool(toolName)) return <WebPreview content={block.content} toolName={block.toolName || 'web'} />
  if (isTaskTool(toolName)) return <TaskPreview block={block} toolArguments={toolArguments} />
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

function WebPreview({ content, toolName }: { content: string; toolName: string }) {
  const results = parseWebResults(content)
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
  const prompt = stringValue(toolArguments.prompt) || stringValue(toolArguments.description) || stringValue(toolArguments.task) || stringValue(block.metadata.prompt)
  const model = stringValue(toolArguments.model) || stringValue(block.metadata.model)
  const status = block.isError ? 'failed' : stringValue(block.metadata.status) || 'completed'
  return (
    <section className={'tool-preview tool-preview-task tool-preview-task-' + status} aria-label="Subagent result">
      <header>
        <span><Bot size={14} /><strong>{prompt || 'Subagent task'}</strong></span>
        <em>{status}</em>
      </header>
      {model ? <p className="tool-preview-note">Model: {model}</p> : null}
      {block.content ? <pre className="tool-preview-task-output"><code>{block.content}</code></pre> : null}
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

function parseWebResults(content: string): WebResult[] {
  const parsed = parseJson(content)
  const candidates = Array.isArray(parsed)
    ? parsed
    : isRecord(parsed)
      ? arrayValue(parsed.results) || arrayValue(parsed.items) || arrayValue(parsed.search_results) || arrayValue(parsed.data) || []
      : []
  const results: WebResult[] = []
  for (const item of candidates) {
    if (!isRecord(item)) continue
    const title = stringValue(item.title) || stringValue(item.name) || stringValue(item.url)
    if (!title) continue
    results.push({
      title,
      url: stringValue(item.url) || stringValue(item.link),
      snippet: stringValue(item.snippet) || stringValue(item.description) || stringValue(item.content),
    })
  }
  return results
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
