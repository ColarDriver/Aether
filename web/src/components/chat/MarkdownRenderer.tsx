import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { MathRenderer } from './MathRenderer'
import { MermaidRenderer } from './MermaidRenderer'
import { CodeBlock } from './blocks/CodeBlock'

type Props = {
  text: string
  streaming?: boolean
}

type MarkdownBlock =
  | { type: 'code'; language: string; code: string }
  | { type: 'math'; source: string; display: boolean }
  | { type: 'heading'; level: 1 | 2 | 3 | 4; content: string }
  | { type: 'hr' }
  | { type: 'table'; block: string }
  | { type: 'blockquote'; content: string }
  | { type: 'list'; ordered: boolean; items: ListItem[] }
  | { type: 'paragraph'; content: string }

type ListItem = {
  content: string
  checked?: boolean
}

type MarkdownImage = {
  src: string
  alt: string
}

type ImageContext = {
  images: MarkdownImage[]
  openImage: (index: number) => void
}

const MERMAID_DIAGRAM_START = /^(---|graph|flowchart|sequenceDiagram|classDiagram|classDiagram-v2|stateDiagram|stateDiagram-v2|erDiagram|journey|gantt|pie|gitGraph|mindmap|timeline|quadrantChart|requirementDiagram|C4Context|C4Container|C4Component|C4Dynamic|c4Context|c4Container|c4Component|c4Dynamic|sankey-beta|block-beta|packet-beta|xychart-beta|kanban|architecture-beta)\b/i
const PLAIN_CODE_LANGUAGES = new Set(['', 'text', 'plain', 'plaintext'])

export function MarkdownRenderer({ text, streaming = false }: Props) {
  const blocks = splitMarkdownBlocks(text)
  const images = useMemo(() => extractMarkdownImages(text), [text])
  const [activeImageIndex, setActiveImageIndex] = useState<number | null>(null)
  const imageContext = useMemo<ImageContext>(() => ({ images, openImage: setActiveImageIndex }), [images])
  return (
    <div className="markdown-renderer">
      {blocks.map((block, index) => renderBlock(block, index, streaming && index === blocks.length - 1, imageContext))}
      {blocks.length === 0 && streaming ? <span className="streaming-caret" /> : null}
      {activeImageIndex !== null ? (
        <MarkdownImagePreview images={images} activeIndex={activeImageIndex} onClose={() => setActiveImageIndex(null)} onSelect={setActiveImageIndex} />
      ) : null}
    </div>
  )
}

function renderBlock(block: MarkdownBlock, index: number, streaming = false, imageContext?: ImageContext) {
  const caret = streaming ? <StreamingCaret /> : null
  if (block.type === 'heading') {
    const content = <>{renderInline(block.content, imageContext)}{caret}</>
    if (block.level === 1) return <h1 key={index}>{content}</h1>
    if (block.level === 2) return <h2 key={index}>{content}</h2>
    if (block.level === 3) return <h3 key={index}>{content}</h3>
    return <h4 key={index}>{content}</h4>
  }
  if (block.type === 'code') {
    return shouldRenderAsMermaid(block.language, block.code)
      ? <MermaidRenderer code={block.code} key={index} />
      : <CodeBlock code={block.code} key={index} language={block.language} />
  }
  if (block.type === 'math') return <MathRenderer display={block.display} key={index} source={block.source} />
  if (block.type === 'table') return <MarkdownTable block={block.block} imageContext={imageContext} key={index} />
  if (block.type === 'hr') return <hr className="markdown-hr" key={index} />
  if (block.type === 'blockquote') {
    return (
      <blockquote key={index}>
        {renderInline(block.content, imageContext)}
        {caret}
      </blockquote>
    )
  }
  if (block.type === 'list') {
    const Tag = block.ordered ? 'ol' : 'ul'
    return (
      <Tag className={hasTasks(block.items) ? 'markdown-task-list' : undefined} key={index}>
        {block.items.map((item, itemIndex) => (
          <li className={item.checked != null ? 'markdown-task-item' : undefined} key={itemIndex}>
            {item.checked != null ? <input checked={item.checked} readOnly type="checkbox" /> : null}
            <span>{renderInline(item.content, imageContext)}{streaming && itemIndex === block.items.length - 1 ? caret : null}</span>
          </li>
        ))}
      </Tag>
    )
  }
  return <p key={index}>{renderInline(block.content, imageContext)}{caret}</p>
}

function StreamingCaret() {
  return <span aria-hidden="true" className="streaming-caret streaming-caret-inline" />
}

function splitMarkdownBlocks(text: string): MarkdownBlock[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const fence = String.fromCharCode(96, 96, 96)
  const blocks: MarkdownBlock[] = []
  let index = 0

  while (index < lines.length) {
    const line = lines[index] ?? ''
    if (!line.trim()) {
      index += 1
      continue
    }
    if (line.startsWith(fence)) {
      const language = line.slice(fence.length).trim()
      const codeLines: string[] = []
      index += 1
      while (index < lines.length && !(lines[index] ?? '').startsWith(fence)) {
        codeLines.push(lines[index] ?? '')
        index += 1
      }
      if (index < lines.length) index += 1
      blocks.push({ type: 'code', language, code: codeLines.join('\n').trimEnd() })
      continue
    }

    const displayMathStart = parseDisplayMathStart(line)
    if (displayMathStart) {
      const mathLines: string[] = []
      const openingRemainder = displayMathStart.remainder
      const sameLineEnd = openingRemainder.indexOf(displayMathStart.close)
      if (sameLineEnd >= 0) {
        mathLines.push(openingRemainder.slice(0, sameLineEnd))
        index += 1
      } else {
        if (openingRemainder) mathLines.push(openingRemainder)
        index += 1
        while (index < lines.length) {
          const nextLine = lines[index] ?? ''
          const closeIndex = nextLine.indexOf(displayMathStart.close)
          if (closeIndex >= 0) {
            mathLines.push(nextLine.slice(0, closeIndex))
            index += 1
            break
          }
          mathLines.push(nextLine)
          index += 1
        }
      }
      blocks.push({ type: 'math', display: true, source: mathLines.join('\n').trim() })
      continue
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/)
    if (heading?.[1] && heading[2]) {
      blocks.push({ type: 'heading', level: heading[1].length as 1 | 2 | 3 | 4, content: heading[2] })
      index += 1
      continue
    }

    if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      blocks.push({ type: 'hr' })
      index += 1
      continue
    }

    if (line.trim().startsWith('|') && index + 1 < lines.length) {
      const tableLines: string[] = []
      while (index < lines.length && isTableLine(lines[index] ?? '')) {
        tableLines.push(lines[index] ?? '')
        index += 1
      }
      if (isMarkdownTable(tableLines.join('\n'))) {
        blocks.push({ type: 'table', block: tableLines.join('\n') })
      } else {
        blocks.push({ type: 'paragraph', content: tableLines.join('\n') })
      }
      continue
    }

    if (line.trim().startsWith('>')) {
      const quoteLines: string[] = []
      while (index < lines.length && (lines[index] ?? '').trim().startsWith('>')) {
        quoteLines.push((lines[index] ?? '').replace(/^\s*>\s?/, ''))
        index += 1
      }
      blocks.push({ type: 'blockquote', content: quoteLines.join('\n') })
      continue
    }

    const firstListItem = parseListItem(line)
    if (firstListItem) {
      const items: ListItem[] = []
      const ordered = firstListItem.ordered
      while (index < lines.length) {
        const next = parseListItem(lines[index] ?? '')
        if (!next || next.ordered !== ordered) break
        items.push({ content: next.content, checked: next.checked })
        index += 1
      }
      blocks.push({ type: 'list', ordered, items })
      continue
    }

    const paragraphLines: string[] = []
    while (index < lines.length && !startsBlock(lines[index] ?? '', lines[index + 1] ?? '')) {
      paragraphLines.push(lines[index] ?? '')
      index += 1
    }
    if (paragraphLines.length > 0) blocks.push({ type: 'paragraph', content: paragraphLines.join('\n') })
  }

  return blocks
}

function parseDisplayMathStart(line: string): { close: string; remainder: string } | null {
  const trimmed = line.trim()
  if (trimmed.startsWith('$$')) return { close: '$$', remainder: trimmed.slice(2) }
  if (trimmed.startsWith('\\[')) return { close: '\\]', remainder: trimmed.slice(2) }
  return null
}

function startsBlock(line: string, nextLine: string): boolean {
  if (!line.trim()) return true
  const fence = String.fromCharCode(96, 96, 96)
  return (
    line.startsWith(fence) ||
    Boolean(parseDisplayMathStart(line)) ||
    /^(#{1,4})\s+/.test(line) ||
    /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line) ||
    (line.trim().startsWith('|') && isTableLine(nextLine)) ||
    line.trim().startsWith('>') ||
    Boolean(parseListItem(line))
  )
}

function parseListItem(line: string): (ListItem & { ordered: boolean }) | null {
  const unordered = line.match(/^\s{0,3}[-*+]\s+(\[[ xX]\]\s+)?([\s\S]*)$/)
  if (unordered) {
    return {
      ordered: false,
      ...taskContent(unordered[1], unordered[2] ?? ''),
    }
  }
  const ordered = line.match(/^\s{0,3}\d+[.)]\s+(\[[ xX]\]\s+)?([\s\S]*)$/)
  if (ordered) {
    return {
      ordered: true,
      ...taskContent(ordered[1], ordered[2] ?? ''),
    }
  }
  return null
}

function taskContent(marker: string | undefined, content: string): ListItem {
  if (!marker) return { content }
  return { content, checked: /\[[xX]\]/.test(marker) }
}

function isMarkdownTable(block: string): boolean {
  const lines = block.split('\n').map((line) => line.trim())
  return (
    lines.length >= 2 &&
    lines[0].startsWith('|') &&
    (
      /^\|?[\s:-]+\|[\s|:-]+$/.test(lines[1] ?? '') ||
      lines.every((line) => line.startsWith('|') && line.includes('|'))
    )
  )
}

function isTableLine(line: string): boolean {
  const trimmed = line.trim()
  return trimmed.startsWith('|') && trimmed.includes('|')
}

function hasTasks(items: ListItem[]): boolean {
  return items.some((item) => item.checked != null)
}

function shouldRenderAsMermaid(language: string, code: string): boolean {
  const normalizedLanguage = normalizeCodeLanguage(language)
  if (normalizedLanguage === 'mermaid') return true
  if (!PLAIN_CODE_LANGUAGES.has(normalizedLanguage)) return false
  return looksLikeMermaid(code)
}

function normalizeCodeLanguage(language: string): string {
  return language.trim().split(/\s+/, 1)[0]?.toLowerCase() ?? ''
}

function looksLikeMermaid(code: string): boolean {
  const firstMeaningfulLine = code
    .split('\n')
    .map((line) => line.trim())
    .find((line) => line && !line.startsWith('%%'))
  return firstMeaningfulLine ? MERMAID_DIAGRAM_START.test(firstMeaningfulLine) : false
}

function MarkdownTable({ block, imageContext }: { block: string; imageContext?: ImageContext }) {
  const lines = block.split('\n').map((line) => splitTableRow(line))
  const [headers, maybeSeparator, ...rest] = lines
  const hasSeparator = isSeparatorRow(maybeSeparator ?? [])
  const rows = hasSeparator ? rest : [maybeSeparator, ...rest].filter((row): row is string[] => Boolean(row))
  return (
    <div className="markdown-table-wrap">
      <table className="markdown-table">
        <thead>
          <tr>{headers.map((cell) => <th key={cell}>{renderInline(cell, imageContext)}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => <td key={cellIndex}>{renderInline(cell, imageContext)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function splitTableRow(line: string): string[] {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim())
}

function isSeparatorRow(row: string[]): boolean {
  return row.length > 0 && row.every((cell) => /^:?-{2,}:?$/.test(cell) || cell === '')
}

function renderInline(text: string, imageContext?: ImageContext): ReactNode[] {
  const parts: ReactNode[] = []
  const tick = String.fromCharCode(96)
  const pattern = new RegExp(
    '(' +
      '!\\[[^\\]]*\\]\\([^)]+\\)' +
      '|' + '\\[[^\\]]+\\]\\([^)]+\\)' +
      '|' + tick + '[^' + tick + ']+' + tick +
      '|' + '\\\\([^\\n]+?\\\\)' +
      '|' + '\\$([^\\s$\\n][^$\\n]*?[^\\s$\\n]|\\S)\\$' +
      '|\\*\\*[^*]+\\*\\*' +
      '|~~[^~]+~~' +
      '|(?<!\\*)\\*[^*]+\\*(?!\\*)' +
      '|\\bhttps?:\\/\\/[^\\s<>)\\]]+' +
      '|\\n' +
    ')',
    'g',
  )
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index))
    const value = match[0]
    if (value.startsWith(tick)) {
      parts.push(<code className="markdown-inline-code" key={parts.length}>{value.slice(1, -1)}</code>)
    } else if (value.startsWith('\\(') && value.endsWith('\\)')) {
      parts.push(<MathRenderer key={parts.length} source={value.slice(2, -2)} />)
    } else if (value.startsWith('$') && value.endsWith('$')) {
      parts.push(<MathRenderer key={parts.length} source={value.slice(1, -1)} />)
    } else if (value.startsWith('![')) {
      const image = parseImage(value)
      const src = safeImageSrc(image.src)
      const alt = image.alt || imageAltFromUrl(src || '')
      const imageIndex = imageContext?.images.findIndex((item) => item.src === src && item.alt === alt) ?? -1
      parts.push(src ? (
        <button type="button" className="markdown-image" key={parts.length} onClick={() => imageContext?.openImage(imageIndex >= 0 ? imageIndex : 0)}>
          <img src={src} alt={alt} loading="lazy" />
          {alt ? <span>{alt}</span> : null}
        </button>
      ) : value)
    } else if (value.startsWith('[')) {
      const link = parseLink(value)
      parts.push(
        <a href={safeHref(link.href)} key={parts.length} rel="noreferrer" target="_blank">
          {link.label}
        </a>,
      )
    } else if (value.startsWith('~~')) {
      parts.push(<del key={parts.length}>{renderInline(value.slice(2, -2), imageContext)}</del>)
    } else if (value.startsWith('*') && !value.startsWith('**')) {
      parts.push(<em key={parts.length}>{renderInline(value.slice(1, -1), imageContext)}</em>)
    } else if (/^https?:\/\//i.test(value)) {
      const src = safeImageSrc(value)
      if (src) {
        const imageIndex = imageContext?.images.findIndex((item) => item.src === src) ?? -1
        parts.push(
          <button type="button" className="markdown-image markdown-image-url" key={parts.length} onClick={() => imageContext?.openImage(imageIndex >= 0 ? imageIndex : 0)}>
            <img src={src} alt={imageAltFromUrl(src)} loading="lazy" />
            <span>{imageAltFromUrl(src)}</span>
          </button>,
        )
      } else {
        parts.push(
          <a href={safeHref(value)} key={parts.length} rel="noreferrer" target="_blank">
            {value}
          </a>,
        )
      }
    } else if (value === '\n') {
      parts.push(<br key={parts.length} />)
    } else {
      parts.push(<strong key={parts.length}>{renderInline(value.slice(2, -2), imageContext)}</strong>)
    }
    lastIndex = match.index + value.length
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex))
  return parts
}

function MarkdownImagePreview({ images, activeIndex, onClose, onSelect }: { images: MarkdownImage[]; activeIndex: number; onClose: () => void; onSelect: (index: number) => void }) {
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
    <div className="markdown-image-backdrop" role="dialog" aria-modal="true" aria-label={activeImage.alt || 'Image preview'}>
      <div className="markdown-image-modal">
        <header>
          <div>
            <strong>{activeImage.alt || imageAltFromUrl(activeImage.src)}</strong>
            <span>{activeIndex + 1} / {images.length}</span>
          </div>
          <button type="button" aria-label="Close image preview" onClick={onClose}>
            <X aria-hidden="true" size={16} />
          </button>
        </header>
        <div className="markdown-image-stage">
          {images.length > 1 ? (
            <button type="button" aria-label="Previous image" onClick={() => onSelect((activeIndex - 1 + images.length) % images.length)}>
              <ChevronLeft aria-hidden="true" size={18} />
            </button>
          ) : null}
          <img src={activeImage.src} alt={activeImage.alt || imageAltFromUrl(activeImage.src)} />
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

function extractMarkdownImages(text: string): MarkdownImage[] {
  const images: MarkdownImage[] = []
  const seen = new Set<string>()
  const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)|\bhttps?:\/\/[^\s<>)\]]+/g
  let match: RegExpExecArray | null
  while ((match = imagePattern.exec(text)) !== null) {
    const src = safeImageSrc(match[2] ?? match[0])
    if (!src) continue
    const alt = match[1] || imageAltFromUrl(src)
    const key = src + '\n' + alt
    if (seen.has(key)) continue
    seen.add(key)
    images.push({ src, alt })
  }
  return images
}

function parseImage(value: string): { alt: string; src: string } {
  const match = value.match(/^!\[([^\]]*)\]\(([^)]+)\)$/)
  return { alt: match?.[1] ?? '', src: match?.[2] ?? '' }
}

function parseLink(value: string): { label: string; href: string } {
  const match = value.match(/^\[([^\]]+)\]\(([^)]+)\)$/)
  return { label: match?.[1] ?? value, href: match?.[2] ?? '#' }
}

function safeImageSrc(src: string): string | null {
  const trimmed = src.trim()
  if (/^data:image\/(?:png|jpe?g|gif|webp|svg\+xml);base64,/i.test(trimmed)) return trimmed
  if (/^(https?:|\/(?!\/)|\.\/|\.\.\/)/i.test(trimmed) && isImageUrl(trimmed)) return trimmed
  return null
}

function isImageUrl(value: string): boolean {
  return /\.(?:png|jpe?g|gif|webp|svg)(?:[?#].*)?$/i.test(value)
}

function imageAltFromUrl(src: string): string {
  const clean = src.split(/[?#]/, 1)[0] || src
  return clean.split('/').filter(Boolean).pop() || 'image'
}

function safeHref(href: string): string {
  const trimmed = href.trim()
  if (/^(https?:|mailto:|#|\/(?!\/)|\.\/|\.\.\/)/i.test(trimmed)) return trimmed
  return '#'
}
