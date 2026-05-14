export type InlineSegment =
  | { kind: 'text'; text: string }
  | { kind: 'bold'; text: string }
  | { kind: 'italic'; text: string }
  | { kind: 'code'; text: string }

export type MarkdownBlock =
  | { kind: 'paragraph'; segments: InlineSegment[] }
  | { kind: 'heading'; depth: number; segments: InlineSegment[] }
  | { kind: 'list'; ordered: boolean; start: number; items: InlineSegment[][] }
  | { kind: 'hr' }
  | { kind: 'table'; headers: InlineSegment[][]; rows: InlineSegment[][][] }
  | { kind: 'code'; text: string; language: string | null }

export function parseMarkdownLite(input: string): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = []
  const lines = input.split('\n')
  let codeBuffer: string[] | null = null
  let codeLanguage: string | null = null
  let paragraph: string[] = []
  let listState:
    | {
        ordered: boolean
        start: number
        items: string[]
      }
    | null = null

  const flushParagraph = () => {
    if (paragraph.length === 0) {
      return
    }
    blocks.push({ kind: 'paragraph', segments: parseInline(paragraph.join('\n')) })
    paragraph = []
  }

  const flushList = () => {
    if (!listState || listState.items.length === 0) {
      listState = null
      return
    }
    blocks.push({
      kind: 'list',
      ordered: listState.ordered,
      start: listState.start,
      items: listState.items.map((item) => parseInline(item))
    })
    listState = null
  }

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index] ?? ''
    if (line.trim().startsWith('```')) {
      if (codeBuffer) {
        blocks.push({
          kind: 'code',
          text: codeBuffer.join('\n'),
          language: codeLanguage
        })
        codeBuffer = null
        codeLanguage = null
      } else {
        flushParagraph()
        flushList()
        codeBuffer = []
        codeLanguage = line.trim().slice(3).trim() || null
      }
      continue
    }
    if (codeBuffer) {
      codeBuffer.push(line)
      continue
    }
    if (!line.trim()) {
      flushParagraph()
      flushList()
      continue
    }

    if (isHorizontalRule(line)) {
      flushParagraph()
      flushList()
      blocks.push({ kind: 'hr' })
      continue
    }

    const nextLine = lines[index + 1] ?? ''
    if (looksLikeTableHeader(line) && isTableSeparator(nextLine)) {
      flushParagraph()
      flushList()
      const headers = splitTableRow(line).map((cell) => parseInline(cell))
      const rows: InlineSegment[][][] = []
      index += 2
      while (index < lines.length) {
        const rowLine = lines[index] ?? ''
        if (!rowLine.trim() || !looksLikeTableRow(rowLine)) {
          index -= 1
          break
        }
        rows.push(splitTableRow(rowLine).map((cell) => parseInline(cell)))
        index += 1
      }
      blocks.push({ kind: 'table', headers, rows })
      continue
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/)
    if (headingMatch) {
      flushParagraph()
      flushList()
      blocks.push({
        kind: 'heading',
        depth: headingMatch[1]?.length ?? 1,
        segments: parseInline(headingMatch[2] ?? '')
      })
      continue
    }

    const orderedMatch = line.match(/^\s*(\d+)\.\s+(.*)$/)
    const bulletMatch = orderedMatch ? null : line.match(/^\s*[-*+]\s+(.*)$/)
    if (orderedMatch || bulletMatch) {
      flushParagraph()
      const ordered = Boolean(orderedMatch)
      const start = ordered ? Number.parseInt(orderedMatch?.[1] ?? '1', 10) || 1 : 1
      const text = ordered ? (orderedMatch?.[2] ?? '') : (bulletMatch?.[1] ?? '')
      if (!listState || listState.ordered !== ordered) {
        flushList()
        listState = { ordered, start, items: [] }
      }
      listState.items.push(text)
      continue
    }

    if (listState && /^\s{2,}\S/.test(line)) {
      const last = listState.items.length - 1
      if (last >= 0) {
        listState.items[last] = `${listState.items[last] ?? ''} ${line.trim()}`
        continue
      }
    }

    flushList()
    paragraph.push(line)
  }

  if (codeBuffer) {
    blocks.push({
      kind: 'code',
      text: codeBuffer.join('\n'),
      language: codeLanguage
    })
  }
  flushList()
  flushParagraph()
  return blocks
}

export function parseInline(input: string): InlineSegment[] {
  const segments: InlineSegment[] = []
  let cursor = 0

  while (cursor < input.length) {
    const boldStart = input.indexOf('**', cursor)
    const codeStart = input.indexOf('`', cursor)
    const italicStart = findItalicStart(input, cursor)
    const candidates = [
      { kind: 'bold' as const, start: boldStart },
      { kind: 'italic' as const, start: italicStart },
      { kind: 'code' as const, start: codeStart }
    ].filter((candidate) => candidate.start >= 0)
    const next = candidates.sort((left, right) => left.start - right.start)[0]
    const nextStart = next?.start ?? -1

    if (nextStart === -1) {
      pushText(segments, input.slice(cursor))
      break
    }
    if (nextStart > cursor) {
      pushText(segments, input.slice(cursor, nextStart))
    }

    if (next?.kind === 'bold') {
      const end = input.indexOf('**', nextStart + 2)
      if (end === -1) {
        pushText(segments, input.slice(nextStart))
        break
      }
      segments.push({ kind: 'bold', text: input.slice(nextStart + 2, end) })
      cursor = end + 2
      continue
    }

    if (next?.kind === 'italic') {
      const end = input.indexOf('*', nextStart + 1)
      if (end === -1) {
        pushText(segments, input.slice(nextStart))
        break
      }
      segments.push({ kind: 'italic', text: input.slice(nextStart + 1, end) })
      cursor = end + 1
      continue
    }

    if (next?.kind === 'code') {
      const end = input.indexOf('`', nextStart + 1)
      if (end === -1) {
        pushText(segments, input.slice(nextStart))
        break
      }
      segments.push({ kind: 'code', text: input.slice(nextStart + 1, end) })
      cursor = end + 1
    }
  }

  return segments
}

function findItalicStart(input: string, cursor: number): number {
  for (let index = cursor; index < input.length; index++) {
    if (input[index] !== '*') {
      continue
    }
    if (input[index + 1] === '*') {
      index++
      continue
    }
    if (index > 0 && input[index - 1] === '*') {
      continue
    }
    return index
  }
  return -1
}

function isHorizontalRule(line: string): boolean {
  return /^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)
}

function pushText(segments: InlineSegment[], text: string): void {
  if (!text) {
    return
  }
  const last = segments.at(-1)
  if (last?.kind === 'text') {
    last.text += text
    return
  }
  segments.push({ kind: 'text', text })
}

function looksLikeTableHeader(line: string): boolean {
  return splitTableRow(line).length >= 2
}

function looksLikeTableRow(line: string): boolean {
  return splitTableRow(line).length >= 2
}

function isTableSeparator(line: string): boolean {
  return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+(?:\s*:?-{3,}:?\s*)\|?\s*$/.test(line)
}

function splitTableRow(line: string): string[] {
  const trimmed = line.trim()
  if (!trimmed.includes('|')) {
    return []
  }
  const normalized = trimmed.replace(/^\|/, '').replace(/\|$/, '')
  return normalized.split('|').map((cell) => cell.trim())
}
